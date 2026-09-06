"""Research corpus lifecycle: creation, RAG document linking, metadata, and
reconstruction of source text from RAG chunks."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import HTTPException

from backend.core.pagination import MAX_PAGE_LIMIT
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import CorpusDocument, ResearchCorpus

_CSV_METADATA_COLUMNS = (
    "title",
    "organization",
    "organization_type",
    "publication_year",
    "publication_type",
    "country",
    "region",
    "cultural_sphere",
    "language",
    "education_level",
    "source_url",
    "research_notes",
)


class CorpusService(ResearchAccessMixin):
    # ------------------------------------------------------------------
    # Corpus CRUD
    # ------------------------------------------------------------------

    async def create_corpus(
        self, *, project_id: str, user_id: str, name: str, description: str | None
    ) -> ResearchCorpus:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        corpus = await self.repo.create_corpus(
            project_id=project_id, name=name, description=description, created_by=user_id
        )
        await self.db.commit()
        return corpus

    async def list_corpora(self, *, project_id: str, user_id: str) -> list[ResearchCorpus]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_corpora(project_id)

    async def get_corpus(self, corpus_id: str, *, user_id: str) -> ResearchCorpus:
        return await self.get_corpus_or_404(corpus_id, user_id=user_id)

    async def update_corpus(
        self,
        corpus_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> ResearchCorpus:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        updated = await self.repo.update_corpus(corpus, name=name, description=description)
        await self.db.commit()
        return updated

    async def delete_corpus(self, corpus_id: str, *, user_id: str) -> None:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        await self.repo.delete_corpus(corpus)
        await self.db.commit()

    # ------------------------------------------------------------------
    # RAG document linking
    # ------------------------------------------------------------------

    async def add_document(
        self,
        corpus_id: str,
        *,
        user_id: str,
        rag_document_id: str,
        title: str | None = None,
        organization: str | None = None,
        organization_type: str | None = None,
        publication_year: int | None = None,
        publication_type: str | None = None,
        country: str | None = None,
        region: str | None = None,
        cultural_sphere: str | None = None,
        language: str | None = None,
        education_level: str | None = None,
        source_url: str | None = None,
        research_notes: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> CorpusDocument:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)

        rag_repo = RagRepository(self.db)
        rag_document = await rag_repo.get_document(rag_document_id)
        if rag_document is None or rag_document.user_id != user_id:
            raise HTTPException(status_code=404, detail="RAG document not found or not accessible")
        if rag_document.project_id and rag_document.project_id != corpus.project_id:
            raise HTTPException(
                status_code=400, detail="RAG document belongs to a different project"
            )

        existing = await self.repo.get_document_by_rag_id(
            corpus_id=corpus_id, rag_document_id=rag_document_id
        )
        if existing is not None:
            raise HTTPException(
                status_code=409, detail="This RAG document is already part of the corpus"
            )

        document = await self.repo.add_document(
            corpus_id=corpus_id,
            rag_document_id=rag_document_id,
            title=title or rag_document.original_filename,
            organization=organization,
            organization_type=organization_type,
            publication_year=publication_year,
            publication_type=publication_type,
            country=country,
            region=region,
            cultural_sphere=cultural_sphere,
            language=language,
            education_level=education_level,
            source_url=source_url,
            research_notes=research_notes,
            metadata_json=json.dumps(extra_metadata or {}, ensure_ascii=True),
        )
        await self.db.commit()
        return document

    async def list_documents(
        self,
        corpus_id: str,
        *,
        user_id: str,
        organization: str | None = None,
        publication_year: int | None = None,
        region: str | None = None,
        cultural_sphere: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CorpusDocument]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return await self.repo.list_documents(
            corpus_id,
            organization=organization,
            publication_year=publication_year,
            region=region,
            cultural_sphere=cultural_sphere,
            limit=limit,
            offset=offset,
        )

    async def get_document(self, document_id: str, *, user_id: str) -> CorpusDocument:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        return document

    async def update_document_metadata(
        self, document_id: str, *, user_id: str, **fields: Any
    ) -> CorpusDocument:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        extra_metadata = fields.pop("extra_metadata", None)
        if extra_metadata is not None:
            fields["metadata_json"] = json.dumps(extra_metadata, ensure_ascii=True)
        updated = await self.repo.update_document(document, **fields)
        await self.db.commit()
        return updated

    async def bulk_update_metadata(
        self,
        corpus_id: str,
        *,
        user_id: str,
        document_ids: list[str],
        fields: dict[str, Any],
    ) -> list[CorpusDocument]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        updated: list[CorpusDocument] = []
        for document_id in document_ids:
            document = await self.repo.get_document(document_id)
            if document is None or document.corpus_id != corpus_id:
                continue
            updated.append(await self.repo.update_document(document, **fields))
        await self.db.commit()
        return updated

    async def import_metadata_csv(
        self, corpus_id: str, *, user_id: str, csv_content: str
    ) -> dict[str, Any]:
        """Bulk metadata import. CSV must contain a `rag_document_id` (or
        `document_id`) column plus any subset of the known metadata columns."""
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        reader = csv.DictReader(io.StringIO(csv_content))
        updated = 0
        errors: list[str] = []
        row_number = 1
        for row in reader:
            row_number += 1
            document_id = (row.get("document_id") or "").strip()
            rag_document_id = (row.get("rag_document_id") or "").strip()
            document: CorpusDocument | None = None
            if document_id:
                document = await self.repo.get_document(document_id)
            elif rag_document_id:
                document = await self.repo.get_document_by_rag_id(
                    corpus_id=corpus_id, rag_document_id=rag_document_id
                )
            if document is None or document.corpus_id != corpus_id:
                errors.append(f"row {row_number}: document not found in this corpus")
                continue

            fields: dict[str, Any] = {}
            for column in _CSV_METADATA_COLUMNS:
                if column in row and row[column] not in (None, ""):
                    value = row[column]
                    if column == "publication_year":
                        try:
                            value = int(value)
                        except ValueError:
                            errors.append(f"row {row_number}: invalid publication_year '{value}'")
                            continue
                    fields[column] = value
            if fields:
                await self.repo.update_document(document, **fields)
                updated += 1
        await self.db.commit()
        return {"updated": updated, "errors": errors, "rows_processed": row_number - 1}

    async def delete_document(self, document_id: str, *, user_id: str) -> None:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        await self.repo.delete_document(document)
        await self.db.commit()

    # ------------------------------------------------------------------
    # Source text reconstruction (RAG chunks -> research source text)
    # ------------------------------------------------------------------

    async def get_source_text(self, document_id: str, *, user_id: str) -> str:
        """Reconstruct the source text of a corpus document from its RAG
        chunks, ordered by `chunk_index`, joined with newlines.

        RAG chunks are retrieval-optimized, not research units — this raw
        reconstruction is only an intermediate step before segmentation.
        """
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        rag_repo = RagRepository(self.db)
        chunks = []
        offset = 0
        while True:
            page, total = await rag_repo.list_chunks_for_document(
                document.rag_document_id, limit=MAX_PAGE_LIMIT, offset=offset
            )
            chunks.extend(page)
            offset += len(page)
            if not page or offset >= total:
                break
        chunks.sort(key=lambda c: c.chunk_index)
        return "\n".join(chunk.content for chunk in chunks)
