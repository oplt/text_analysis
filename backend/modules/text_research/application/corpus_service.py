"""Research corpus lifecycle: creation, RAG document linking, metadata, and
immutable canonical source text for research analysis.

Research text is NEVER reconstructed by joining overlapping RAG retrieval
chunks. Analysis starts from a persisted ``CanonicalResearchSource``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from fastapi import HTTPException

from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import (
    CanonicalResearchSource,
    CorpusDocument,
    ResearchCorpus,
    dumps,
)
from backend.modules.text_research.infrastructure.canonical_text import (
    CanonicalBuildResult,
    build_canonical_from_full_text,
    build_canonical_from_pages,
    sha256_bytes,
)

logger = logging.getLogger(__name__)

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
        canonical_full_text: str | None = None,
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
        await self._ensure_canonical_for_document(
            document,
            rag_document=rag_document,
            language=language,
            full_text=canonical_full_text,
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
        language: str | None = None,
        search: str | None = None,
        sort_by: str = "created_at",
        sort_dir: str = "asc",
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
            language=language,
            search=search,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )

    async def metadata_facets(
        self, corpus_id: str, *, user_id: str
    ) -> dict[str, list[dict[str, str | int]]]:
        """Return SQL-aggregated metadata facets for an accessible corpus."""
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return await self.repo.list_document_metadata_facets(corpus_id)

    async def paginate_documents(
        self,
        corpus_id: str,
        *,
        user_id: str,
        organization: str | None = None,
        publication_year: int | None = None,
        region: str | None = None,
        cultural_sphere: str | None = None,
        language: str | None = None,
        search: str | None = None,
        sort_by: str = "created_at",
        sort_dir: str = "asc",
        limit: int,
        offset: int = 0,
    ) -> tuple[list[CorpusDocument], int]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return await self.repo.paginate_documents(
            corpus_id,
            organization=organization,
            publication_year=publication_year,
            region=region,
            cultural_sphere=cultural_sphere,
            language=language,
            search=search,
            sort_by=sort_by,
            sort_dir=sort_dir,
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
        documents = await self.repo.list_documents_by_ids(list(dict.fromkeys(document_ids)))
        updated = [document for document in documents if document.corpus_id == corpus_id]
        for document in updated:
            for key, value in fields.items():
                if value is not None:
                    setattr(document, key, value)
        await self.db.flush()
        await self.db.commit()
        return updated

    async def import_metadata_csv(
        self, corpus_id: str, *, user_id: str, csv_content: str
    ) -> dict[str, Any]:
        """Bulk metadata import. CSV must contain a `rag_document_id` (or
        `document_id`) column plus any subset of the known metadata columns."""
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        rows = list(csv.DictReader(io.StringIO(csv_content)))
        document_ids = [(row.get("document_id") or "").strip() for row in rows]
        rag_document_ids = [(row.get("rag_document_id") or "").strip() for row in rows]
        documents = await self.repo.list_documents(corpus_id)
        documents_by_id = {document.id: document for document in documents}
        documents_by_rag_id = {document.rag_document_id: document for document in documents}
        updated = 0
        errors: list[str] = []
        row_number = 1
        for row, document_id, rag_document_id in zip(rows, document_ids, rag_document_ids, strict=True):
            row_number += 1
            document = documents_by_id.get(document_id) if document_id else documents_by_rag_id.get(rag_document_id)
            if document is None:
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
                for key, value in fields.items():
                    setattr(document, key, value)
                updated += 1
        await self.db.flush()
        await self.db.commit()
        return {"updated": updated, "errors": errors, "rows_processed": row_number - 1}

    async def delete_document(self, document_id: str, *, user_id: str) -> None:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        await self.repo.delete_document(document)
        await self.db.commit()

    # ------------------------------------------------------------------
    # Canonical research source (immutable; never from overlapping RAG chunks)
    # ------------------------------------------------------------------

    async def get_canonical_source(
        self, document_id: str, *, user_id: str
    ) -> CanonicalResearchSource:
        return await self.ensure_canonical_source(document_id, user_id=user_id)

    async def get_source_text(self, document_id: str, *, user_id: str) -> str:
        """Return immutable canonical research text for analysis/segmentation."""
        source = await self.ensure_canonical_source(document_id, user_id=user_id)
        return source.canonical_text

    async def ensure_canonical_source(
        self,
        document_id: str,
        *,
        user_id: str,
        full_text: str | None = None,
    ) -> CanonicalResearchSource:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        existing = await self.repo.get_canonical_source(document.id)
        if existing is not None:
            return existing

        rag_repo = RagRepository(self.db)
        rag_document = await rag_repo.get_document(document.rag_document_id)
        if rag_document is None:
            raise HTTPException(
                status_code=422,
                detail="Linked RAG document missing; cannot build canonical research source.",
            )
        return await self._ensure_canonical_for_document(
            document,
            rag_document=rag_document,
            language=document.language,
            full_text=full_text,
        )

    async def _ensure_canonical_for_document(
        self,
        document: CorpusDocument,
        *,
        rag_document: Any,
        language: str | None,
        full_text: str | None = None,
    ) -> CanonicalResearchSource:
        existing = await self.repo.get_canonical_source(document.id)
        if existing is not None:
            return existing

        build = await self._build_canonical_result(
            rag_document,
            language=language or document.language,
            full_text=full_text,
        )
        source = CanonicalResearchSource(
            corpus_document_id=document.id,
            canonical_text=build.text,
            canonical_text_checksum=build.text_checksum,
            raw_extracted_text=build.text,
            raw_extracted_checksum=build.text_checksum,
            original_file_checksum=build.original_file_checksum,
            parser_name=build.parser_name,
            parser_version=build.parser_version,
            extracted_at=build.extracted_at,
            source_rag_document_id=rag_document.id,
            source_storage_path=getattr(rag_document, "storage_path", None),
            source_filename=getattr(rag_document, "original_filename", None),
            language=build.language,
            page_provenance_json=dumps(build.page_provenance),
            transformation_metadata_json=dumps(
                {
                    **(build.transformation_metadata or {}),
                    "raw_equals_canonical": True,
                    "cleaning_applied": False,
                }
            ),
        )
        return await self.repo.create_canonical_source(source)

    async def _build_canonical_result(
        self,
        rag_document: Any,
        *,
        language: str | None,
        full_text: str | None = None,
    ) -> CanonicalBuildResult:
        if full_text is not None:
            return build_canonical_from_full_text(
                full_text,
                language=language,
                source_file_reference=getattr(rag_document, "storage_path", None),
                extra_transformation={"source": "explicit_full_text"},
            )

        storage_path = getattr(rag_document, "storage_path", None)
        if storage_path:
            try:
                from backend.modules.rag.application.document_parser_service import (
                    DocumentParserService,
                )
                from backend.modules.rag.infrastructure.file_storage_adapter import (
                    FileStorageAdapter,
                )

                content = await FileStorageAdapter().download_document(storage_path)
                parsed = await DocumentParserService().parse_bytes(
                    content=content,
                    filename=getattr(rag_document, "original_filename", "document"),
                    content_type=getattr(rag_document, "content_type", "application/octet-stream"),
                    metadata={"document_id": rag_document.id},
                )
                if not parsed:
                    raise HTTPException(
                        status_code=422,
                        detail="Parser returned no text for canonical research source.",
                    )
                pages = [
                    {
                        "content": page.content,
                        "page_number": page.page_number,
                        "section_heading": (page.metadata or {}).get("section_heading"),
                    }
                    for page in parsed
                ]
                return build_canonical_from_pages(
                    pages,
                    language=language,
                    original_file_checksum=sha256_bytes(content),
                    source_file_reference=storage_path,
                    extra_transformation={"source": "storage_reparse"},
                )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Canonical reparse from storage failed for rag_document=%s: %s",
                    rag_document.id,
                    exc,
                )
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Unable to build canonical research source from original file. "
                        "Research analysis does not reconstruct overlapping RAG chunks."
                    ),
                ) from exc

        raise HTTPException(
            status_code=422,
            detail=(
                "No canonical research source available. Provide the original file "
                "(storage) or explicit full text. Joining RAG retrieval chunks is forbidden."
            ),
        )
