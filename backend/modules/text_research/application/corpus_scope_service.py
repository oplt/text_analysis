"""Resolve authoritative corpus evidence scope for Ask Corpus (I1/I2/I3/I5)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import CorpusDocument, ResearchCorpus
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True)
class CorpusScopeSnapshot:
    corpus_id: str
    project_id: str
    corpus_name: str
    rag_document_ids: list[str]
    corpus_document_ids: list[str]
    indexed_rag_document_ids: list[str]
    unavailable_rag_document_ids: list[str]
    scope_hash: str
    index_version: str
    retrieval_version: str
    total_documents: int
    indexed_count: int
    unavailable_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "corpus_id": self.corpus_id,
            "project_id": self.project_id,
            "corpus_name": self.corpus_name,
            "rag_document_ids": self.rag_document_ids,
            "corpus_document_ids": self.corpus_document_ids,
            "indexed_rag_document_ids": self.indexed_rag_document_ids,
            "unavailable_rag_document_ids": self.unavailable_rag_document_ids,
            "scope_hash": self.scope_hash,
            "index_version": self.index_version,
            "retrieval_version": self.retrieval_version,
            "total_documents": self.total_documents,
            "indexed_count": self.indexed_count,
            "unavailable_count": self.unavailable_count,
            "warnings": self.warnings,
        }


def _scope_hash(rag_document_ids: list[str], *, corpus_id: str, project_id: str) -> str:
    payload = json.dumps(
        {
            "corpus_id": corpus_id,
            "project_id": project_id,
            "rag_document_ids": sorted(rag_document_ids),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class CorpusScopeService(ResearchAccessMixin):
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.rag_repo = RagRepository(db)
        self.rag_config = RagConfig.from_settings()

    async def resolve(
        self,
        *,
        corpus_id: str,
        user_id: str,
        document_subset: list[str] | None = None,
    ) -> CorpusScopeSnapshot:
        """Always returns a concrete document_ids list (never None) — I1."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus.id, limit=10_000, offset=0)
        return await self._build_snapshot(
            corpus=corpus,
            documents=documents,
            document_subset=document_subset,
        )

    async def _build_snapshot(
        self,
        *,
        corpus: ResearchCorpus,
        documents: list[CorpusDocument],
        document_subset: list[str] | None,
    ) -> CorpusScopeSnapshot:
        warnings: list[str] = []
        if document_subset is not None:
            subset = set(document_subset)
            membership_ids = {d.id for d in documents}
            unknown = subset - membership_ids
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail="Selected documents are not members of this corpus",
                )
            documents = [d for d in documents if d.id in subset]

        rag_ids = [d.rag_document_id for d in documents if d.rag_document_id]
        corpus_doc_ids = [d.id for d in documents]

        # Authoritative RAG status (batched) — I2: no uploader filter
        rag_docs = await self.rag_repo.get_documents_by_ids(rag_ids, user_id=None)
        by_id = {d.id: d for d in rag_docs}

        indexed: list[str] = []
        unavailable: list[str] = []
        for rag_id in sorted(set(rag_ids)):
            doc = by_id.get(rag_id)
            if (
                doc is None
                or doc.deleted_at is not None
                or doc.status != DocumentStatus.INDEXED.value
            ):
                unavailable.append(rag_id)
            else:
                indexed.append(rag_id)

        if not documents:
            warnings.append("Corpus has no documents")
        elif not indexed:
            warnings.append("No indexed documents available for retrieval")
        elif unavailable:
            warnings.append(
                f"{len(unavailable)} corpus document(s) unavailable or still indexing"
            )

        # Evidence allow-list = indexed only (I1/I5); empty list if none
        allow_list = sorted(indexed)
        return CorpusScopeSnapshot(
            corpus_id=corpus.id,
            project_id=corpus.project_id,
            corpus_name=corpus.name,
            rag_document_ids=allow_list,
            corpus_document_ids=sorted(corpus_doc_ids),
            indexed_rag_document_ids=allow_list,
            unavailable_rag_document_ids=sorted(unavailable),
            scope_hash=_scope_hash(
                allow_list, corpus_id=corpus.id, project_id=corpus.project_id
            ),
            index_version=self.rag_config.index_version,
            retrieval_version=self.rag_config.retrieval_algorithm_version,
            total_documents=len(documents),
            indexed_count=len(indexed),
            unavailable_count=len(unavailable),
            warnings=warnings,
        )
