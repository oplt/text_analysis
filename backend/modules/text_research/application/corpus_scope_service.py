"""Resolve authoritative corpus evidence scope for Ask Corpus (I1/I2/I3/I5)."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.rag.application.evidence_revision import (
    build_evidence_revision_hash,
    chunk_revision_identity,
)
from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AssistantScopeMode
from backend.modules.text_research.domain.models import (
    CorpusDocument,
    ResearchAssistantScopeEvent,
    ResearchAssistantThread,
    ResearchCorpus,
)


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
    evidence_revision_hash: str | None
    total_documents: int
    indexed_count: int
    unavailable_count: int
    unavailable_corpus_document_ids: list[str] = field(default_factory=list)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    scope_mode: str = AssistantScopeMode.FIXED.value

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
            "evidence_revision_hash": self.evidence_revision_hash,
            "total_documents": self.total_documents,
            "indexed_count": self.indexed_count,
            "unavailable_count": self.unavailable_count,
            "unavailable_corpus_document_ids": self.unavailable_corpus_document_ids,
            "unavailable_reasons": self.unavailable_reasons,
            "warnings": self.warnings,
            "scope_mode": self.scope_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CorpusScopeSnapshot:
        return cls(
            corpus_id=str(value["corpus_id"]),
            project_id=str(value["project_id"]),
            corpus_name=str(value.get("corpus_name", "")),
            rag_document_ids=list(value.get("rag_document_ids", [])),
            corpus_document_ids=list(value.get("corpus_document_ids", [])),
            indexed_rag_document_ids=list(value.get("indexed_rag_document_ids", [])),
            unavailable_rag_document_ids=list(value.get("unavailable_rag_document_ids", [])),
            scope_hash=str(value["scope_hash"]),
            index_version=value.get("index_version"),
            retrieval_version=value.get("retrieval_version"),
            evidence_revision_hash=value.get("evidence_revision_hash"),
            total_documents=int(value.get("total_documents", 0)),
            indexed_count=int(value.get("indexed_count", 0)),
            unavailable_count=int(value.get("unavailable_count", 0)),
            unavailable_corpus_document_ids=list(
                value.get("unavailable_corpus_document_ids", [])
            ),
            unavailable_reasons=dict(value.get("unavailable_reasons", {})),
            warnings=list(value.get("warnings", [])),
            scope_mode=str(value.get("scope_mode", AssistantScopeMode.FIXED.value)),
        )


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
        # Scope resolution must not silently truncate large corpora. Keep each
        # query bounded while materializing the complete, deterministic scope.
        documents: list[CorpusDocument] = []
        offset = 0
        while True:
            page = await self.repo.list_documents(
                corpus.id,
                limit=DEFAULT_PAGE_LIMIT,
                offset=offset,
            )
            if not page:
                break
            documents.extend(page)
            offset += len(page)
            if len(page) < DEFAULT_PAGE_LIMIT:
                break
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
        rag_docs = await self.rag_repo.get_documents_by_ids(
            rag_ids,
            user_id=None,
            include_deleted=True,
        )
        by_id = {d.id: d for d in rag_docs}

        indexed: set[str] = set()
        unavailable: set[str] = set()
        unavailable_corpus_document_ids: list[str] = []
        unavailable_reasons: dict[str, str] = {}
        indexed_corpus_document_count = 0
        for corpus_document in documents:
            rag_id = corpus_document.rag_document_id
            rag_document = by_id.get(rag_id) if rag_id else None
            reason: str | None = None
            if not rag_id:
                reason = "missing_rag_document_id"
            elif rag_document is None:
                reason = "rag_source_missing"
            elif (
                rag_document.deleted_at is not None
                or rag_document.status == DocumentStatus.DELETED.value
            ):
                reason = "rag_source_deleted"
            elif rag_document.status == DocumentStatus.FAILED.value:
                reason = "indexing_failed"
            elif rag_document.status == DocumentStatus.UPLOADED.value:
                reason = "uploaded_not_indexed"
            elif rag_document.status != DocumentStatus.INDEXED.value:
                reason = "not_indexed"

            if reason is None:
                indexed.add(rag_id)
                indexed_corpus_document_count += 1
            else:
                if rag_id:
                    unavailable.add(rag_id)
                unavailable_corpus_document_ids.append(corpus_document.id)
                unavailable_reasons[corpus_document.id] = reason

        if not documents:
            warnings.append("Corpus has no documents")
        elif not indexed:
            warnings.append("No indexed documents available for retrieval")
        elif unavailable_corpus_document_ids:
            warnings.append(
                f"{len(unavailable_corpus_document_ids)} corpus document(s) "
                "unavailable or still indexing"
            )

        # Evidence allow-list = indexed only (I1/I5); empty list if none
        allow_list = sorted(indexed)
        revision_chunks = await self.rag_repo.list_evidence_revision_chunks(allow_list)
        chunks_by_document: dict[str, list[dict[str, Any]]] = {}
        for chunk in revision_chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(
                chunk_revision_identity(chunk)
            )
        document_revisions: list[dict[str, Any]] = []
        for rag_id in allow_list:
            document = by_id[rag_id]
            try:
                metadata = json.loads(document.metadata_json or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            document_revisions.append(
                {
                    "rag_document_id": rag_id,
                    "content_fingerprint": metadata.get("checksum_sha256"),
                    "chunks": chunks_by_document.get(rag_id, []),
                }
            )
        evidence_revision_hash = build_evidence_revision_hash(
            corpus_id=corpus.id,
            project_id=corpus.project_id,
            document_revisions=document_revisions,
            config=self.rag_config,
        )
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
            evidence_revision_hash=evidence_revision_hash,
            total_documents=len(documents),
            indexed_count=indexed_corpus_document_count,
            unavailable_count=len(unavailable_corpus_document_ids),
            unavailable_corpus_document_ids=sorted(unavailable_corpus_document_ids),
            unavailable_reasons=unavailable_reasons,
            warnings=warnings,
        )

    async def resolve_for_thread(
        self, thread: ResearchAssistantThread, *, user_id: str
    ) -> CorpusScopeSnapshot:
        """Resolve a thread's configured scope without widening fixed threads."""
        await self.get_corpus_or_404(
            thread.corpus_id,
            user_id=user_id,
            project_id=thread.project_id,
        )
        mode = getattr(thread, "scope_mode", None) or AssistantScopeMode.FIXED.value
        if mode not in {item.value for item in AssistantScopeMode}:
            raise HTTPException(status_code=409, detail="Assistant thread scope mode is invalid")
        if mode == AssistantScopeMode.LIVE.value:
            snapshot = await self.resolve(corpus_id=thread.corpus_id, user_id=user_id)
            snapshot.scope_mode = mode
            return snapshot

        raw_snapshot = getattr(thread, "scope_snapshot_json", None)
        if not isinstance(raw_snapshot, str) or not raw_snapshot:
            conversation = await self.rag_repo.get_conversation(thread.rag_conversation_id)
            raw_snapshot = conversation.scope_snapshot_json if conversation else None
        if not isinstance(raw_snapshot, str) or not raw_snapshot:
            raise HTTPException(
                status_code=409,
                detail="This thread has no persisted scope; start a new assistant thread",
            )
        try:
            snapshot = CorpusScopeSnapshot.from_dict(json.loads(raw_snapshot))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=409, detail="Assistant thread scope is invalid"
            ) from exc
        if snapshot.corpus_id != thread.corpus_id or snapshot.project_id != thread.project_id:
            raise HTTPException(status_code=409, detail="Assistant thread scope boundary mismatch")
        snapshot.scope_mode = mode
        return snapshot

    async def record_scope_event(
        self,
        *,
        thread: ResearchAssistantThread,
        actor_id: str,
        action: str,
        scope: CorpusScopeSnapshot,
        previous_scope_hash: str | None = None,
        reason: str | None = None,
    ) -> ResearchAssistantScopeEvent:
        event = ResearchAssistantScopeEvent(
            id=str(uuid4()),
            thread_id=thread.id,
            actor_id=actor_id,
            action=action,
            scope_mode=scope.scope_mode,
            corpus_id=scope.corpus_id,
            project_id=scope.project_id,
            previous_scope_hash=previous_scope_hash,
            new_scope_hash=scope.scope_hash,
            evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
            rag_document_ids_json=json.dumps(scope.rag_document_ids, ensure_ascii=True),
            corpus_document_ids_json=json.dumps(scope.corpus_document_ids, ensure_ascii=True),
            reason=reason,
            created_at=datetime.now(UTC),
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def list_scope_events(
        self, thread_id: str, *, user_id: str
    ) -> list[ResearchAssistantScopeEvent]:
        thread = await self.db.get(ResearchAssistantThread, thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Assistant conversation not found")
        await self.ensure_project_access(user_id=user_id, project_id=thread.project_id)
        result = await self.db.execute(
            select(ResearchAssistantScopeEvent)
            .where(ResearchAssistantScopeEvent.thread_id == thread_id)
            .order_by(ResearchAssistantScopeEvent.created_at.asc())
        )
        return list(result.scalars().all())

    async def update_thread_scope(
        self,
        *,
        thread: ResearchAssistantThread,
        user_id: str,
        document_subset: list[str] | None,
        scope_mode: str | None,
        reason: str | None,
    ) -> CorpusScopeSnapshot:
        await self.get_corpus_or_404(
            thread.corpus_id,
            user_id=user_id,
            project_id=thread.project_id,
        )
        mode = scope_mode or getattr(thread, "scope_mode", AssistantScopeMode.FIXED.value)
        if mode not in {item.value for item in AssistantScopeMode}:
            raise HTTPException(status_code=422, detail="scope_mode must be fixed or live")
        if mode == AssistantScopeMode.LIVE.value and document_subset is not None:
            raise HTTPException(
                status_code=422,
                detail="A live thread follows the current corpus and cannot use a document subset",
            )
        previous_hash = None
        old_snapshot = getattr(thread, "scope_snapshot_json", None)
        if isinstance(old_snapshot, str) and old_snapshot:
            with suppress(json.JSONDecodeError):
                previous_hash = json.loads(old_snapshot).get("scope_hash")
        scope = await self.resolve(
            corpus_id=thread.corpus_id,
            user_id=user_id,
            document_subset=document_subset,
        )
        scope.scope_mode = mode
        thread.scope_mode = mode
        thread.evidence_revision_hash = getattr(scope, "evidence_revision_hash", None)
        thread.scope_snapshot_json = json.dumps(scope.to_dict(), ensure_ascii=True)
        conversation = await self.rag_repo.get_conversation(thread.rag_conversation_id)
        if conversation is not None:
            conversation.scope_snapshot_json = thread.scope_snapshot_json
        await self.record_scope_event(
            thread=thread,
            actor_id=user_id,
            action="scope_updated",
            scope=scope,
            previous_scope_hash=previous_hash,
            reason=reason,
        )
        await self.db.flush()
        return scope
