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
class CorpusScopeDocumentBinding:
    """Immutable corpus-document to RAG-document provenance within one scope."""

    corpus_document_id: str
    rag_document_id: str | None
    availability: str
    status: str | None = None
    unavailable_reason: str | None = None
    index_revision_id: str | None = None
    document_revision: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "corpus_document_id": self.corpus_document_id,
            "rag_document_id": self.rag_document_id,
            "availability": self.availability,
            "status": self.status,
            "unavailable_reason": self.unavailable_reason,
            "index_revision_id": self.index_revision_id,
            "document_revision": self.document_revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CorpusScopeDocumentBinding:
        return cls(
            corpus_document_id=str(value["corpus_document_id"]),
            rag_document_id=(
                str(value["rag_document_id"]) if value.get("rag_document_id") else None
            ),
            availability=str(value.get("availability", "unavailable")),
            status=str(value["status"]) if value.get("status") else None,
            unavailable_reason=(
                str(value["unavailable_reason"]) if value.get("unavailable_reason") else None
            ),
            index_revision_id=(
                str(value["index_revision_id"]) if value.get("index_revision_id") else None
            ),
            document_revision=(
                str(value["document_revision"]) if value.get("document_revision") else None
            ),
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
    document_bindings: list[CorpusScopeDocumentBinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scope_mode: str = AssistantScopeMode.FIXED.value
    mapping_status: str = "ok"

    @property
    def documents(self) -> list[CorpusScopeDocumentBinding]:
        """Canonical explicit corpus/RAG identity mapping.

        ``document_bindings`` remains as the compatibility name used by older
        callers and persisted snapshots.
        """
        return self.document_bindings

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
            "documents": [binding.to_dict() for binding in self.document_bindings],
            "document_bindings": [binding.to_dict() for binding in self.document_bindings],
            "warnings": self.warnings,
            "scope_mode": self.scope_mode,
            "mapping_status": self.mapping_status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CorpusScopeSnapshot:
        raw_bindings = value.get("documents", value.get("document_bindings", []))
        if not isinstance(raw_bindings, list):
            raw_bindings = []
        document_bindings = [
            CorpusScopeDocumentBinding.from_dict(binding)
            for binding in raw_bindings
            if isinstance(binding, dict)
        ]
        if "mapping_status" in value and value.get("mapping_status"):
            mapping_status = str(value["mapping_status"])
        elif document_bindings:
            mapping_status = "ok"
        elif value.get("rag_document_ids") or value.get("corpus_document_ids"):
            # Legacy snapshots with parallel ID lists but no explicit mapping.
            mapping_status = "unverifiable"
        else:
            mapping_status = "ok"
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
            unavailable_corpus_document_ids=list(value.get("unavailable_corpus_document_ids", [])),
            unavailable_reasons=dict(value.get("unavailable_reasons", {})),
            document_bindings=document_bindings,
            warnings=list(value.get("warnings", [])),
            scope_mode=str(value.get("scope_mode", AssistantScopeMode.FIXED.value)),
            mapping_status=mapping_status,
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

        # Authoritative RAG status (batched) — I2: no uploader filter
        rag_docs = await self.rag_repo.get_documents_by_ids(
            rag_ids,
            user_id=None,
            include_deleted=True,
        )
        by_id = {d.id: d for d in rag_docs}

        indexed: set[str] = set()
        unavailable_corpus_document_ids: list[str] = []
        unavailable_reasons: dict[str, str] = {}
        document_bindings: list[CorpusScopeDocumentBinding] = []
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
                unavailable_corpus_document_ids.append(corpus_document.id)
                unavailable_reasons[corpus_document.id] = reason

            metadata: dict[str, Any] = {}
            if rag_document is not None:
                try:
                    metadata = json.loads(rag_document.metadata_json or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
            document_bindings.append(
                CorpusScopeDocumentBinding(
                    corpus_document_id=corpus_document.id,
                    rag_document_id=rag_id,
                    availability="indexed" if reason is None else "unavailable",
                    status=(str(rag_document.status) if rag_document is not None else None),
                    unavailable_reason=reason,
                    index_revision_id=(
                        current_revision_id
                        if rag_document is not None
                        and isinstance(
                            current_revision_id := getattr(
                                rag_document, "current_revision_id", None
                            ),
                            str,
                        )
                        else None
                    ),
                    document_revision=(
                        metadata.get("checksum_sha256") or metadata.get("document_revision")
                    ),
                )
            )

        if not documents:
            warnings.append("Corpus has no documents")
        elif not indexed:
            warnings.append("No indexed documents available for retrieval")
        elif unavailable_corpus_document_ids:
            warnings.append(
                f"{len(unavailable_corpus_document_ids)} corpus document(s) "
                "unavailable or still indexing"
            )

        # Evidence allow-list and compatibility ID lists are derived from the
        # explicit bindings only — never from independently sorted parallel arrays.
        document_bindings = sorted(
            document_bindings, key=lambda binding: binding.corpus_document_id
        )
        allow_list = sorted(
            binding.rag_document_id
            for binding in document_bindings
            if binding.availability == "indexed" and binding.rag_document_id
        )
        unavailable_rag_ids = sorted(
            {
                binding.rag_document_id
                for binding in document_bindings
                if binding.availability == "unavailable" and binding.rag_document_id
            }
        )
        corpus_document_ids = [binding.corpus_document_id for binding in document_bindings]
        unavailable_corpus_document_ids = sorted(
            binding.corpus_document_id
            for binding in document_bindings
            if binding.availability == "unavailable"
        )
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
            corpus_document_ids=corpus_document_ids,
            indexed_rag_document_ids=allow_list,
            unavailable_rag_document_ids=unavailable_rag_ids,
            scope_hash=_scope_hash(allow_list, corpus_id=corpus.id, project_id=corpus.project_id),
            index_version=self.rag_config.index_version,
            retrieval_version=self.rag_config.retrieval_algorithm_version,
            evidence_revision_hash=evidence_revision_hash,
            total_documents=len(documents),
            indexed_count=indexed_corpus_document_count,
            unavailable_count=len(unavailable_corpus_document_ids),
            unavailable_corpus_document_ids=unavailable_corpus_document_ids,
            unavailable_reasons=unavailable_reasons,
            document_bindings=document_bindings,
            warnings=warnings,
            mapping_status="ok",
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

    async def frozen_revision_ids(self, scope: CorpusScopeSnapshot) -> list[str]:
        """Validate a fixed snapshot and return its exact revision allow-list.

        A fixed scope is replayable only when every indexed member carries an
        immutable revision and the stored evidence hash still describes those
        revision chunks.  Never substitute a document's current revision.
        """
        bindings = [
            binding
            for binding in scope.document_bindings
            if binding.availability == "indexed" and binding.rag_document_id
        ]
        revisions_by_document = {
            binding.rag_document_id: binding.index_revision_id for binding in bindings
        }
        if set(revisions_by_document) != set(scope.rag_document_ids) or any(
            not revision_id for revision_id in revisions_by_document.values()
        ):
            raise HTTPException(
                status_code=409,
                detail=("historical_revision_unavailable: fixed scope has no exact revisions"),
            )

        revision_ids = [
            str(revisions_by_document[document_id]) for document_id in scope.rag_document_ids
        ]
        available = await self.rag_repo.list_available_revision_ids(
            document_ids=list(scope.rag_document_ids),
            revision_ids=revision_ids,
        )
        if set(available) != set(revision_ids):
            raise HTTPException(
                status_code=409,
                detail=("historical_revision_unavailable: frozen revision is not retrievable"),
            )

        chunks = await self.rag_repo.list_evidence_revision_chunks(
            list(scope.rag_document_ids), revision_ids=revision_ids
        )
        chunks_by_document: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(
                chunk_revision_identity(chunk)
            )
        actual_hash = build_evidence_revision_hash(
            corpus_id=scope.corpus_id,
            project_id=scope.project_id,
            document_revisions=[
                {
                    "rag_document_id": document_id,
                    "content_fingerprint": next(
                        binding.document_revision
                        for binding in bindings
                        if binding.rag_document_id == document_id
                    ),
                    "chunks": chunks_by_document.get(document_id, []),
                }
                for document_id in scope.rag_document_ids
            ],
            config=self.rag_config,
        )
        if actual_hash != scope.evidence_revision_hash:
            raise HTTPException(
                status_code=409,
                detail=("historical_revision_unavailable: frozen evidence hash no longer matches"),
            )
        return revision_ids

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
