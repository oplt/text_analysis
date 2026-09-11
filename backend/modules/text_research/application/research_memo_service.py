"""Application workflow for durable, evidence-backed research memos."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import ResearchMemoSourceType, ResearchMemoStatus
from backend.modules.text_research.domain.models import ResearchMemo, loads


class ResearchMemoService(ResearchAccessMixin):
    """Own memo authorization and snapshot semantics."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    @staticmethod
    def _validate_source(source_type: str) -> None:
        if source_type not in {item.value for item in ResearchMemoSourceType}:
            raise HTTPException(status_code=422, detail="Unsupported research memo source type")

    @staticmethod
    def _copy(value: Any, default: Any) -> Any:
        return value if value is not None else default

    @staticmethod
    def to_dict(memo: ResearchMemo) -> dict[str, Any]:
        return {
            "id": memo.id,
            "project_id": memo.project_id,
            "corpus_id": memo.corpus_id,
            "user_id": memo.user_id,
            "source_type": memo.source_type,
            "status": memo.status,
            "title": memo.title,
            "body": memo.body,
            "originating_assistant_message_id": memo.originating_assistant_message_id,
            "originating_synthesis_run_id": memo.originating_synthesis_run_id,
            "evidence_revision_hash": memo.evidence_revision_hash,
            "citations": loads(memo.citations_json, []),
            "claims": loads(memo.claims_json, []),
            "provenance": loads(memo.provenance_json, {}),
            "created_at": memo.created_at,
            "updated_at": memo.updated_at,
            "archived_at": memo.archived_at,
        }

    async def create(
        self,
        *,
        user_id: str,
        project_id: str,
        corpus_id: str | None,
        title: str,
        body: str,
        source_type: str = ResearchMemoSourceType.MANUAL.value,
        citations: list[dict[str, Any]] | None = None,
        claims: list[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
        evidence_revision_hash: str | None = None,
        originating_assistant_message_id: str | None = None,
        originating_synthesis_run_id: str | None = None,
    ) -> ResearchMemo:
        self._validate_source(source_type)
        if corpus_id:
            corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id, project_id=project_id)
        else:
            await self.ensure_project_access(user_id=user_id, project_id=project_id)
            corpus = None
        if originating_assistant_message_id and originating_synthesis_run_id:
            raise HTTPException(
                status_code=422, detail="A memo can have only one originating source"
            )

        if originating_assistant_message_id:
            if corpus is None or source_type != ResearchMemoSourceType.ASSISTANT_ANSWER.value:
                raise HTTPException(
                    status_code=422, detail="Assistant memos must belong to a corpus"
                )
            message = await self.repo.get_assistant_message_for_corpus(
                message_id=originating_assistant_message_id,
                corpus_id=corpus.id,
                user_id=user_id,
            )
            if message is None:
                raise HTTPException(status_code=404, detail="Assistant message not found")
            citations = loads(message.citations_json, [])
            claims = loads(message.claims_json, [])
            metadata = loads(message.metadata_json, {})
            scope_snapshot = await self.repo.get_scope_snapshot_for_message(message.id)
            provenance = {
                "kind": "assistant_answer",
                "message_id": message.id,
                "conversation_id": message.conversation_id,
                "retrieval_trace_id": message.retrieval_trace_id,
                "metadata": metadata,
                "scope_snapshot": (
                    {
                        "scope_hash": scope_snapshot.scope_hash,
                        "evidence_revision_hash": scope_snapshot.evidence_revision_hash,
                        "rag_document_ids": loads(scope_snapshot.rag_document_ids_json, []),
                        "corpus_document_ids": loads(scope_snapshot.corpus_document_ids_json, []),
                        "indexed_rag_document_ids": loads(
                            scope_snapshot.indexed_rag_document_ids_json, []
                        ),
                        "unavailable_rag_document_ids": loads(
                            scope_snapshot.unavailable_rag_document_ids_json, []
                        ),
                    }
                    if scope_snapshot
                    else None
                ),
            }
            evidence_revision_hash = message.evidence_revision_hash
            body = body or message.content
        elif originating_synthesis_run_id:
            if corpus is None or source_type != ResearchMemoSourceType.ANALYSIS_RESULT.value:
                raise HTTPException(
                    status_code=422, detail="Synthesis memos must belong to a corpus"
                )
            run = await self.get_run_or_404(originating_synthesis_run_id, user_id=user_id)
            if run.corpus_id != corpus.id or run.project_id != project_id:
                raise HTTPException(status_code=404, detail="Synthesis run not found")
            results = loads(run.results_json, {}) or {}
            citations = results.get("citations", [])
            claims = results.get("claims", [])
            provenance = {
                "kind": "synthesis_result",
                "run_id": run.id,
                "run_version": run.run_version,
                "status": run.status,
                "results": results.get("synthesis_provenance", {}),
                "parameters": loads(run.parameters_json, {}),
            }
            evidence_revision_hash = run.evidence_revision_hash or results.get(
                "evidence_revision_hash"
            )
            body = body or str(results.get("answer", ""))

        memo = ResearchMemo(
            project_id=project_id,
            corpus_id=corpus_id,
            user_id=user_id,
            source_type=source_type,
            status=ResearchMemoStatus.ACTIVE.value,
            title=title,
            body=body,
            originating_assistant_message_id=originating_assistant_message_id,
            originating_synthesis_run_id=originating_synthesis_run_id,
            evidence_revision_hash=evidence_revision_hash,
            citations_json=json.dumps(self._copy(citations, []), ensure_ascii=True, default=str),
            claims_json=json.dumps(self._copy(claims, []), ensure_ascii=True, default=str),
            provenance_json=json.dumps(self._copy(provenance, {}), ensure_ascii=True, default=str),
        )
        await self.repo.create_memo(memo)
        await self.db.commit()
        await self.db.refresh(memo)
        return memo

    async def list(
        self, *, user_id: str, project_id: str, corpus_id: str | None, include_archived: bool
    ) -> list[ResearchMemo]:
        if corpus_id:
            await self.get_corpus_or_404(corpus_id, user_id=user_id, project_id=project_id)
        else:
            await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_memos(
            project_id=project_id,
            user_id=user_id,
            corpus_id=corpus_id,
            include_archived=include_archived,
        )

    async def get(self, *, memo_id: str, user_id: str) -> ResearchMemo:
        memo = await self.repo.get_memo(memo_id)
        if memo is None:
            raise HTTPException(status_code=404, detail="Research memo not found")
        await self.ensure_project_access(user_id=user_id, project_id=memo.project_id)
        if memo.user_id != user_id:
            raise HTTPException(status_code=404, detail="Research memo not found")
        return memo

    async def update(self, *, memo_id: str, user_id: str, title: str | None, body: str | None):
        memo = await self.get(memo_id=memo_id, user_id=user_id)
        if memo.status == ResearchMemoStatus.ARCHIVED.value:
            raise HTTPException(status_code=409, detail="Archived research memos are read-only")
        if title is not None:
            memo.title = title
        if body is not None:
            memo.body = body
        memo.updated_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(memo)
        return memo

    async def archive(self, *, memo_id: str, user_id: str):
        memo = await self.get(memo_id=memo_id, user_id=user_id)
        memo.status = ResearchMemoStatus.ARCHIVED.value
        memo.archived_at = datetime.now(UTC)
        memo.updated_at = memo.archived_at
        await self.db.commit()
        await self.db.refresh(memo)
        return memo
