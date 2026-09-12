"""Corpus-scoped Ask Corpus orchestration (research → RAG; invariants I1–I5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.identity_access.models import User
from backend.modules.rag.application.evidence_revision import StaleEvidenceRevisionError
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import MessageRole
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.conversation_context_service import (
    ConversationContextService,
)
from backend.modules.text_research.application.corpus_scope_service import (
    CorpusScopeService,
    CorpusScopeSnapshot,
)
from backend.modules.text_research.domain.enums import AssistantScopeMode
from backend.modules.text_research.domain.models import (
    ResearchAssistantScopeSnapshot,
    ResearchAssistantThread,
)

logger = logging.getLogger(__name__)


class CorpusAssistantService(ResearchAccessMixin):
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.scope_service = CorpusScopeService(db)
        self.rag_repo = RagRepository(db)
        self.rag_config = RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.rag_config)
        self.answers = RagAnswerService(db, self.rag_config)
        self.conversation_context = ConversationContextService(
            max_context_tokens=min(1500, self.rag_config.max_context_tokens),
        )

    async def create_thread(
        self,
        *,
        corpus_id: str,
        user: User,
        title: str | None = None,
        document_subset: list[str] | None = None,
        scope_mode: str = AssistantScopeMode.FIXED.value,
    ) -> tuple[ResearchAssistantThread, CorpusScopeSnapshot]:
        if scope_mode not in {item.value for item in AssistantScopeMode}:
            raise HTTPException(status_code=422, detail="scope_mode must be fixed or live")
        if scope_mode == AssistantScopeMode.LIVE.value and document_subset is not None:
            raise HTTPException(
                status_code=422,
                detail="A live thread follows the current corpus and cannot use a document subset",
            )
        scope = await self.scope_service.resolve(
            corpus_id=corpus_id,
            user_id=user.id,
            document_subset=document_subset,
        )
        scope.scope_mode = scope_mode
        conversation = await self.rag_repo.create_conversation(
            user_id=user.id,
            project_id=scope.project_id,
            title=title or f"Ask Corpus — {scope.corpus_name}",
            scope_snapshot=scope.to_dict(),
        )
        thread = ResearchAssistantThread(
            id=str(uuid4()),
            corpus_id=scope.corpus_id,
            project_id=scope.project_id,
            user_id=user.id,
            rag_conversation_id=conversation.id,
            title=conversation.title,
            scope_mode=scope_mode,
            evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
            scope_snapshot_json=json.dumps(scope.to_dict(), ensure_ascii=True),
        )
        self.db.add(thread)
        await self.db.flush()
        await self.scope_service.record_scope_event(
            thread=thread,
            actor_id=user.id,
            action="thread_created",
            scope=scope,
        )
        await self.db.commit()
        return thread, scope

    async def list_threads(self, *, corpus_id: str, user_id: str) -> list[ResearchAssistantThread]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        result = await self.db.execute(
            select(ResearchAssistantThread)
            .where(
                ResearchAssistantThread.corpus_id == corpus_id,
                ResearchAssistantThread.user_id == user_id,
            )
            .order_by(ResearchAssistantThread.updated_at.desc())
        )
        return list(result.scalars().all())

    async def get_thread(
        self, thread_id: str, *, user_id: str
    ) -> tuple[ResearchAssistantThread, list]:
        thread = await self._get_thread_or_404(thread_id, user_id=user_id)
        messages, _ = await self.rag_repo.list_messages(thread.rag_conversation_id, limit=500)
        return thread, messages

    async def retrieve(
        self,
        *,
        corpus_id: str,
        user: User,
        query: str,
        intent: str | None = None,
        document_subset: list[str] | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
    ) -> tuple[CorpusScopeSnapshot, object]:
        scope = await self.scope_service.resolve(
            corpus_id=corpus_id,
            user_id=user.id,
            document_subset=document_subset,
        )
        outcome = await self.retrieval.retrieve(
            query,
            user_id=user.id,
            project_id=scope.project_id,
            top_k=top_k,
            filters={
                "document_ids": list(scope.rag_document_ids),
                "owner_scoped": False,
                "retrieval_mode": retrieval_mode or "hybrid",
            },
            intent=intent,
            persist_trace=True,
            evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
        )
        return scope, outcome

    @staticmethod
    async def _rag_to_corpus_document_map(scope: CorpusScopeSnapshot) -> dict[str, str]:
        """Resolve citations only through the immutable scope binding."""
        return {
            binding.rag_document_id: binding.corpus_document_id
            for binding in scope.document_bindings
            if binding.availability == "indexed" and binding.rag_document_id
        }

    def _enrich_citations(
        self,
        citations: list,
        *,
        rag_to_corpus: dict[str, str],
        revision_by_document: dict[str, str] | None = None,
    ) -> list[dict]:
        enriched: list[dict] = []
        revision_map = revision_by_document or {}
        for c in citations:
            corpus_document_id = rag_to_corpus.get(c.document_id)
            if corpus_document_id is None:
                # Never substitute a current corpus document for a citation outside
                # the frozen scope / revision binding.
                raise HTTPException(
                    status_code=409,
                    detail="historical_source_unavailable",
                )
            index_revision_id = getattr(c, "index_revision_id", None) or revision_map.get(
                c.document_id
            )
            source_status = getattr(c, "source_status", None)
            if revision_map and not index_revision_id:
                # Missing frozen revision must not reopen as the live document.
                source_status = "historical_source_unavailable"
            enriched.append(
                {
                    "document_id": c.document_id,
                    "corpus_document_id": corpus_document_id,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "score": c.score,
                    "snippet": c.snippet,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "citation_number": c.citation_number,
                    "used_in_answer": c.used_in_answer,
                    "section_heading": c.section_heading,
                    "char_start": getattr(c, "char_start", None),
                    "char_end": getattr(c, "char_end", None),
                    "source_span_ids": getattr(c, "source_span_ids", None),
                    "parent_context_id": getattr(c, "parent_context_id", None),
                    "offset_coordinate_system": getattr(c, "offset_coordinate_system", None),
                    "offset_scope": getattr(c, "offset_scope", None),
                    "offset_scope_id": getattr(c, "offset_scope_id", None),
                    "source_spans": getattr(c, "source_spans", None),
                    "index_revision_id": index_revision_id,
                    "source_status": source_status,
                }
            )
        return enriched

    @staticmethod
    def _mark_turn_message(message, *, status: str, **fields: object) -> None:
        try:
            metadata = json.loads(getattr(message, "metadata_json", None) or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["turn_status"] = status
        metadata.update(fields)
        message.metadata_json = json.dumps(metadata, ensure_ascii=True)

    async def _persist_failed_turn(
        self,
        *,
        conversation_id: str,
        pending_user_message,
        query: str,
        original_query: str,
        resolved_retrieval_query: str,
        context_message_ids: list[str],
        error: Exception,
        scope: CorpusScopeSnapshot,
        retrieval_trace_id: str | None,
        retrieval_completed: bool,
    ) -> None:
        await self.db.rollback()
        status_code = getattr(error, "status_code", None)
        retryable = not isinstance(status_code, int) or status_code >= 500
        failure_type = "generation_failed" if retrieval_completed else "retrieval_failed"
        safe_error_summary = (
            "The assistant could not complete this request. Please try again."
            if retryable
            else "This request could not be completed with the selected corpus scope."
        )
        failed_assistant = await self.rag_repo.create_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT.value,
            content=safe_error_summary,
            model_name="error",
            retrieval_trace_id=retrieval_trace_id,
            evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
            metadata={
                "turn_status": "failed",
                "failure_type": failure_type,
                "safe_error_summary": safe_error_summary,
                "retrieval_trace_id": retrieval_trace_id,
                "retrieval_completed": retrieval_completed,
                "scope_hash": getattr(scope, "scope_hash", None),
                "evidence_revision_hash": getattr(scope, "evidence_revision_hash", None),
                "retryable": retryable,
                "original_query": original_query,
                "resolved_retrieval_query": resolved_retrieval_query,
                "context_message_ids": context_message_ids,
                "query": query,
            },
        )
        self._mark_turn_message(
            pending_user_message,
            status="failed",
            failed_assistant_message_id=failed_assistant.id,
            failure_type=failure_type,
            retryable=retryable,
        )
        await self.db.commit()

    async def ask(
        self,
        *,
        corpus_id: str,
        user: User,
        query: str,
        thread_id: str | None = None,
        intent: str | None = None,
        document_subset: list[str] | None = None,
        progress_callback: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> dict:
        """Run one turn with explicit pending, completed, and failed states.

        The pending user message is committed before retrieval/generation. Those
        downstream services run without committing, so the assistant message,
        scope snapshot, and completed user state commit together. Failures roll
        back uncommitted work and commit a failed assistant message.
        """

        async def emit(event: str, payload: dict | None = None) -> None:
            if progress_callback is not None:
                await progress_callback(event, payload or {})

        if not self.rag_config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        if thread_id:
            thread = await self._get_thread_or_404(thread_id, user_id=user.id)
            if thread.corpus_id != corpus_id:
                raise HTTPException(status_code=400, detail="Thread does not belong to this corpus")
            if document_subset is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Update thread scope explicitly before sending a message with a new scope"
                    ),
                )
            scope = await self.scope_service.resolve_for_thread(thread, user_id=user.id)
        else:
            thread, scope = await self.create_thread(
                corpus_id=corpus_id,
                user=user,
                document_subset=document_subset,
            )

        conversation_id = thread.rag_conversation_id

        frozen_revision_ids: list[str] | None = None
        if (
            isinstance(scope, CorpusScopeSnapshot)
            and scope.scope_mode == AssistantScopeMode.FIXED.value
        ):
            frozen_revision_ids = await self.scope_service.frozen_revision_ids(scope)

        # Load prior turns BEFORE writing the new user message (no cross-thread leakage).
        prior_messages, _ = await self.rag_repo.list_messages(conversation_id, limit=100)
        ctx = self.conversation_context.build(
            original_query=query,
            prior_messages=prior_messages,
        )

        pending_user_message = await self.rag_repo.create_message(
            conversation_id=conversation_id,
            role=MessageRole.USER.value,
            content=query,
            metadata={
                "turn_status": "pending",
                "original_query": ctx.original_query,
                "resolved_retrieval_query": ctx.resolved_retrieval_query,
                "context_message_ids": ctx.prior_message_ids,
            },
        )
        # The pending user turn is durable before the external AI call starts.
        await self.db.commit()
        await emit(
            "turn_created",
            {
                "thread_id": thread.id,
                "conversation_id": conversation_id,
                "message_id": pending_user_message.id,
            },
        )

        outcome = None
        retrieval_completed = False
        try:
            # I4: exactly one retrieve (resolved query), then answer_from_retrieval
            await emit("retrieval_started")
            try:
                outcome = await self.retrieval.retrieve(
                    ctx.resolved_retrieval_query,
                    user_id=user.id,
                    project_id=scope.project_id,
                    filters={
                        "document_ids": list(scope.rag_document_ids),
                        "owner_scoped": False,
                        **(
                            {"index_revision_ids": frozen_revision_ids}
                            if frozen_revision_ids is not None
                            else {}
                        ),
                    },
                    intent=intent,
                    conversation_id=conversation_id,
                    persist_trace=True,
                    evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                )
            except StaleEvidenceRevisionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            retrieval_completed = True
            await emit(
                "retrieval_complete",
                {
                    "retrieval_trace_id": outcome.retrieval_trace_id,
                    "retrieved_chunk_count": len(outcome.chunks),
                },
            )

            await emit("generation_started")
            answer = await self.answers.answer_from_retrieval(
                query,
                outcome=outcome,
                user=user,
                project_id=scope.project_id,
                document_ids=list(scope.rag_document_ids),
                include_memory=False,
                conversation_context=self.conversation_context.format_for_generation(ctx),
                resolved_retrieval_query=ctx.resolved_retrieval_query,
                context_message_ids=ctx.prior_message_ids,
                commit=False,
            )
            await emit(
                "citation_validation_complete",
                {"status": answer.citation_validation_status},
            )

            rag_to_corpus = await self._rag_to_corpus_document_map(scope)
            revision_by_document = {
                binding.rag_document_id: binding.index_revision_id
                for binding in scope.document_bindings
                if binding.availability == "indexed"
                and binding.rag_document_id
                and binding.index_revision_id
            }
            citation_payload = self._enrich_citations(
                answer.citations,
                rag_to_corpus=rag_to_corpus,
                revision_by_document=revision_by_document,
            )

            assistant_message = await self.rag_repo.create_message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT.value,
                content=answer.answer,
                model_name=answer.model_name,
                prompt_template_id=answer.prompt_template_id,
                prompt_version_id=answer.prompt_version_id,
                retrieval_trace_id=answer.retrieval_trace_id,
                evidence_revision_hash=getattr(answer, "evidence_revision_hash", None),
                citations=citation_payload,
                claims=[
                    {
                        "text": claim.text,
                        "chunk_ids": claim.chunk_ids,
                        "citation_numbers": claim.citation_numbers,
                    }
                    for claim in answer.claims
                ],
                metadata={
                    "turn_status": "completed",
                    "coverage": {
                        "documents_in_scope": (
                            answer.coverage.documents_in_scope if answer.coverage else 0
                        ),
                        "documents_with_retrieved_evidence": (
                            answer.coverage.documents_with_retrieved_evidence
                            if answer.coverage
                            else 0
                        ),
                        "retrieved_passage_count": (
                            answer.coverage.retrieved_passage_count if answer.coverage else 0
                        ),
                        "coverage_ratio": answer.coverage.coverage_ratio
                        if answer.coverage
                        else 0.0,
                    },
                    "citation_validation_failed": answer.citation_validation_failed,
                    "citation_validation_status": answer.citation_validation_status,
                    "no_context_found": answer.no_context_found,
                    "retrieval_degraded": answer.retrieval_degraded,
                    "degradation_reason": answer.degradation_reason,
                    "ai_run_id": answer.ai_run_id,
                    "original_query": ctx.original_query,
                    "resolved_retrieval_query": ctx.resolved_retrieval_query,
                    "context_message_ids": ctx.prior_message_ids,
                    "scope_hash": scope.scope_hash,
                    "evidence_revision_hash": getattr(scope, "evidence_revision_hash", None),
                    "scope_mode": scope.scope_mode,
                    "retrieval_algorithm_version": scope.retrieval_version,
                    "index_version": scope.index_version,
                    "fusion_method": outcome.fusion_method,
                },
            )

            unavailable_corpus_document_ids = getattr(scope, "unavailable_corpus_document_ids", [])
            if not isinstance(unavailable_corpus_document_ids, list):
                unavailable_corpus_document_ids = []
            unavailable_reasons = getattr(scope, "unavailable_reasons", {})
            if not isinstance(unavailable_reasons, dict):
                unavailable_reasons = {}
            scope_bindings = getattr(scope, "document_bindings", [])
            document_bindings = (
                [binding.to_dict() for binding in scope_bindings]
                if isinstance(scope_bindings, list)
                else []
            )

            snapshot_row = ResearchAssistantScopeSnapshot(
                id=str(uuid4()),
                thread_id=thread.id,
                rag_message_id=assistant_message.id,
                retrieval_trace_id=answer.retrieval_trace_id,
                corpus_id=scope.corpus_id,
                project_id=scope.project_id,
                scope_hash=scope.scope_hash,
                rag_document_ids_json=json.dumps(scope.rag_document_ids, ensure_ascii=True),
                corpus_document_ids_json=json.dumps(scope.corpus_document_ids, ensure_ascii=True),
                indexed_rag_document_ids_json=json.dumps(
                    scope.indexed_rag_document_ids, ensure_ascii=True
                ),
                unavailable_rag_document_ids_json=json.dumps(
                    scope.unavailable_rag_document_ids, ensure_ascii=True
                ),
                unavailable_corpus_document_ids_json=json.dumps(
                    unavailable_corpus_document_ids, ensure_ascii=True
                ),
                unavailable_reasons_json=json.dumps(unavailable_reasons, ensure_ascii=True),
                document_bindings_json=json.dumps(document_bindings, ensure_ascii=True),
                scope_mode=scope.scope_mode,
                index_version=scope.index_version,
                retrieval_version=scope.retrieval_version,
                evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                created_at=datetime.now(UTC),
            )
            self.db.add(snapshot_row)
            thread.updated_at = datetime.now(UTC)
            self._mark_turn_message(
                pending_user_message,
                status="completed",
                assistant_message_id=assistant_message.id,
            )
            await self.db.commit()

            return {
                "thread_id": thread.id,
                "conversation_id": conversation_id,
                "message_id": assistant_message.id,
                "retrieval_trace_id": answer.retrieval_trace_id,
                "evidence_revision_hash": getattr(answer, "evidence_revision_hash", None),
                "query": query,
                "original_query": ctx.original_query,
                "resolved_retrieval_query": ctx.resolved_retrieval_query,
                "answer": answer.answer,
                "citations": citation_payload,
                "claims": [
                    {
                        "text": claim.text,
                        "chunk_ids": claim.chunk_ids,
                        "citation_numbers": claim.citation_numbers,
                    }
                    for claim in answer.claims
                ],
                "retrieved_chunk_ids": answer.retrieved_chunk_ids,
                "model_name": answer.model_name,
                "latency_ms": answer.latency_ms,
                "no_context_found": answer.no_context_found,
                "retrieval_degraded": answer.retrieval_degraded,
                "degradation_reason": answer.degradation_reason,
                "citation_validation_failed": answer.citation_validation_failed,
                "citation_validation_status": answer.citation_validation_status,
                "injection_chunks_filtered": answer.injection_chunks_filtered,
                "scope": scope.to_dict(),
                "coverage": {
                    "documents_in_scope": (
                        answer.coverage.documents_in_scope if answer.coverage else 0
                    ),
                    "documents_with_retrieved_evidence": (
                        answer.coverage.documents_with_retrieved_evidence if answer.coverage else 0
                    ),
                    "retrieved_passage_count": (
                        answer.coverage.retrieved_passage_count if answer.coverage else 0
                    ),
                    "coverage_ratio": answer.coverage.coverage_ratio if answer.coverage else 0.0,
                },
                "context_message_ids": ctx.prior_message_ids,
                "ai_run_id": answer.ai_run_id,
                "fusion_method": outcome.fusion_method,
            }
        except Exception as exc:
            try:
                await self._persist_failed_turn(
                    conversation_id=conversation_id,
                    pending_user_message=pending_user_message,
                    query=query,
                    original_query=ctx.original_query,
                    resolved_retrieval_query=ctx.resolved_retrieval_query,
                    context_message_ids=ctx.prior_message_ids,
                    error=exc,
                    scope=scope,
                    retrieval_trace_id=getattr(outcome, "retrieval_trace_id", None),
                    retrieval_completed=retrieval_completed,
                )
            except Exception:
                logger.exception("Failed to persist failed assistant turn")
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=502, detail="Assistant turn failed") from exc

    async def _get_thread_or_404(self, thread_id: str, *, user_id: str) -> ResearchAssistantThread:
        result = await self.db.execute(
            select(ResearchAssistantThread).where(ResearchAssistantThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()
        if thread is None:
            raise HTTPException(status_code=404, detail="Assistant conversation not found")
        await self.ensure_project_access(user_id=user_id, project_id=thread.project_id)
        if thread.user_id != user_id:
            raise HTTPException(status_code=404, detail="Assistant conversation not found")
        return thread

    async def get_thread_scope(
        self, thread: ResearchAssistantThread, *, user_id: str
    ) -> CorpusScopeSnapshot:
        return await self.scope_service.resolve_for_thread(thread, user_id=user_id)

    async def update_thread_scope(
        self,
        *,
        thread_id: str,
        user: User,
        document_subset: list[str] | None,
        scope_mode: str | None,
        reason: str | None,
    ) -> CorpusScopeSnapshot:
        thread = await self._get_thread_or_404(thread_id, user_id=user.id)
        return await self.scope_service.update_thread_scope(
            thread=thread,
            user_id=user.id,
            document_subset=document_subset,
            scope_mode=scope_mode,
            reason=reason,
        )
