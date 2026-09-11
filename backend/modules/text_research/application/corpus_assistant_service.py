"""Corpus-scoped Ask Corpus orchestration (research → RAG; invariants I1–I5)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from backend.modules.identity_access.models import User
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import MessageRole, RetrievalIntent
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.corpus_scope_service import (
    CorpusScopeService,
    CorpusScopeSnapshot,
)
from backend.modules.text_research.domain.models import (
    ResearchAssistantScopeSnapshot,
    ResearchAssistantThread,
)
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class CorpusAssistantService(ResearchAccessMixin):
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.scope_service = CorpusScopeService(db)
        self.rag_repo = RagRepository(db)
        self.rag_config = RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.rag_config)
        self.answers = RagAnswerService(db, self.rag_config)

    async def create_thread(
        self,
        *,
        corpus_id: str,
        user: User,
        title: str | None = None,
        document_subset: list[str] | None = None,
    ) -> tuple[ResearchAssistantThread, CorpusScopeSnapshot]:
        scope = await self.scope_service.resolve(
            corpus_id=corpus_id,
            user_id=user.id,
            document_subset=document_subset,
        )
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
        )
        self.db.add(thread)
        await self.db.flush()
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
        # I1: always pass concrete list (possibly empty). I2: owner_scoped=False.
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
            intent=intent or RetrievalIntent.SEMANTIC_SEARCH,
            persist_trace=True,
        )
        return scope, outcome

    async def ask(
        self,
        *,
        corpus_id: str,
        user: User,
        query: str,
        thread_id: str | None = None,
        intent: str | None = None,
        document_subset: list[str] | None = None,
    ) -> dict:
        if not self.rag_config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        scope = await self.scope_service.resolve(
            corpus_id=corpus_id,
            user_id=user.id,
            document_subset=document_subset,
        )

        if thread_id:
            thread = await self._get_thread_or_404(thread_id, user_id=user.id)
            if thread.corpus_id != corpus_id:
                raise HTTPException(status_code=400, detail="Thread does not belong to this corpus")
        else:
            thread, _ = await self.create_thread(
                corpus_id=corpus_id,
                user=user,
                document_subset=document_subset,
            )

        conversation_id = thread.rag_conversation_id

        await self.rag_repo.create_message(
            conversation_id=conversation_id,
            role=MessageRole.USER.value,
            content=query,
        )

        # I4: exactly one retrieve, then answer_from_retrieval
        outcome = await self.retrieval.retrieve(
            query,
            user_id=user.id,
            project_id=scope.project_id,
            filters={
                "document_ids": list(scope.rag_document_ids),
                "owner_scoped": False,
            },
            intent=intent or RetrievalIntent.EVIDENCE,
            conversation_id=conversation_id,
            persist_trace=True,
        )

        answer = await self.answers.answer_from_retrieval(
            query,
            outcome=outcome,
            user=user,
            project_id=scope.project_id,
            document_ids=list(scope.rag_document_ids),
            include_memory=False,  # reproducibility: no mutable memory for research answers
        )

        assistant_message = await self.rag_repo.create_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT.value,
            content=answer.answer,
            model_name=answer.model_name,
            retrieval_trace_id=answer.retrieval_trace_id,
            citations=[
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "score": c.score,
                    "snippet": c.snippet,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "citation_number": c.citation_number,
                    "used_in_answer": c.used_in_answer,
                    "section_heading": c.section_heading,
                }
                for c in answer.citations
            ],
            claims=[
                {
                    "text": claim.text,
                    "chunk_ids": claim.chunk_ids,
                    "citation_numbers": claim.citation_numbers,
                }
                for claim in answer.claims
            ],
            metadata={
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
                    "coverage_ratio": answer.coverage.coverage_ratio if answer.coverage else 0.0,
                },
                "citation_validation_failed": answer.citation_validation_failed,
                "no_context_found": answer.no_context_found,
                "retrieval_degraded": answer.retrieval_degraded,
                "degradation_reason": answer.degradation_reason,
            },
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
            index_version=scope.index_version,
            retrieval_version=scope.retrieval_version,
            created_at=datetime.now(UTC),
        )
        self.db.add(snapshot_row)
        thread.updated_at = datetime.now(UTC)
        await self.db.commit()

        return {
            "thread_id": thread.id,
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "retrieval_trace_id": answer.retrieval_trace_id,
            "query": query,
            "answer": answer.answer,
            "citations": [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "score": c.score,
                    "snippet": c.snippet,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "citation_number": c.citation_number,
                    "used_in_answer": c.used_in_answer,
                    "section_heading": c.section_heading,
                }
                for c in answer.citations
            ],
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
            "injection_chunks_filtered": answer.injection_chunks_filtered,
            "scope": scope.to_dict(),
            "coverage": {
                "documents_in_scope": answer.coverage.documents_in_scope if answer.coverage else 0,
                "documents_with_retrieved_evidence": (
                    answer.coverage.documents_with_retrieved_evidence if answer.coverage else 0
                ),
                "retrieved_passage_count": (
                    answer.coverage.retrieved_passage_count if answer.coverage else 0
                ),
                "coverage_ratio": answer.coverage.coverage_ratio if answer.coverage else 0.0,
            },
        }

    async def _get_thread_or_404(
        self, thread_id: str, *, user_id: str
    ) -> ResearchAssistantThread:
        result = await self.db.execute(
            select(ResearchAssistantThread).where(ResearchAssistantThread.id == thread_id)
        )
        thread = result.scalar_one_or_none()
        if thread is None:
            raise HTTPException(status_code=404, detail="Assistant conversation not found")
        await self.ensure_project_access(user_id=user_id, project_id=thread.project_id)
        if thread.user_id != user_id:
            # Collaborators may view project corpora but threads are per-user for now
            raise HTTPException(status_code=404, detail="Assistant conversation not found")
        return thread
