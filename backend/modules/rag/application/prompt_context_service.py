from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.models import MemoryItem, MemorySearchRequest
from backend.modules.memory.infrastructure.memory_config import MemoryConfig
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptContextOutcome:
    system_context: str | None
    document_context: str
    memory_context: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieved_memories: list[MemoryItem] = field(default_factory=list)
    retrieval_degraded: bool = False
    memory_degraded: bool = False
    degradation_reason: str | None = None
    no_matches: bool = False
    injection_chunks_filtered: int = 0

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.chunks]


class PromptContextService:
    """Composes bounded RAG and memory context for every generation entry point."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        rag_config: RagConfig | None = None,
        memory: MemoryService | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self.rag_config = rag_config or RagConfig.from_settings()
        self.memory_config = memory_config or MemoryConfig.from_settings()
        self.retrieval = RetrievalService(db, self.rag_config)
        self.memory = memory or MemoryService(db)
        self.context_builder = RagContextBuilder()

    async def build(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str,
        run_id: str | None,
        project_id: str | None,
        document_ids: list[str] | None,
        top_k: int | None,
        include_memory: bool = True,
    ) -> PromptContextOutcome:
        retrieval_result, memory_result = await asyncio.gather(
            self._retrieve(
                query=query,
                user_id=user_id,
                project_id=project_id,
                document_ids=document_ids,
                top_k=top_k,
            ),
            self._recall_memory(
                query=query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                project_id=project_id,
                include_memory=include_memory,
            ),
            return_exceptions=True,
        )

        retrieval = self._normalize_retrieval(retrieval_result, user_id=user_id)
        memory_context, memories, memory_degraded = self._normalize_memory(
            memory_result,
            user_id=user_id,
            run_id=run_id,
        )
        bounded_chunks = self.context_builder.trim_chunks_to_token_budget(
            retrieval.chunks,
            max_tokens=self.rag_config.max_context_tokens,
            overlap_dedupe_threshold=getattr(
                self.rag_config, "context_overlap_dedupe_threshold", 0.8
            ),
            ordering_policy=getattr(self.rag_config, "context_ordering_policy", "relevance"),
        )
        document_context = self.context_builder.build_document_context_block(bounded_chunks)
        system_context = self.context_builder.assemble_agent_system_context(
            memory_context=memory_context or None,
            document_context=document_context or None,
        )
        reason = retrieval.degradation_reason
        if memory_degraded and not reason:
            reason = "memory_recall_failed"
        return PromptContextOutcome(
            system_context=system_context,
            document_context=document_context,
            memory_context=memory_context,
            chunks=bounded_chunks,
            retrieved_memories=memories,
            retrieval_degraded=retrieval.degraded,
            memory_degraded=memory_degraded,
            degradation_reason=reason,
            no_matches=retrieval.no_matches,
            injection_chunks_filtered=retrieval.injection_chunks_filtered,
        )

    async def _retrieve(
        self,
        *,
        query: str,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        top_k: int | None,
    ) -> RetrievalOutcome:
        if not self.rag_config.enabled or not query:
            return RetrievalOutcome(chunks=[])
        filters: dict | None = None
        if document_ids is not None:
            filters = {"document_ids": document_ids, "owner_scoped": True}
        return await self.retrieval.retrieve(
            query,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k,
            filters=filters,
        )

    async def _recall_memory(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str,
        run_id: str | None,
        project_id: str | None,
        include_memory: bool,
    ) -> tuple[str, list[MemoryItem], bool]:
        if not include_memory or not self.memory_config.enabled or not query:
            return "", [], False
        return await self.memory.recall_for_prompt(
            MemorySearchRequest(
                user_id=user_id,
                agent_id=agent_id,
                query=query,
                run_id=run_id,
                project_id=project_id,
            )
        )

    @staticmethod
    def _normalize_retrieval(result: object, *, user_id: str) -> RetrievalOutcome:
        if isinstance(result, BaseException):
            logger.error("Prompt retrieval degraded user=%s: %s", user_id, result)
            return RetrievalOutcome(
                chunks=[],
                degraded=True,
                degradation_reason="retrieval_failed",
            )
        if isinstance(result, RetrievalOutcome):
            return result
        return RetrievalOutcome(
            chunks=[],
            degraded=True,
            degradation_reason="retrieval_invalid_result",
        )

    @staticmethod
    def _normalize_memory(
        result: object,
        *,
        user_id: str,
        run_id: str | None,
    ) -> tuple[str, list[MemoryItem], bool]:
        if isinstance(result, BaseException):
            logger.error("Prompt memory degraded user=%s run=%s: %s", user_id, run_id, result)
            return "", [], True
        if isinstance(result, tuple) and len(result) == 3:
            context, memories, degraded = result
            return str(context), list(memories), bool(degraded)
        return "", [], True
