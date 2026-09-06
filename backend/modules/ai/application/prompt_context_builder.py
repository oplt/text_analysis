from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.models import MemoryItem
from backend.modules.rag.application.prompt_context_service import PromptContextService
from backend.modules.rag.infrastructure.rag_config import RagConfig


@dataclass(frozen=True, slots=True)
class AgentPromptContext:
    additional_system_context: str | None
    effective_retrieval_query: str | None
    retrieved_memories: list[MemoryItem]
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_degraded: bool = False
    memory_degraded: bool = False
    degradation_reason: str | None = None
    injection_chunks_filtered: int = 0


class AgentPromptContextBuilder:
    """AI adapter for the RAG-owned prompt-context orchestration service."""

    def __init__(self, db: AsyncSession, memory: MemoryService):
        self.context = PromptContextService(db, memory=memory)

    async def build(
        self,
        *,
        user_id: str,
        agent_id: str,
        query: str,
        run_id: str | None,
        project_id: str | None,
        retrieval_query: str | None,
        document_ids: list[str],
        top_k: int,
    ) -> AgentPromptContext:
        outcome = await self.context.build(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            project_id=project_id,
            document_ids=document_ids or None,
            top_k=top_k,
        )
        return AgentPromptContext(
            additional_system_context=outcome.system_context,
            effective_retrieval_query=(
                None if RagConfig.from_settings().enabled else retrieval_query
            ),
            retrieved_memories=outcome.retrieved_memories,
            retrieved_chunk_ids=outcome.retrieved_chunk_ids,
            retrieval_degraded=outcome.retrieval_degraded,
            memory_degraded=outcome.memory_degraded,
            degradation_reason=outcome.degradation_reason,
            injection_chunks_filtered=outcome.injection_chunks_filtered,
        )
