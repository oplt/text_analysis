from __future__ import annotations

import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.ai import metrics
from backend.modules.ai.application.prompt_context_builder import AgentPromptContextBuilder
from backend.modules.ai.service import AiService
from backend.modules.identity_access.models import User
from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.models import WorkingMemoryContext
from backend.modules.memory.infrastructure.memory_config import MemoryConfig
from backend.modules.memory.workers import queue_turn_memory_extraction

logger = logging.getLogger(__name__)

DEFAULT_AGENT_ID = "default"


class AgentService:
    """
    Orchestrates authenticated AI agent runs with layered memory and RAG context.

    User identity always comes from the authenticated User object — never from message text.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ai = AiService(db)
        self.memory = MemoryService(db)
        self.config = MemoryConfig.from_settings()
        self.prompt_context = AgentPromptContextBuilder(db, self.memory)

    async def run_agent_prompt(
        self,
        user: User,
        *,
        prompt_template_key: str | None,
        prompt_version_id: str | None,
        variables: dict[str, Any],
        retrieval_query: str | None,
        document_ids: list[str],
        top_k: int,
        review_required: bool,
        agent_id: str = DEFAULT_AGENT_ID,
        run_id: str | None = None,
        project_id: str | None = None,
        user_message: str | None = None,
    ):
        started = perf_counter()
        resolved_run_id = run_id or str(uuid4())
        working = WorkingMemoryContext(
            user_message=user_message or str(variables.get("user_message", "")),
            run_id=resolved_run_id,
            project_id=project_id,
        )

        query = retrieval_query or user_message or str(variables.get("user_message", ""))
        prompt_context = await self.prompt_context.build(
            user_id=user.id,
            agent_id=agent_id,
            query=query,
            run_id=resolved_run_id,
            project_id=project_id,
            retrieval_query=retrieval_query,
            document_ids=document_ids,
            top_k=top_k,
        )
        working.retrieved_memories = prompt_context.retrieved_memories

        run = await self.ai.run_prompt(
            user,
            prompt_template_key=prompt_template_key,
            prompt_version_id=prompt_version_id,
            variables=variables,
            retrieval_query=prompt_context.effective_retrieval_query,
            document_ids=document_ids,
            top_k=top_k,
            review_required=review_required,
            additional_system_context=prompt_context.additional_system_context,
            retrieved_chunk_ids=prompt_context.retrieved_chunk_ids,
            retrieval_degraded=prompt_context.retrieval_degraded,
            memory_degraded=prompt_context.memory_degraded,
            degradation_reason=prompt_context.degradation_reason,
            injection_chunks_filtered=prompt_context.injection_chunks_filtered,
        )

        if prompt_context.retrieval_degraded:
            metrics.agent_context_degraded_total.labels(source="retrieval").inc()
        if prompt_context.memory_degraded:
            metrics.agent_context_degraded_total.labels(source="memory").inc()

        if self.config.enabled and self.config.write_enabled and run.output_text:
            try:
                queue_turn_memory_extraction(
                    user_id=user.id,
                    agent_id=agent_id,
                    run_id=resolved_run_id,
                    project_id=project_id,
                    user_message=working.user_message,
                    assistant_message=run.output_text,
                    source_message_id=run.id,
                )
            except Exception:
                logger.exception(
                    "Memory extraction queue degraded for user=%s run=%s",
                    user.id,
                    resolved_run_id,
                )

        metrics.agent_run_latency_ms.observe((perf_counter() - started) * 1000)
        return run, resolved_run_id, working
