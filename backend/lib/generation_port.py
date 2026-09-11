from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.identity_access.models import User


@dataclass(slots=True)
class RagGenerationResult:
    id: str
    output_text: str | None
    model_name: str


class GenerationPort(Protocol):
    async def run_rag_answer(
        self,
        user: User,
        *,
        query: str,
        combined_context: str,
        retrieved_chunk_ids: list[str],
        review_required: bool = False,
        retrieval_degraded: bool = False,
        memory_degraded: bool = False,
        degradation_reason: str | None = None,
        injection_chunks_filtered: int = 0,
        commit: bool = True,
    ) -> RagGenerationResult: ...


class AiServiceGenerationPort:
    def __init__(self, db: AsyncSession):
        self._db = db

    def _get_service(self):
        from backend.modules.ai.service import AiService

        return AiService(self._db)

    async def run_rag_answer(
        self,
        user: User,
        *,
        query: str,
        combined_context: str,
        retrieved_chunk_ids: list[str],
        review_required: bool = False,
        retrieval_degraded: bool = False,
        memory_degraded: bool = False,
        degradation_reason: str | None = None,
        injection_chunks_filtered: int = 0,
        commit: bool = True,
    ) -> RagGenerationResult:
        run = await self._get_service().run_rag_answer(
            user,
            query=query,
            combined_context=combined_context,
            retrieved_chunk_ids=retrieved_chunk_ids,
            review_required=review_required,
            retrieval_degraded=retrieval_degraded,
            memory_degraded=memory_degraded,
            degradation_reason=degradation_reason,
            injection_chunks_filtered=injection_chunks_filtered,
            commit=commit,
        )
        return RagGenerationResult(
            id=run.id,
            output_text=run.output_text,
            model_name=run.model_name,
        )
