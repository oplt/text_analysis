from __future__ import annotations

from time import perf_counter

from backend.lib.generation_port import AiServiceGenerationPort, GenerationPort
from backend.lib.project_access import ProjectAccessPort, SqlAlchemyProjectAccessPort
from backend.modules.identity_access.models import User
from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.infrastructure.memory_config import MemoryConfig
from backend.modules.rag.application.citation_service import CitationService
from backend.modules.rag.application.prompt_context_service import PromptContextService
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.models import RagAnswer
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

NO_CONTEXT_ANSWER = (
    "I could not find relevant document context for your question in the indexed documents."
)


class RagAnswerService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.config)
        self.context_builder = RagContextBuilder(CitationService())
        self.repo = RagRepository(db)
        self.project_access: ProjectAccessPort = SqlAlchemyProjectAccessPort(db)
        self.memory = MemoryService(db)
        self.memory_config = MemoryConfig.from_settings()
        self.prompt_context = PromptContextService(
            db,
            rag_config=self.config,
            memory=self.memory,
            memory_config=self.memory_config,
        )
        self.generation: GenerationPort = AiServiceGenerationPort(db)

    async def answer(
        self,
        query: str,
        *,
        user: User,
        project_id: str | None,
        run_id: str | None = None,
        agent_id: str | None = None,
        document_ids: list[str] | None = None,
        organization_id: str | None = None,
    ) -> RagAnswer:
        if not self.config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        user_id = user.id
        if project_id:
            await self.project_access.ensure_project_access(user_id, project_id)

        started = perf_counter()
        self.prompt_context.retrieval = self.retrieval
        self.prompt_context.memory = self.memory
        self.prompt_context.memory_config = self.memory_config
        context = await self.prompt_context.build(
            query=query,
            user_id=user_id,
            agent_id=agent_id or "default",
            run_id=run_id,
            project_id=project_id,
            document_ids=document_ids,
            top_k=None,
        )
        bounded_chunks = context.chunks
        citations = self.context_builder.citations.build_citations(bounded_chunks)

        if not bounded_chunks:
            latency_ms = int((perf_counter() - started) * 1000)
            metrics.rag_answer_latency_ms.observe(latency_ms)
            await self._log_query(
                user_id=user_id,
                project_id=project_id,
                organization_id=organization_id,
                query=query,
                answer=NO_CONTEXT_ANSWER,
                chunk_ids=[],
                model_name="none",
                latency_ms=latency_ms,
            )
            return RagAnswer(
                query=query,
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                retrieved_chunk_ids=[],
                model_name="none",
                latency_ms=latency_ms,
                no_context_found=True,
                ai_run_id=None,
                retrieval_degraded=context.retrieval_degraded,
                memory_degraded=context.memory_degraded,
                degradation_reason=context.degradation_reason,
                injection_chunks_filtered=context.injection_chunks_filtered,
            )

        reference_context = context.system_context or ""
        chunk_ids = [c.chunk_id for c in bounded_chunks]

        ai_run = await self.generation.run_rag_answer(
            user,
            query=query,
            combined_context=reference_context,
            retrieved_chunk_ids=chunk_ids,
            retrieval_degraded=context.retrieval_degraded,
            memory_degraded=context.memory_degraded,
            degradation_reason=context.degradation_reason,
            injection_chunks_filtered=context.injection_chunks_filtered,
        )

        latency_ms = int((perf_counter() - started) * 1000)
        metrics.rag_answer_latency_ms.observe(latency_ms)

        await self._log_query(
            user_id=user_id,
            project_id=project_id,
            organization_id=organization_id,
            query=query,
            answer=ai_run.output_text or "",
            chunk_ids=chunk_ids,
            model_name=ai_run.model_name,
            latency_ms=latency_ms,
        )

        return RagAnswer(
            query=query,
            answer=ai_run.output_text or "",
            citations=citations,
            retrieved_chunk_ids=chunk_ids,
            model_name=ai_run.model_name,
            latency_ms=latency_ms,
            no_context_found=False,
            ai_run_id=ai_run.id,
            retrieval_degraded=context.retrieval_degraded,
            memory_degraded=context.memory_degraded,
            degradation_reason=context.degradation_reason,
            injection_chunks_filtered=context.injection_chunks_filtered,
        )

    async def _log_query(
        self,
        *,
        user_id: str,
        project_id: str | None,
        organization_id: str | None,
        query: str,
        answer: str,
        chunk_ids: list[str],
        model_name: str,
        latency_ms: int,
    ) -> None:
        await self.repo.create_query_record(
            user_id=user_id,
            project_id=project_id,
            organization_id=organization_id,
            query=query,
            answer=answer,
            retrieved_chunk_ids=chunk_ids,
            model_name=model_name,
            latency_ms=latency_ms,
        )
        await self.db.commit()
