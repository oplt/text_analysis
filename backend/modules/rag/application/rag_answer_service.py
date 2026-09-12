from __future__ import annotations

from time import perf_counter

from backend.lib.generation_port import AiServiceGenerationPort, GenerationPort
from backend.lib.project_access import ProjectAccessPort, SqlAlchemyProjectAccessPort
from backend.modules.identity_access.models import User
from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.infrastructure.memory_config import MemoryConfig
from backend.modules.rag.application.citation_service import CitationService
from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.application.prompt_context_service import PromptContextService
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.citation_validation_context import CitationValidationContext
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RagAnswer, RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession


def _revision_by_document(chunks: list[RetrievedChunk]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in chunks:
        revision = chunk.index_revision_id or (chunk.metadata or {}).get("revision_id")
        if not isinstance(revision, str) or not revision:
            continue
        prior = mapping.get(chunk.document_id)
        if prior is None:
            mapping[chunk.document_id] = revision
        elif prior != revision:
            # Conflicting revisions for one document — omit so validation fails closed.
            mapping.pop(chunk.document_id, None)
    return mapping


def _build_citation_validation_context(
    *,
    user_id: str,
    project_id: str | None,
    document_ids: list[str] | None,
    outcome: RetrievalOutcome,
    chunks: list[RetrievedChunk],
    corpus_id: str | None = None,
) -> CitationValidationContext | None:
    """Bind citation checks to frozen revision identity when retrieval carried one."""
    if not outcome.index_revision_ids and not outcome.evidence_revision_hash:
        return None
    return CitationValidationContext(
        user_id=user_id,
        project_id=project_id,
        corpus_id=corpus_id,
        evidence_revision_hash=outcome.evidence_revision_hash,
        index_revision_ids=list(outcome.index_revision_ids),
        revision_by_document=_revision_by_document(chunks),
        allowed_document_ids=document_ids,
        retrieval_trace_id=outcome.retrieval_trace_id,
    )


NO_CONTEXT_ANSWER = (
    "I could not find relevant document context for your question in the indexed documents."
)
CITATION_FAILURE_ANSWER = (
    "I could not generate a fully cited answer from the retrieved evidence. "
    "The retrieved evidence is shown separately below."
)

STRUCTURED_CLAIM_INSTRUCTION = (
    "Respond with a single JSON object only, of the form:\n"
    '{"claims":[{"text":"...","chunk_ids":["chunk-id"]}],'
    '"no_evidence":false,"insufficient_evidence_reason":null}.\n'
    "Every substantive answer statement must be one cited claim. "
    "Every claim must cite chunk_ids from the provided sources. "
    "Do not invent chunk IDs. If evidence is insufficient, set no_evidence to true, "
    "claims to [], and explain why in insufficient_evidence_reason."
)

STRUCTURED_REPAIR_INSTRUCTION = (
    "Your previous reply failed citation validation. "
    "Reply again with ONLY the JSON object "
    '{"claims":[{"text":"...","chunk_ids":["chunk-id"]}],'
    '"no_evidence":false,"insufficient_evidence_reason":null}. '
    "If evidence is insufficient, set no_evidence to true and claims to []."
)


class RagAnswerService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.config)
        self.context_builder = RagContextBuilder(CitationService())
        self.citation_validator = CitationValidationService()
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
        intent: RetrievalIntent | str | None = None,
        owner_scoped: bool = True,
        conversation_id: str | None = None,
        include_memory: bool = True,
        commit: bool = True,
    ) -> RagAnswer:
        """Convenience wrapper: retrieve once, then answer_from_retrieval (I4)."""
        if not self.config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        if project_id:
            await self.project_access.ensure_project_access(user.id, project_id)

        filters: dict = {"owner_scoped": owner_scoped}
        if document_ids is not None:
            filters["document_ids"] = document_ids

        outcome = await self.retrieval.retrieve(
            query,
            user_id=user.id,
            project_id=project_id,
            filters=filters,
            intent=intent,
            conversation_id=conversation_id,
            persist_trace=True,
        )
        return await self.answer_from_retrieval(
            query,
            outcome=outcome,
            user=user,
            project_id=project_id,
            run_id=run_id,
            agent_id=agent_id,
            document_ids=document_ids,
            organization_id=organization_id,
            include_memory=include_memory,
            commit=commit,
        )

    async def answer_from_retrieval(
        self,
        query: str,
        *,
        outcome: RetrievalOutcome,
        user: User,
        project_id: str | None,
        run_id: str | None = None,
        agent_id: str | None = None,
        document_ids: list[str] | None = None,
        organization_id: str | None = None,
        include_memory: bool = True,
        conversation_context: str | None = None,
        resolved_retrieval_query: str | None = None,
        context_message_ids: list[str] | None = None,
        commit: bool = True,
    ) -> RagAnswer:
        """Generate from an existing retrieval outcome — never re-retrieves (I4)."""
        if not self.config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        started = perf_counter()
        user_id = user.id
        bounded_chunks = self.context_builder.trim_chunks_to_token_budget(
            outcome.chunks,
            max_tokens=self.config.max_context_tokens,
            overlap_dedupe_threshold=getattr(self.config, "context_overlap_dedupe_threshold", 0.8),
            ordering_policy=getattr(self.config, "context_ordering_policy", "relevance"),
        )

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
            if commit:
                await self.db.commit()
            return RagAnswer(
                query=query,
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                retrieved_chunk_ids=[],
                model_name="none",
                latency_ms=latency_ms,
                no_context_found=True,
                ai_run_id=None,
                retrieval_degraded=outcome.degraded,
                memory_degraded=False,
                degradation_reason=outcome.degradation_reason,
                injection_chunks_filtered=outcome.injection_chunks_filtered,
                retrieval_trace_id=outcome.retrieval_trace_id,
                coverage=outcome.coverage,
                citation_validation_status="valid",
                evidence_revision_hash=outcome.evidence_revision_hash,
                resolved_retrieval_query=resolved_retrieval_query or query,
                context_message_ids=list(context_message_ids or []),
                index_revision_ids=list(outcome.index_revision_ids),
            )

        document_context = self.context_builder.build_document_context_block(bounded_chunks)
        memory_context = ""
        memory_degraded = False
        if include_memory and self.memory_config.enabled:
            try:
                from backend.modules.memory.domain.models import MemorySearchRequest

                memory_context, _, memory_degraded = await self.memory.recall_for_prompt(
                    MemorySearchRequest(
                        user_id=user_id,
                        agent_id=agent_id or "default",
                        query=query,
                        run_id=run_id,
                        project_id=project_id,
                    )
                )
            except Exception:
                memory_degraded = True

        system_context = self.context_builder.assemble_agent_system_context(
            memory_context=memory_context or None,
            document_context=document_context or None,
        )
        if conversation_context:
            system_context = f"{system_context or ''}\n\n{conversation_context}".strip()
        combined = f"{system_context or ''}\n\n{STRUCTURED_CLAIM_INSTRUCTION}".strip()
        chunk_ids = [c.chunk_id for c in bounded_chunks]

        ai_run = await self.generation.run_rag_answer(
            user,
            query=query,
            combined_context=combined,
            retrieved_chunk_ids=chunk_ids,
            retrieval_degraded=outcome.degraded,
            memory_degraded=memory_degraded,
            degradation_reason=outcome.degradation_reason,
            injection_chunks_filtered=outcome.injection_chunks_filtered,
            commit=commit,
        )

        citation_context = _build_citation_validation_context(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            outcome=outcome,
            chunks=bounded_chunks,
        )
        validated = self.citation_validator.validate(
            raw_output=ai_run.output_text or "",
            retrieved_chunks=bounded_chunks,
            allowed_document_ids=document_ids,
            expected_index_revision_ids=outcome.index_revision_ids or None,
            context=citation_context,
        )

        # At most one bounded repair/retry for any citation-validation failure.
        if validated.citation_validation_failed:
            repair_context = (
                f"{combined}\n\n{STRUCTURED_REPAIR_INSTRUCTION}\n\n"
                f"Previous invalid output:\n{(ai_run.output_text or '')[:2000]}"
            )
            ai_run = await self.generation.run_rag_answer(
                user,
                query=query,
                combined_context=repair_context,
                retrieved_chunk_ids=chunk_ids,
                retrieval_degraded=outcome.degraded,
                memory_degraded=memory_degraded,
                degradation_reason=outcome.degradation_reason,
                injection_chunks_filtered=outcome.injection_chunks_filtered,
                commit=commit,
            )
            validated = self.citation_validator.validate(
                raw_output=ai_run.output_text or "",
                retrieved_chunks=bounded_chunks,
                allowed_document_ids=document_ids,
                expected_index_revision_ids=outcome.index_revision_ids or None,
                context=citation_context,
            )

        if validated.citation_validation_failed:
            validated.answer = CITATION_FAILURE_ANSWER
            validated.claims = []
            for citation in validated.citations:
                citation.used_in_answer = False
                citation.citation_number = None

        latency_ms = int((perf_counter() - started) * 1000)
        metrics.rag_answer_latency_ms.observe(latency_ms)

        await self._log_query(
            user_id=user_id,
            project_id=project_id,
            organization_id=organization_id,
            query=query,
            answer=validated.answer,
            chunk_ids=chunk_ids,
            model_name=ai_run.model_name,
            latency_ms=latency_ms,
        )
        if commit:
            await self.db.commit()

        prompt_template_id = getattr(ai_run, "prompt_template_id", None)
        prompt_version_id = getattr(ai_run, "prompt_version_id", None)

        return RagAnswer(
            query=query,
            answer=validated.answer,
            citations=validated.citations,
            retrieved_chunk_ids=chunk_ids,
            model_name=ai_run.model_name,
            latency_ms=latency_ms,
            no_context_found=validated.no_evidence,
            ai_run_id=ai_run.id,
            retrieval_degraded=outcome.degraded,
            memory_degraded=memory_degraded,
            degradation_reason=outcome.degradation_reason,
            injection_chunks_filtered=outcome.injection_chunks_filtered,
            claims=validated.claims,
            retrieval_trace_id=outcome.retrieval_trace_id,
            coverage=outcome.coverage,
            citation_validation_failed=validated.citation_validation_failed,
            citation_validation_status=validated.citation_validation_status,
            evidence_revision_hash=outcome.evidence_revision_hash,
            prompt_template_id=prompt_template_id,
            prompt_version_id=prompt_version_id,
            resolved_retrieval_query=resolved_retrieval_query or query,
            context_message_ids=list(context_message_ids or []),
            index_revision_ids=list(outcome.index_revision_ids),
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
