"""Bounded hierarchical corpus synthesis (still corpus allow-list scoped)."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.modules.identity_access.models import User
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Auto-enqueue when indexed allow-list is large enough to fan out many retrieves.
_ASYNC_DOC_THRESHOLD = 8


class CorpusSynthesisService(ResearchAccessMixin):
    """Document-aware map/reduce synthesis under I1–I5 invariants."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.scope_service = CorpusScopeService(db)
        self.rag_config = RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.rag_config)
        self.answers = RagAnswerService(db, self.rag_config)

    async def synthesize(
        self,
        *,
        corpus_id: str,
        user: User,
        query: str,
        document_subset: list[str] | None = None,
        async_mode: bool | None = None,
    ) -> dict:
        if not self.rag_config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        scope = await self.scope_service.resolve(
            corpus_id=corpus_id,
            user_id=user.id,
            document_subset=document_subset,
        )
        allow_list = list(scope.rag_document_ids)
        if not allow_list:
            return {
                "mode": "sync",
                "answer": "No indexed documents are available for corpus synthesis.",
                "scope": scope.to_dict(),
                "document_findings": [],
                "retrieval_trace_ids": [],
                "documents_total": 0,
                "documents_considered": 0,
                "documents_with_evidence": 0,
                "truncated": False,
                "coverage": {
                    "documents_in_scope": 0,
                    "documents_with_retrieved_evidence": 0,
                    "retrieved_passage_count": 0,
                    "coverage_ratio": 0.0,
                },
            }

        use_async = (
            async_mode
            if async_mode is not None
            else len(allow_list) >= _ASYNC_DOC_THRESHOLD
        )
        if use_async:
            run = await self._enqueue(
                corpus_id=corpus_id,
                project_id=scope.project_id,
                user_id=user.id,
                query=query,
                document_subset=document_subset,
                scope_hash=scope.scope_hash,
            )
            return {
                "mode": "async",
                "run_id": run.id,
                "status": run.status,
                "scope": scope.to_dict(),
                "message": "Corpus synthesis queued; stream progress via research run events.",
            }

        return await self._synthesize_sync(
            user=user,
            query=query,
            scope=scope,
            allow_list=allow_list,
        )

    async def execute_synthesis(self, run_id: str) -> AnalysisRun:
        """Celery / worker entry for async corpus synthesis."""
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {
            AnalysisRunStatus.COMPLETED.value,
            AnalysisRunStatus.CANCELLED.value,
        }:
            return run

        params = loads(run.parameters_json, {}) or {}
        query = str(params.get("query") or "")
        document_subset = params.get("document_subset")
        user_id = run.created_by

        from backend.modules.identity_access.models import User
        from sqlalchemy import select

        user_result = await self.db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        if user is None:
            raise ValueError(f"User {user_id} not found for synthesis run {run_id}")

        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="synthesizing",
            started_at=datetime.now(UTC),
        )
        await self.db.commit()

        try:
            result = await self.synthesize(
                corpus_id=run.corpus_id,
                user=user,
                query=query,
                document_subset=document_subset,
                async_mode=False,
            )
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=datetime.now(UTC),
                results_json=dumps(result),
                metrics_json=dumps(result.get("coverage") or {}),
            )
            await self.db.commit()
        except Exception as exc:
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                error_message=str(exc)[:2000],
                completed_at=datetime.now(UTC),
            )
            await self.db.commit()
            raise
        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def _enqueue(
        self,
        *,
        corpus_id: str,
        project_id: str,
        user_id: str,
        query: str,
        document_subset: list[str] | None,
        scope_hash: str,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.execution_service import ExecutionService

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.CORPUS_SYNTHESIS.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(
                    {
                        "query": query,
                        "document_subset": document_subset,
                        "scope_hash": scope_hash,
                    }
                ),
                created_by=user_id,
            )
        )
        await self.db.commit()
        await ExecutionService.submit(
            db=self.db, run=run, operation="corpus_synthesis", user_id=user_id
        )
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def _synthesize_sync(
        self,
        *,
        user: User,
        query: str,
        scope,
        allow_list: list[str],
    ) -> dict:
        max_docs = min(len(allow_list), self.rag_config.synthesis_max_documents)
        considered_ids = allow_list[:max_docs]
        truncated = len(allow_list) > max_docs
        per_doc = self.rag_config.synthesis_passages_per_document
        findings: list[dict] = []
        trace_ids: list[str] = []
        map_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        # Map phase: per-document findings with raw chunk provenance.
        for rag_document_id in considered_ids:
            outcome = await self.retrieval.retrieve(
                query,
                user_id=user.id,
                project_id=scope.project_id,
                top_k=per_doc,
                filters={
                    "document_ids": [rag_document_id],
                    "owner_scoped": False,
                },
                intent=RetrievalIntent.SYNTHESIS,
                persist_trace=True,
            )
            if outcome.retrieval_trace_id:
                trace_ids.append(outcome.retrieval_trace_id)
            if not outcome.chunks:
                findings.append(
                    {
                        "rag_document_id": rag_document_id,
                        "filename": None,
                        "finding": "No relevant passages retrieved for this document.",
                        "agreements": [],
                        "disagreements": [],
                        "exceptions": [],
                        "missing_evidence": True,
                        "passages": [],
                        "retrieval_trace_id": outcome.retrieval_trace_id,
                    }
                )
                continue

            for chunk in outcome.chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.chunk_id)
                    map_chunks.append(chunk)

            passage_blob = "\n".join(
                f"- [{c.chunk_id}] {c.content[:350]}" for c in outcome.chunks
            )
            findings.append(
                {
                    "rag_document_id": rag_document_id,
                    "filename": outcome.chunks[0].filename,
                    "finding": (
                        f"Document evidence for synthesis query.\n{passage_blob}"
                    ),
                    "agreements": [],
                    "disagreements": [],
                    "exceptions": [],
                    "missing_evidence": False,
                    "passages": [
                        {
                            "chunk_id": c.chunk_id,
                            "snippet": c.content[:400],
                            "score": c.score,
                            "page_number": c.page_number,
                        }
                        for c in outcome.chunks
                    ],
                    "chunk_ids": [c.chunk_id for c in outcome.chunks],
                    "retrieval_trace_id": outcome.retrieval_trace_id,
                }
            )

        # Reduce phase: synthesize ONLY from map findings / mapped chunks (no union retrieve).
        findings_with_evidence = [f for f in findings if not f.get("missing_evidence")]
        reduce_prompt_parts = [
            f"Research question: {query}",
            (
                f"You are reducing {len(findings_with_evidence)} document-level findings "
                f"from {len(considered_ids)} considered documents "
                f"({len(allow_list)} indexed in scope"
                f"{'; truncated' if truncated else ''})."
            ),
            "Explicitly identify agreements, disagreements, exceptions, missing evidence, "
            "and coverage limits. Cite only the provided chunk_ids.",
            "Document findings:",
        ]
        for finding in findings_with_evidence:
            reduce_prompt_parts.append(
                f"\n### {finding.get('filename') or finding['rag_document_id']}\n"
                f"{finding['finding']}"
            )
        reduce_query = "\n".join(reduce_prompt_parts)

        # Build a synthetic outcome from map chunks — never a second corpus-wide retrieve.
        map_outcome = RetrievalOutcome(
            chunks=map_chunks,
            intent=RetrievalIntent.SYNTHESIS,
            fusion_method="hierarchical_map_reduce",
            coverage=None,
            retrieval_trace_id=trace_ids[-1] if trace_ids else None,
            scope_hash=scope.scope_hash,
        )

        answer = await self.answers.answer_from_retrieval(
            reduce_query,
            outcome=map_outcome,
            user=user,
            project_id=scope.project_id,
            document_ids=allow_list,
            include_memory=False,
        )
        if answer.retrieval_trace_id:
            trace_ids.append(answer.retrieval_trace_id)

        docs_with_evidence = len(findings_with_evidence)
        return {
            "mode": "sync",
            "answer": answer.answer,
            "citations": [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "snippet": c.snippet,
                    "citation_number": c.citation_number,
                    "used_in_answer": c.used_in_answer,
                    "page_number": c.page_number,
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
            "scope": scope.to_dict(),
            "document_findings": findings,
            "retrieval_trace_ids": trace_ids,
            "documents_total": len(allow_list),
            "documents_considered": len(considered_ids),
            "documents_with_evidence": docs_with_evidence,
            "truncated": truncated,
            "coverage": {
                "documents_in_scope": len(allow_list),
                "documents_with_retrieved_evidence": docs_with_evidence,
                "retrieved_passage_count": sum(len(f.get("passages") or []) for f in findings),
                "coverage_ratio": (
                    docs_with_evidence / len(allow_list) if allow_list else 0.0
                ),
                "documents_total": len(allow_list),
                "documents_considered": len(considered_ids),
                "truncated": truncated,
            },
            "retrieval_trace_id": answer.retrieval_trace_id,
            "citation_validation_status": answer.citation_validation_status,
        }
