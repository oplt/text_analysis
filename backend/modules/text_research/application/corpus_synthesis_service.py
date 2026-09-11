"""Bounded hierarchical corpus synthesis (still corpus allow-list scoped)."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.modules.identity_access.models import User
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import RetrievalIntent
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
    """Document-aware synthesis under the same I1–I5 invariants as Ask Corpus."""

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
        # I1: never None; empty short-circuits before fan-out / Celery
        if not allow_list:
            return {
                "mode": "sync",
                "answer": "No indexed documents are available for corpus synthesis.",
                "scope": scope.to_dict(),
                "document_findings": [],
                "retrieval_trace_ids": [],
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
        per_doc = self.rag_config.synthesis_passages_per_document
        findings: list[dict] = []
        trace_ids: list[str] = []

        for rag_document_id in allow_list[:max_docs]:
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
                continue
            findings.append(
                {
                    "rag_document_id": rag_document_id,
                    "filename": outcome.chunks[0].filename,
                    "passages": [
                        {
                            "chunk_id": c.chunk_id,
                            "snippet": c.content[:400],
                            "score": c.score,
                            "page_number": c.page_number,
                        }
                        for c in outcome.chunks
                    ],
                    "retrieval_trace_id": outcome.retrieval_trace_id,
                }
            )

        union_outcome = await self.retrieval.retrieve(
            query,
            user_id=user.id,
            project_id=scope.project_id,
            filters={
                "document_ids": allow_list,
                "owner_scoped": False,
            },
            intent=RetrievalIntent.SYNTHESIS,
            persist_trace=True,
        )
        if union_outcome.retrieval_trace_id:
            trace_ids.append(union_outcome.retrieval_trace_id)

        answer = await self.answers.answer_from_retrieval(
            (
                f"{query}\n\nSynthesize across the corpus. Mention variation and disagreement. "
                f"Evidence was retrieved from {len(findings)} of {scope.indexed_count} indexed documents."
            ),
            outcome=union_outcome,
            user=user,
            project_id=scope.project_id,
            document_ids=allow_list,
            include_memory=False,
        )

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
                }
                for c in answer.citations
            ],
            "scope": scope.to_dict(),
            "document_findings": findings,
            "retrieval_trace_ids": trace_ids,
            "coverage": {
                "documents_in_scope": scope.indexed_count,
                "documents_with_retrieved_evidence": len(findings),
                "retrieved_passage_count": sum(len(f["passages"]) for f in findings),
                "coverage_ratio": (
                    len(findings) / scope.indexed_count if scope.indexed_count else 0.0
                ),
            },
            "retrieval_trace_id": answer.retrieval_trace_id,
        }
