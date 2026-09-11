"""Auditable corpus synthesis with deterministic map and structured reduction."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.identity_access.models import User
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService
from backend.modules.text_research.application.corpus_synthesis_provenance import (
    build_document_map_finding,
    build_synthesis_provenance,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads

# Auto-enqueue when indexed allow-list is large enough to fan out many retrieves.
_ASYNC_DOC_THRESHOLD = 8


class CorpusSynthesisService(ResearchAccessMixin):
    """Document-aware deterministic-map/LLM-reduction under I1–I5 invariants."""

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
            provenance = build_synthesis_provenance(
                original_question=query,
                evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                corpus_id=scope.corpus_id,
                project_id=scope.project_id,
                user_id=user.id,
                documents_in_scope=[],
                documents_considered=[],
                omitted_documents=[],
                findings=[],
                retrieval_trace_ids=[],
            )
            return {
                "mode": "sync",
                "answer": "No indexed documents are available for corpus synthesis.",
                "scope": scope.to_dict(),
                "evidence_revision_hash": getattr(scope, "evidence_revision_hash", None),
                "document_findings": [],
                "retrieval_trace_ids": [],
                "documents_total": 0,
                "documents_considered": 0,
                "documents_with_evidence": 0,
                "truncated": False,
                "omitted_document_count": 0,
                "omitted_document_ids": [],
                "synthesis_provenance": provenance,
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
                evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                documents_in_scope=allow_list,
                documents_considered=allow_list,
                omitted_documents=[],
            )
            return {
                "mode": "async",
                "run_id": run.id,
                "status": run.status,
                "scope": scope.to_dict(),
                "evidence_revision_hash": getattr(scope, "evidence_revision_hash", None),
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

        from sqlalchemy import select

        from backend.modules.identity_access.models import User

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
                evidence_revision_hash=result.get("evidence_revision_hash"),
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
        evidence_revision_hash: str | None,
        documents_in_scope: list[str],
        documents_considered: list[str],
        omitted_documents: list[dict[str, str]],
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
                        "evidence_revision_hash": evidence_revision_hash,
                        "synthesis_provenance": build_synthesis_provenance(
                            original_question=query,
                            evidence_revision_hash=evidence_revision_hash,
                            corpus_id=corpus_id,
                            project_id=project_id,
                            user_id=user_id,
                            documents_in_scope=documents_in_scope,
                            documents_considered=documents_considered,
                            omitted_documents=omitted_documents,
                            findings=[],
                            retrieval_trace_ids=[],
                        ),
                    }
                ),
                evidence_revision_hash=evidence_revision_hash,
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
        considered_ids = list(allow_list)
        truncated = False
        per_doc = self.rag_config.synthesis_passages_per_document
        findings: list[dict] = []
        trace_ids: list[str] = []
        map_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        # Map phase: bounded, deterministic evidence collection per document.
        batch_size = max(1, int(getattr(self.rag_config, "synthesis_batch_size", 8)))
        for batch_start in range(0, len(considered_ids), batch_size):
            document_batch = considered_ids[batch_start : batch_start + batch_size]
            for rag_document_id in document_batch:
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
                    evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                )
                if outcome.retrieval_trace_id:
                    trace_ids.append(outcome.retrieval_trace_id)
                finding = build_document_map_finding(
                    question=query,
                    rag_document_id=rag_document_id,
                    outcome=outcome,
                )
                findings.append(finding)
                for chunk_id in finding["supporting_chunk_ids"]:
                    chunk = next(c for c in outcome.chunks if c.chunk_id == chunk_id)
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        map_chunks.append(chunk)

        # Reduce phase: synthesize only from structured map findings (no union retrieve).
        findings_with_evidence = [f for f in findings if not f.get("missing_evidence")]
        reduce_query = "\n".join(
            [
                f"Research question: {query}",
                (
                    f"Reduce {len(findings_with_evidence)} structured document findings "
                    f"from {len(considered_ids)} considered documents "
                    f"({len(allow_list)} indexed in scope"
                    f"{'; truncated' if truncated else ''})."
                ),
                "Identify agreements, disagreements, exceptions, missing evidence, and "
                "coverage limits. Cite only supporting_chunk_ids from the JSON findings.",
                "Structured map findings JSON:",
                dumps(findings_with_evidence),
            ]
        )

        # Build a synthetic outcome from map chunks — never a second corpus-wide retrieve.
        map_outcome = RetrievalOutcome(
            chunks=map_chunks,
            intent=RetrievalIntent.SYNTHESIS,
            fusion_method="deterministic_map_reduce",
            coverage=None,
            retrieval_trace_id=None,
            scope_hash=scope.scope_hash,
            evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
        )

        answer = await self.answers.answer_from_retrieval(
            reduce_query,
            outcome=map_outcome,
            user=user,
            project_id=scope.project_id,
            document_ids=allow_list,
            include_memory=False,
        )
        rag_to_corpus_document = {
            rag_document_id: corpus_document_id
            for rag_document_id, corpus_document_id in zip(
                getattr(scope, "rag_document_ids", []),
                getattr(scope, "corpus_document_ids", []),
                strict=False,
            )
        }
        docs_with_evidence = len(findings_with_evidence)
        provenance = build_synthesis_provenance(
            original_question=query,
            evidence_revision_hash=getattr(answer, "evidence_revision_hash", None)
            or getattr(scope, "evidence_revision_hash", None),
            corpus_id=scope.corpus_id,
            project_id=scope.project_id,
            user_id=user.id,
            documents_in_scope=allow_list,
            documents_considered=considered_ids,
            omitted_documents=[],
            findings=findings,
            retrieval_trace_ids=trace_ids,
            reduction_prompt=reduce_query,
            answer=answer,
            created_at=datetime.now(UTC),
        )
        return {
            "mode": "sync",
            "answer": answer.answer,
            "citations": [
                {
                    "document_id": c.document_id,
                    "corpus_document_id": rag_to_corpus_document.get(c.document_id),
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "snippet": c.snippet,
                    "citation_number": c.citation_number,
                    "used_in_answer": c.used_in_answer,
                    "page_number": c.page_number,
                    "section_heading": c.section_heading,
                    "char_start": getattr(c, "char_start", None),
                    "char_end": getattr(c, "char_end", None),
                    "source_span_ids": getattr(c, "source_span_ids", None),
                    "parent_context_id": getattr(c, "parent_context_id", None),
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
            "evidence_revision_hash": getattr(answer, "evidence_revision_hash", None),
            "document_findings": findings,
            "retrieval_trace_ids": trace_ids,
            "documents_total": len(allow_list),
            "documents_considered": len(considered_ids),
            "documents_with_evidence": docs_with_evidence,
            "truncated": truncated,
            "omitted_document_count": 0,
            "omitted_document_ids": [],
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
            # A synthesis has no single retrieval trace; use the per-document
            # trace IDs in the explicit provenance artifact instead.
            "retrieval_trace_id": None,
            "citation_validation_status": answer.citation_validation_status,
            "synthesis_provenance": provenance,
        }
