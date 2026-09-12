"""Auditable corpus synthesis with deterministic map and structured reduction."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.identity_access.models import User
from backend.modules.rag.application.evidence_revision import StaleEvidenceRevisionError
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.corpus_scope_service import (
    CorpusScopeService,
    CorpusScopeSnapshot,
)
from backend.modules.text_research.application.corpus_synthesis_provenance import (
    build_document_map_finding,
    build_synthesis_provenance,
)
from backend.modules.text_research.application.hierarchical_reduce import (
    filter_claims_to_supporting_chunks,
    pack_reduction_nodes,
    write_reduce_checkpoint,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads


class CorpusSynthesisService(ResearchAccessMixin):
    """Document-aware deterministic-map/LLM-reduction under I1–I5 invariants."""

    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self.scope_service = CorpusScopeService(db)
        self.rag_config = RagConfig.from_settings()
        self.retrieval = RetrievalService(db, self.rag_config)
        self.answers = RagAnswerService(db, self.rag_config)

    async def _rag_to_corpus_document_map(self, scope) -> dict[str, str]:
        """Map citations through immutable scope bindings, never array position."""
        bindings = getattr(scope, "document_bindings", None)
        mapping_status = getattr(scope, "mapping_status", "ok")
        if mapping_status == "unverifiable":
            raise HTTPException(
                status_code=409,
                detail="Corpus scope mapping is unverifiable; refuse positional ID pairing",
            )
        if isinstance(bindings, list) and bindings:
            return {
                binding.rag_document_id: binding.corpus_document_id
                for binding in bindings
                if binding.rag_document_id and binding.availability == "indexed"
            }
        # Empty bindings are only valid when the scope itself has no indexed docs.
        if list(getattr(scope, "rag_document_ids", []) or []):
            raise HTTPException(
                status_code=409,
                detail="Corpus scope mapping is unverifiable; refuse positional ID pairing",
            )
        return {}

    def _enrich_citations(
        self,
        citations: list,
        *,
        rag_to_corpus: dict[str, str],
    ) -> list[dict]:
        enriched: list[dict] = []
        for citation in citations:
            corpus_document_id = rag_to_corpus.get(citation.document_id)
            if corpus_document_id is None:
                raise HTTPException(
                    status_code=409,
                    detail="Citation document is not present in the frozen corpus scope",
                )
            enriched.append(
                {
                    "document_id": citation.document_id,
                    "corpus_document_id": corpus_document_id,
                    "chunk_id": citation.chunk_id,
                    "filename": citation.filename,
                    "snippet": citation.snippet,
                    "citation_number": citation.citation_number,
                    "used_in_answer": citation.used_in_answer,
                    "page_number": citation.page_number,
                    "section_heading": citation.section_heading,
                    "char_start": getattr(citation, "char_start", None),
                    "char_end": getattr(citation, "char_end", None),
                    "source_span_ids": getattr(citation, "source_span_ids", None),
                    "parent_context_id": getattr(citation, "parent_context_id", None),
                    "offset_coordinate_system": getattr(
                        citation, "offset_coordinate_system", None
                    ),
                    "offset_scope": getattr(citation, "offset_scope", None),
                    "offset_scope_id": getattr(citation, "offset_scope_id", None),
                    "source_spans": getattr(citation, "source_spans", None),
                }
            )
        return enriched

    def _freeze_scope(self, scope) -> dict:
        """Serialize the exact evidence authority required by an async run."""
        bindings = getattr(scope, "document_bindings", [])
        revisions = {
            binding.rag_document_id: binding.index_revision_id
            for binding in bindings
            if binding.availability == "indexed" and binding.rag_document_id
        }
        missing_revisions = [
            document_id for document_id in scope.rag_document_ids if not revisions.get(document_id)
        ]
        if missing_revisions:
            raise HTTPException(
                status_code=409,
                detail="historical_revision_unavailable: frozen synthesis missing index revision",
            )
        return {
            "scope": scope.to_dict(),
            "retrieval_config": {
                "index_version": scope.index_version,
                "retrieval_version": scope.retrieval_version,
                "synthesis_passages_per_document": self.rag_config.synthesis_passages_per_document,
            },
        }

    @staticmethod
    def _frozen_scope_from_run(params: dict, run: AnalysisRun) -> CorpusScopeSnapshot:
        frozen = params.get("frozen_synthesis_scope")
        if not isinstance(frozen, dict) or not isinstance(frozen.get("scope"), dict):
            raise ValueError(
                "historical_revision_unavailable: frozen synthesis scope is missing"
            )
        scope = CorpusScopeSnapshot.from_dict(frozen["scope"])
        if scope.corpus_id != run.corpus_id or scope.project_id != run.project_id:
            raise ValueError("Frozen synthesis scope boundary mismatch")
        if params.get("scope_hash") != scope.scope_hash:
            raise ValueError("Frozen synthesis scope hash mismatch")
        if params.get("evidence_revision_hash") != scope.evidence_revision_hash:
            raise ValueError("Frozen synthesis evidence revision mismatch")
        if getattr(scope, "mapping_status", "ok") == "unverifiable":
            raise ValueError(
                "historical_revision_unavailable: frozen synthesis mapping is unverifiable"
            )
        return scope

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
            else len(allow_list)
            >= max(1, int(getattr(self.rag_config, "synthesis_async_document_threshold", 8)))
        )
        if use_async:
            run = await self._enqueue(
                scope=scope,
                user_id=user.id,
                query=query,
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
        from backend.modules.text_research.application.run_lifecycle import (
            RunCancelledError,
            TERMINAL_RUN_STATUSES,
            complete_if_active,
            ensure_not_cancelled,
            fail_if_active,
        )

        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in TERMINAL_RUN_STATUSES:
            return run

        params = loads(run.parameters_json, {}) or {}
        query = str(params.get("query") or "")
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
            run = await ensure_not_cancelled(self.repo, run)
            scope = self._frozen_scope_from_run(params, run)
            # Existence of the revision IDs alone is insufficient: the queued
            # evidence hash must still describe those exact retained chunks.
            # This also fails closed for incomplete legacy bindings.
            await self.scope_service.frozen_revision_ids(scope)
            run = await ensure_not_cancelled(self.repo, run)
            result = await self._synthesize_sync(
                user=user,
                query=query,
                scope=scope,
                allow_list=list(scope.rag_document_ids),
            )
            run = await ensure_not_cancelled(self.repo, run)
            await complete_if_active(
                self.repo,
                run,
                progress_stage="completed",
                completed_at=datetime.now(UTC),
                results_json=dumps(result),
                metrics_json=dumps(result.get("coverage") or {}),
                evidence_revision_hash=result.get("evidence_revision_hash"),
            )
            await self.db.commit()
        except RunCancelledError:
            await self.db.commit()
        except Exception as exc:
            await fail_if_active(
                self.repo,
                run,
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
        scope,
        user_id: str,
        query: str,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.execution_service import ExecutionService

        frozen_synthesis_scope = self._freeze_scope(scope)
        documents_in_scope = list(scope.rag_document_ids)
        evidence_revision_hash = scope.evidence_revision_hash
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=scope.project_id,
                corpus_id=scope.corpus_id,
                run_type=AnalysisRunType.CORPUS_SYNTHESIS.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(
                    {
                        "query": query,
                        "scope_hash": scope.scope_hash,
                        "evidence_revision_hash": evidence_revision_hash,
                        "frozen_synthesis_scope": frozen_synthesis_scope,
                        "synthesis_provenance": build_synthesis_provenance(
                            original_question=query,
                            evidence_revision_hash=evidence_revision_hash,
                            corpus_id=scope.corpus_id,
                            project_id=scope.project_id,
                            user_id=user_id,
                            documents_in_scope=documents_in_scope,
                            documents_considered=documents_in_scope,
                            omitted_documents=[],
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
        revision_by_document = {
            binding.rag_document_id: binding.index_revision_id
            for binding in getattr(scope, "document_bindings", [])
            if binding.availability == "indexed"
            and binding.rag_document_id
            and binding.index_revision_id
        }

        # Map phase: bounded, deterministic evidence collection per document.
        batch_size = max(1, int(getattr(self.rag_config, "synthesis_batch_size", 8)))
        for batch_start in range(0, len(considered_ids), batch_size):
            document_batch = considered_ids[batch_start : batch_start + batch_size]
            for rag_document_id in document_batch:
                revision_id = revision_by_document.get(rag_document_id)
                if not revision_id:
                    findings.append(
                        {
                            "rag_document_id": rag_document_id,
                            "supporting_chunk_ids": [],
                            "missing_evidence": True,
                            "missing_evidence_reason": "historical_revision_unavailable",
                            "map_failure": True,
                        }
                    )
                    continue
                try:
                    outcome = await self.retrieval.retrieve(
                        query,
                        user_id=user.id,
                        project_id=scope.project_id,
                        top_k=per_doc,
                        filters={
                            "document_ids": [rag_document_id],
                            "owner_scoped": False,
                            "index_revision_ids": [revision_id],
                        },
                        intent=RetrievalIntent.SYNTHESIS,
                        persist_trace=True,
                        evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                    )
                except StaleEvidenceRevisionError as exc:
                    findings.append(
                        {
                            "rag_document_id": rag_document_id,
                            "supporting_chunk_ids": [],
                            "missing_evidence": True,
                            "missing_evidence_reason": str(exc.code),
                            "map_failure": True,
                        }
                    )
                    continue
                except Exception as exc:
                    findings.append(
                        {
                            "rag_document_id": rag_document_id,
                            "supporting_chunk_ids": [],
                            "missing_evidence": True,
                            "missing_evidence_reason": f"map_retrieval_failed:{type(exc).__name__}",
                            "map_failure": True,
                        }
                    )
                    continue
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

        # Reduce structured map findings in bounded fan-in batches.  The legacy
        # ``synthesis_max_documents`` setting is deliberately a reduction limit,
        # never a corpus exclusion limit.
        findings_with_evidence = [f for f in findings if not f.get("missing_evidence")]
        reduction_nodes = findings_with_evidence or findings
        chunks_by_id = {chunk.chunk_id: chunk for chunk in map_chunks}
        fan_in = max(1, int(getattr(self.rag_config, "synthesis_max_documents", 25)))
        reduce_token_budget = max(
            512, int(getattr(self.rag_config, "synthesis_reduce_token_budget", 6000))
        )
        reduction_stages: list[dict] = []
        reduction_checkpoints: dict[str, dict] = {}
        final_answer = None
        final_reduce_query = ""
        level = 1

        while reduction_nodes:
            next_nodes: list[dict] = []
            for batch_index, batch in enumerate(
                pack_reduction_nodes(
                    reduction_nodes,
                    max_tokens=reduce_token_budget,
                    max_items=fan_in,
                )
            ):
                batch_chunk_ids = list(
                    dict.fromkeys(
                        chunk_id
                        for finding in batch
                        for chunk_id in finding.get("supporting_chunk_ids", [])
                    )
                )
                batch_chunks = [
                    chunks_by_id[chunk_id]
                    for chunk_id in batch_chunk_ids
                    if chunk_id in chunks_by_id
                ]
                final_reduce_query = "\n".join(
                    [
                        f"Research question: {query}",
                        (
                            f"Reduction level {level}: synthesize {len(batch)} structured "
                            f"findings from {len(considered_ids)} considered documents."
                        ),
                        "Identify agreements, disagreements, exceptions, missing evidence, and "
                        "coverage limits. Cite only supporting_chunk_ids from the JSON findings.",
                        "Structured findings JSON:",
                        dumps(batch),
                    ]
                )
                batch_outcome = RetrievalOutcome(
                    chunks=batch_chunks,
                    intent=RetrievalIntent.SYNTHESIS,
                    fusion_method="deterministic_map_reduce",
                    scope_hash=scope.scope_hash,
                    evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
                )
                batch_answer = await self.answers.answer_from_retrieval(
                    final_reduce_query,
                    outcome=batch_outcome,
                    user=user,
                    project_id=scope.project_id,
                    document_ids=allow_list,
                    include_memory=False,
                )
                supporting_allow = set(batch_chunk_ids)
                kept_claims, rejected_claims = filter_claims_to_supporting_chunks(
                    batch_answer.claims, supporting_allow
                )
                claimed_chunk_ids = list(
                    dict.fromkeys(
                        chunk_id
                        for claim in kept_claims
                        for chunk_id in claim.get("chunk_ids", [])
                    )
                )
                node = {
                    "reduction_level": level,
                    "source_document_ids": list(
                        dict.fromkeys(
                            document_id
                            for finding in batch
                            for document_id in finding.get(
                                "source_document_ids", [finding.get("rag_document_id")]
                            )
                            if document_id
                        )
                    ),
                    "supporting_chunk_ids": claimed_chunk_ids or batch_chunk_ids,
                    "claims": kept_claims,
                    "rejected_claims": rejected_claims,
                    "missing_evidence": batch_answer.no_context_found,
                }
                next_nodes.append(node)
                reduction_checkpoints = write_reduce_checkpoint(
                    reduction_checkpoints,
                    level=level,
                    batch=batch_index,
                    payload={
                        "supporting_chunk_ids": node["supporting_chunk_ids"],
                        "source_document_ids": node["source_document_ids"],
                        "claim_count": len(kept_claims),
                        "rejected_claim_count": len(rejected_claims),
                    },
                )
                reduction_stages.append(
                    {
                        "level": level,
                        "batch": batch_index,
                        "input_count": len(batch),
                        "source_document_ids": node["source_document_ids"],
                        "supporting_chunk_ids": node["supporting_chunk_ids"],
                    }
                )
                final_answer = batch_answer
            if len(next_nodes) == 1:
                break
            reduction_nodes = next_nodes
            level += 1

        if final_answer is None:
            empty_outcome = RetrievalOutcome(
                chunks=[],
                intent=RetrievalIntent.SYNTHESIS,
                fusion_method="deterministic_map_reduce",
                scope_hash=scope.scope_hash,
                evidence_revision_hash=getattr(scope, "evidence_revision_hash", None),
            )
            final_reduce_query = f"Research question: {query}\nNo document evidence was retrieved."
            final_answer = await self.answers.answer_from_retrieval(
                final_reduce_query,
                outcome=empty_outcome,
                user=user,
                project_id=scope.project_id,
                document_ids=allow_list,
                include_memory=False,
            )
        answer = final_answer
        rag_to_corpus_document = await self._rag_to_corpus_document_map(scope)
        docs_with_evidence = len(findings_with_evidence)
        omitted_documents = [
            {
                "rag_document_id": finding["rag_document_id"],
                "reason": finding["missing_evidence_reason"],
            }
            for finding in findings
            if finding.get("map_failure") and finding.get("missing_evidence_reason")
        ]
        provenance = build_synthesis_provenance(
            original_question=query,
            evidence_revision_hash=getattr(answer, "evidence_revision_hash", None)
            or getattr(scope, "evidence_revision_hash", None),
            corpus_id=scope.corpus_id,
            project_id=scope.project_id,
            user_id=user.id,
            documents_in_scope=allow_list,
            documents_considered=considered_ids,
            omitted_documents=omitted_documents,
            findings=findings,
            retrieval_trace_ids=trace_ids,
            reduction_prompt=final_reduce_query,
            answer=answer,
            created_at=datetime.now(UTC),
        )
        provenance["reduction_stages"] = reduction_stages
        provenance["reduction_checkpoints"] = reduction_checkpoints
        map_supporting_ids = set(seen_chunk_ids)
        filtered_claims, rejected_final_claims = filter_claims_to_supporting_chunks(
            [
                {
                    "text": claim.text,
                    "chunk_ids": claim.chunk_ids,
                    "citation_numbers": claim.citation_numbers,
                }
                for claim in answer.claims
            ],
            map_supporting_ids,
        )
        provenance["rejected_claim_escapes"] = rejected_final_claims
        return {
            "mode": "sync",
            "answer": answer.answer,
            "citations": self._enrich_citations(
                answer.citations, rag_to_corpus=rag_to_corpus_document
            ),
            "claims": filtered_claims,
            "scope": scope.to_dict(),
            "evidence_revision_hash": getattr(answer, "evidence_revision_hash", None),
            "document_findings": findings,
            "retrieval_trace_ids": trace_ids,
            "documents_total": len(allow_list),
            "documents_considered": len(considered_ids),
            "documents_with_evidence": docs_with_evidence,
            "truncated": truncated,
            "omitted_document_count": len(omitted_documents),
            "omitted_document_ids": [item["rag_document_id"] for item in omitted_documents],
            "coverage": {
                "documents_in_scope": len(allow_list),
                "documents_with_retrieved_evidence": docs_with_evidence,
                "retrieved_passage_count": sum(len(f.get("passages") or []) for f in findings),
                "coverage_ratio": (docs_with_evidence / len(allow_list) if allow_list else 0.0),
                "documents_total": len(allow_list),
                "documents_considered": len(considered_ids),
                "truncated": truncated,
            },
            # A synthesis has no single retrieval trace; use the per-document
            # trace IDs in the explicit provenance artifact instead.
            "retrieval_trace_id": None,
            "citation_validation_status": answer.citation_validation_status,
            "synthesis_provenance": provenance,
            "reduction_checkpoints": list(reduction_checkpoints.values()),
        }
