from __future__ import annotations

import hashlib
import json
import logging
from time import perf_counter

from backend.lib.retrieval_cache import get_cached_retrieval, set_cached_retrieval
from backend.lib.vectors import can_index_embedding
from backend.modules.rag.application.document_scope import document_ids_is_empty_allow_list
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.evidence_revision import (
    build_retrieved_evidence_revision_hash,
)
from backend.modules.rag.application.parent_context_service import expand_parent_chunks
from backend.modules.rag.application.query_expansion_service import (
    QUERY_EXPANSION_VERSION,
    QueryExpansionService,
)
from backend.modules.rag.application.reranker_port import build_reranker
from backend.modules.rag.application.retrieval_filters import exclude_injection_flagged_chunks
from backend.modules.rag.application.retrieval_fusion import reciprocal_rank_fusion
from backend.modules.rag.application.retrieval_planner import plan_retrieval
from backend.modules.rag.application.source_diversifier import (
    diversify_by_document,
    filter_to_allow_list,
)
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievalCoverage, RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.pgvector_adapter import _parse_scope
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.rag.infrastructure.vector_store_adapter import build_vector_store
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def scope_hash_for_document_ids(document_ids: list[str] | None) -> str | None:
    if document_ids is None:
        return None
    payload = json.dumps(sorted(document_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _filters_for_cache(
    document_ids: list[str] | None,
    *,
    owner_scoped: bool,
    intent: str,
    retrieval_mode: str = "hybrid",
) -> dict:
    """Always include document_ids key when not None so [] ≠ omitted (I1/I5)."""
    filters: dict = {
        "owner_scoped": owner_scoped,
        "intent": intent,
        "retrieval_mode": retrieval_mode,
    }
    if document_ids is not None:
        filters["document_ids"] = sorted(document_ids)
    return filters


def _actual_fusion_method(
    *,
    dense_ok: bool,
    lexical_ok: bool,
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    planned_dense: int,
    planned_lexical: int,
) -> str:
    """Describe what actually ran, not merely configuration."""
    if planned_dense <= 0 and planned_lexical > 0:
        return "lexical_only"
    if planned_lexical <= 0 and planned_dense > 0:
        return "dense_only"
    if dense and lexical and dense_ok and lexical_ok:
        return "rrf"
    if dense and dense_ok and (not lexical_ok or not lexical):
        return "dense_only"
    if lexical and lexical_ok and (not dense_ok or not dense):
        return "lexical_only"
    return "none"


class RetrievalService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.vector_store = build_vector_store(db, self.config)
        self.embeddings = EmbeddingService(self.config)
        self.ranker = build_reranker(
            enabled=bool(getattr(self.config, "rerank_enabled", False)),
            heuristic=bool(getattr(self.config, "rerank_heuristic_enabled", False)),
        )
        self.repo = RagRepository(db)
        self.query_expansion = QueryExpansionService()

    async def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        top_k: int | None = None,
        filters: dict | None = None,
        intent: RetrievalIntent | str | None = None,
        conversation_id: str | None = None,
        persist_trace: bool = False,
        evidence_revision_hash: str | None = None,
    ) -> RetrievalOutcome:
        if not self.config.enabled:
            return RetrievalOutcome(chunks=[], no_matches=True)

        document_ids, owner_scoped = _parse_scope(filters)
        retrieval_mode = str((filters or {}).get("retrieval_mode") or "hybrid").lower()
        plan = plan_retrieval(
            intent, self.config, top_k=top_k, retrieval_mode=retrieval_mode
        )
        exclude_parents = bool(getattr(self.config, "parent_context_enabled", False))

        # I1: explicit empty allow-list → zero evidence, never widen.
        if document_ids_is_empty_allow_list(document_ids):
            coverage = RetrievalCoverage(
                documents_in_scope=0,
                documents_with_retrieved_evidence=0,
                retrieved_passage_count=0,
                coverage_ratio=0.0,
            )
            outcome = RetrievalOutcome(
                chunks=[],
                no_matches=True,
                intent=plan.intent,
                fusion_method="none",
                coverage=coverage,
                scope_hash=scope_hash_for_document_ids(document_ids),
            )
            outcome.evidence_revision_hash = evidence_revision_hash or (
                build_retrieved_evidence_revision_hash(
                    project_id=project_id,
                    document_ids=document_ids,
                    chunks=outcome.chunks,
                    config=self.config,
                )
            )
            if persist_trace:
                outcome = await self._persist_trace(
                    outcome,
                    query=query,
                    user_id=user_id,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    document_ids=document_ids,
                    latency_ms=0,
                )
            return outcome

        cache_filters = _filters_for_cache(
            document_ids,
            owner_scoped=owner_scoped,
            intent=plan.intent.value,
            retrieval_mode=retrieval_mode,
        )
        cache_variant = (
            f"{getattr(self.config, 'retrieval_algorithm_version', 'hybrid-rrf-v1')}"
            f":{getattr(self.config, 'index_version', 'pgvector-fts-v1')}"
            f":rerank={self.ranker.name}:{self.ranker.version}"
            f":mode={retrieval_mode}"
            f":parent={int(exclude_parents)}"
            f":xq={QUERY_EXPANSION_VERSION if plan.multi_query else 'none'}"
        )
        started = perf_counter()
        query_variants_meta: list[dict] = []
        try:
            cached = await get_cached_retrieval(
                user_id=user_id,
                project_id=project_id,
                query=query,
                top_k=plan.top_k,
                filters=cache_filters,
                variant=cache_variant,
            )
            if cached is not None:
                filtered, removed = exclude_injection_flagged_chunks(cached)
                filtered = filter_to_allow_list(filtered, document_ids)
                if plan.diversify:
                    filtered, coverage = diversify_by_document(
                        filtered,
                        limit=plan.top_k,
                        max_per_document=plan.max_per_document,
                        documents_in_scope=len(document_ids) if document_ids is not None else 0,
                    )
                else:
                    coverage = RetrievalCoverage(
                        documents_in_scope=len(document_ids) if document_ids is not None else 0,
                        documents_with_retrieved_evidence=len({c.document_id for c in filtered}),
                        retrieved_passage_count=len(filtered),
                        coverage_ratio=0.0,
                    )
                    if coverage.documents_in_scope:
                        coverage.coverage_ratio = (
                            coverage.documents_with_retrieved_evidence
                            / coverage.documents_in_scope
                        )
                if removed:
                    metrics.rag_injection_chunks_filtered_total.inc(removed)
                metrics.rag_retrieved_chunks.observe(len(filtered))
                if exclude_parents and filtered:
                    filtered = await expand_parent_chunks(
                        filtered, repo=self.repo, document_ids=document_ids
                    )
                    filtered = filter_to_allow_list(filtered, document_ids)
                outcome = RetrievalOutcome(
                    chunks=filtered,
                    injection_chunks_filtered=removed,
                    no_matches=len(filtered) == 0,
                    intent=plan.intent,
                    fusion_method="cached",
                    coverage=coverage,
                    scope_hash=scope_hash_for_document_ids(document_ids),
                )
                outcome.evidence_revision_hash = evidence_revision_hash or (
                    build_retrieved_evidence_revision_hash(
                        project_id=project_id,
                        document_ids=document_ids,
                        chunks=outcome.chunks,
                        config=self.config,
                    )
                )
                if persist_trace:
                    outcome = await self._persist_trace(
                        outcome,
                        query=query,
                        user_id=user_id,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        document_ids=document_ids,
                        latency_ms=int((perf_counter() - started) * 1000),
                        query_variants=query_variants_meta,
                    )
                return outcome

            variants = self.query_expansion.expand_if_needed(
                query, multi_query=plan.multi_query
            )
            ranked_lists: list[list[RetrievedChunk]] = []
            total_dense = 0
            total_lexical = 0
            any_dense_ok = False
            any_lexical_ok = False
            dense_attempted = plan.dense_candidates > 0
            lexical_attempted = plan.lexical_candidates > 0
            branch_errors: list[str] = []
            lexical_queries_seen: set[str] = set()

            for variant in variants:
                query_variants_meta.append(
                    {
                        "label": variant.label,
                        "text": variant.text,
                        "dense_text": variant.dense_text or variant.text,
                        "lexical_text": variant.lexical_text or variant.text,
                        "language": variant.language,
                    }
                )
                dense, dense_ok, dense_err = await self._dense_branch(
                    variant.dense_text or variant.text,
                    user_id=user_id,
                    project_id=project_id,
                    document_ids=document_ids,
                    owner_scoped=owner_scoped,
                    top_k=plan.dense_candidates,
                    exclude_parents=exclude_parents,
                )
                lexical_query = variant.lexical_text or variant.text
                if lexical_query in lexical_queries_seen:
                    lexical, lexical_ok, lexical_err = [], True, None
                else:
                    lexical_queries_seen.add(lexical_query)
                    lexical, lexical_ok, lexical_err = await self._lexical_branch(
                        lexical_query,
                        user_id=user_id,
                        project_id=project_id,
                        document_ids=document_ids,
                        top_k=plan.lexical_candidates,
                        owner_scoped=owner_scoped,
                        exclude_parents=exclude_parents,
                    )
                if dense_err:
                    branch_errors.append(f"dense:{dense_err}")
                if lexical_err:
                    branch_errors.append(f"lexical:{lexical_err}")
                if dense_ok:
                    any_dense_ok = True
                if lexical_ok:
                    any_lexical_ok = True
                dense = filter_to_allow_list(dense, document_ids)
                lexical = filter_to_allow_list(lexical, document_ids)
                total_dense += len(dense)
                total_lexical += len(lexical)
                if dense and lexical:
                    ranked_lists.append(dense)
                    ranked_lists.append(lexical)
                elif dense:
                    ranked_lists.append(dense)
                elif lexical:
                    ranked_lists.append(lexical)

            if len(ranked_lists) > 1:
                fused = reciprocal_rank_fusion(
                    ranked_lists,
                    k=getattr(self.config, "rrf_k", 60),
                    limit=max(plan.top_k * 3, plan.top_k),
                )
            elif ranked_lists:
                fused = ranked_lists[0]
            else:
                fused = []
            fused = filter_to_allow_list(fused, document_ids)

            fusion_method = _actual_fusion_method(
                dense_ok=any_dense_ok,
                lexical_ok=any_lexical_ok,
                dense=[c for lst in ranked_lists for c in lst if "dense" in c.retrieval_sources]
                or ([] if not any_dense_ok else fused),
                lexical=[
                    c for lst in ranked_lists for c in lst if "lexical" in c.retrieval_sources
                ]
                or ([] if not any_lexical_ok else fused),
                planned_dense=plan.dense_candidates,
                planned_lexical=plan.lexical_candidates,
            )
            if plan.multi_query and len(variants) > 1 and fused:
                fusion_method = (
                    f"multi_query_rrf:{QUERY_EXPANSION_VERSION}"
                    if len(ranked_lists) > 1
                    else fusion_method
                )

            degraded = False
            degradation_reason: str | None = None
            if dense_attempted and lexical_attempted:
                if any_dense_ok and not any_lexical_ok and fused:
                    degraded = True
                    degradation_reason = "lexical_branch_failed"
                elif any_lexical_ok and not any_dense_ok and fused:
                    degraded = True
                    degradation_reason = "dense_branch_failed"
                elif not any_dense_ok and not any_lexical_ok:
                    degraded = True
                    degradation_reason = "both_branches_failed"
            elif dense_attempted and not any_dense_ok and not fused:
                degraded = True
                degradation_reason = "dense_branch_failed"
            elif lexical_attempted and not any_lexical_ok and not fused:
                degraded = True
                degradation_reason = "lexical_branch_failed"

            if degraded:
                metrics.rag_retrieval_degraded_total.inc()
                if "dense" in (degradation_reason or ""):
                    metrics.rag_vector_unavailable_total.inc()

            filtered, removed = exclude_injection_flagged_chunks(fused)
            if removed:
                metrics.rag_injection_chunks_filtered_total.inc(removed)

            if exclude_parents and filtered:
                filtered = await expand_parent_chunks(
                    filtered, repo=self.repo, document_ids=document_ids
                )
                filtered = filter_to_allow_list(filtered, document_ids)

            if filtered:
                rerank_started = perf_counter()
                filtered = self.ranker.rerank(query, filtered, limit=len(filtered))
                metrics.rag_rerank_latency_ms.observe((perf_counter() - rerank_started) * 1000)
                filtered = filter_to_allow_list(filtered, document_ids)

            if plan.diversify:
                filtered, coverage = diversify_by_document(
                    filtered,
                    limit=plan.top_k,
                    max_per_document=plan.max_per_document,
                    documents_in_scope=len(document_ids) if document_ids is not None else 0,
                )
            else:
                filtered = filtered[: plan.top_k]
                for idx, chunk in enumerate(filtered, start=1):
                    chunk.rank = idx
                coverage = RetrievalCoverage(
                    documents_in_scope=len(document_ids) if document_ids is not None else 0,
                    documents_with_retrieved_evidence=len({c.document_id for c in filtered}),
                    retrieved_passage_count=len(filtered),
                    coverage_ratio=0.0,
                )
                if coverage.documents_in_scope:
                    coverage.coverage_ratio = (
                        coverage.documents_with_retrieved_evidence / coverage.documents_in_scope
                    )

            outcome = RetrievalOutcome(
                chunks=filtered,
                injection_chunks_filtered=removed,
                no_matches=len(filtered) == 0,
                intent=plan.intent,
                fusion_method=fusion_method,
                coverage=coverage,
                dense_candidate_count=total_dense,
                lexical_candidate_count=total_lexical,
                fused_candidate_count=len(fused),
                scope_hash=scope_hash_for_document_ids(document_ids),
                degraded=degraded,
                degradation_reason=degradation_reason,
            )
            outcome.evidence_revision_hash = evidence_revision_hash or (
                build_retrieved_evidence_revision_hash(
                    project_id=project_id,
                    document_ids=document_ids,
                    chunks=outcome.chunks,
                    config=self.config,
                )
            )
            if outcome.no_matches and fused and removed == len(fused):
                outcome.degradation_reason = "injection_filtered_all_matches"
                outcome.degraded = True

            if removed == 0 and not degraded:
                await set_cached_retrieval(
                    user_id=user_id,
                    project_id=project_id,
                    query=query,
                    top_k=plan.top_k,
                    filters=cache_filters,
                    chunks=filtered,
                    variant=cache_variant,
                )
            metrics.rag_retrieved_chunks.observe(len(filtered))
            if outcome.no_matches:
                metrics.rag_retrieval_no_match_total.inc()
            if persist_trace:
                outcome = await self._persist_trace(
                    outcome,
                    query=query,
                    user_id=user_id,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    document_ids=document_ids,
                    latency_ms=int((perf_counter() - started) * 1000),
                    query_variants=query_variants_meta,
                    branch_errors=branch_errors,
                )
            return outcome
        except Exception:
            logger.exception("RAG retrieval failed for user=%s", user_id)
            metrics.rag_vector_unavailable_total.inc()
            metrics.rag_retrieval_degraded_total.inc()
            outcome = RetrievalOutcome(
                chunks=[],
                degraded=True,
                degradation_reason="retrieval_failed",
                intent=plan.intent,
                fusion_method="none",
                scope_hash=scope_hash_for_document_ids(document_ids),
            )
            outcome.evidence_revision_hash = evidence_revision_hash or (
                build_retrieved_evidence_revision_hash(
                    project_id=project_id,
                    document_ids=document_ids,
                    chunks=outcome.chunks,
                    config=self.config,
                )
            )
            if persist_trace:
                try:
                    outcome = await self._persist_trace(
                        outcome,
                        query=query,
                        user_id=user_id,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        document_ids=document_ids,
                        latency_ms=int((perf_counter() - started) * 1000),
                        query_variants=query_variants_meta,
                    )
                except Exception:
                    logger.exception("Failed to persist degraded retrieval trace")
            return outcome
        finally:
            metrics.rag_retrieval_latency_ms.observe((perf_counter() - started) * 1000)

    async def _dense_branch(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        owner_scoped: bool,
        top_k: int,
        exclude_parents: bool,
    ) -> tuple[list[RetrievedChunk], bool, str | None]:
        if top_k <= 0:
            return [], True, None
        try:
            query_embedding = (await self.embeddings.embed_texts([query]))[0]
            if not can_index_embedding(
                query_embedding,
                expected_dimensions=self.config.embedding_dimensions,
            ):
                return [], False, "embedding_dimension_mismatch"
            search_filters: dict = {
                "owner_scoped": owner_scoped,
                "exclude_parents": exclude_parents,
            }
            if document_ids is not None:
                search_filters["document_ids"] = document_ids
            dense = await self.vector_store.similarity_search(
                query,
                user_id=user_id,
                project_id=project_id,
                top_k=top_k,
                filters=search_filters,
                query_embedding=query_embedding,
            )
            return dense or [], True, None
        except Exception as exc:
            logger.exception("Dense retrieval branch failed")
            return [], False, type(exc).__name__

    async def _lexical_branch(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        owner_scoped: bool,
        top_k: int,
        exclude_parents: bool,
    ) -> tuple[list[RetrievedChunk], bool, str | None]:
        if top_k <= 0:
            return [], True, None
        try:
            lexical = await self.repo.lexical_search(
                user_id=user_id,
                project_id=project_id,
                document_ids=document_ids,
                query=query,
                top_k=top_k,
                owner_scoped=owner_scoped,
                exclude_parents=exclude_parents,
            )
            return lexical, True, None
        except Exception as exc:
            logger.exception("Lexical retrieval branch failed")
            return [], False, type(exc).__name__

    async def _persist_trace(
        self,
        outcome: RetrievalOutcome,
        *,
        query: str,
        user_id: str,
        project_id: str | None,
        conversation_id: str | None,
        document_ids: list[str] | None,
        latency_ms: int,
        query_variants: list[dict] | None = None,
        branch_errors: list[str] | None = None,
    ) -> RetrievalOutcome:
        chunk_revision_ids = list(
            dict.fromkeys(
                chunk.index_revision_id
                for chunk in outcome.chunks
                if chunk.index_revision_id
            )
        )
        outcome.index_revision_ids = (
            chunk_revision_ids
            or await self.repo.list_current_revision_ids(document_ids)
        )
        coverage = outcome.coverage
        trace = await self.repo.create_retrieval_trace(
            user_id=user_id,
            project_id=project_id,
            conversation_id=conversation_id,
            query=query,
            intent=outcome.intent.value if outcome.intent else None,
            scope_hash=outcome.scope_hash,
            evidence_revision_hash=outcome.evidence_revision_hash,
            index_revision_ids=outcome.index_revision_ids,
            document_ids=document_ids,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "citation_chunk_id": c.citation_chunk_id or c.chunk_id,
                    "document_id": c.document_id,
                    "score": c.score,
                    "rank": c.rank,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "citation_content": c.citation_content or c.content,
                    "snippet": (c.citation_content or c.content)[:400],
                    "char_start": (c.metadata or {}).get("char_start"),
                    "char_end": (c.metadata or {}).get("char_end"),
                    "source_unit_ids": (c.metadata or {}).get("source_unit_ids")
                    or (c.metadata or {}).get("source_span_ids"),
                    "offset_coordinate_system": (c.metadata or {}).get(
                        "offset_coordinate_system"
                    ),
                    "parent_context_id": c.parent_context_id,
                    "index_revision_id": c.index_revision_id,
                    "context_content": c.context_content,
                    "context_snippet": (
                        c.context_content[:400] if c.context_content is not None else None
                    ),
                    "retrieval_sources": list(c.retrieval_sources),
                }
                for c in outcome.chunks
            ],
            coverage={
                "documents_in_scope": coverage.documents_in_scope if coverage else 0,
                "documents_with_retrieved_evidence": (
                    coverage.documents_with_retrieved_evidence if coverage else 0
                ),
                "retrieved_passage_count": coverage.retrieved_passage_count if coverage else 0,
                "coverage_ratio": coverage.coverage_ratio if coverage else 0.0,
            },
            config={
                "fusion_method": outcome.fusion_method,
                "retrieval_algorithm_version": getattr(
                    self.config, "retrieval_algorithm_version", "hybrid-rrf-v1"
                ),
                "index_version": getattr(self.config, "index_version", "pgvector-fts-v1"),
                "dense_candidate_count": outcome.dense_candidate_count,
                "lexical_candidate_count": outcome.lexical_candidate_count,
                "fused_candidate_count": outcome.fused_candidate_count,
                "reranker": f"{self.ranker.name}:{self.ranker.version}",
                "query_variants": query_variants or [],
                "query_expansion_version": (
                    QUERY_EXPANSION_VERSION if query_variants and len(query_variants) > 1 else None
                ),
                "branch_errors": branch_errors or [],
                "original_query": query,
            },
            degraded=outcome.degraded,
            degradation_reason=outcome.degradation_reason,
            no_matches=outcome.no_matches,
            injection_chunks_filtered=outcome.injection_chunks_filtered,
            latency_ms=latency_ms,
        )
        outcome.retrieval_trace_id = trace.id
        return outcome
