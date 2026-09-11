from __future__ import annotations

import hashlib
import json
import logging
from time import perf_counter

from backend.core.config import settings
from backend.lib.retrieval_cache import get_cached_retrieval, set_cached_retrieval
from backend.lib.vectors import can_index_embedding
from backend.modules.rag.application.document_scope import document_ids_is_empty_allow_list
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.parent_context_service import expand_parent_chunks
from backend.modules.rag.application.retrieval_filters import exclude_injection_flagged_chunks
from backend.modules.rag.application.retrieval_fusion import reciprocal_rank_fusion
from backend.modules.rag.application.retrieval_planner import plan_retrieval
from backend.modules.rag.application.retrieval_ranker import HybridRetrievalRanker
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


class RetrievalService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.vector_store = build_vector_store(db, self.config)
        self.embeddings = EmbeddingService(self.config)
        self.ranker = HybridRetrievalRanker()
        self.repo = RagRepository(db)

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
    ) -> RetrievalOutcome:
        if not self.config.enabled:
            return RetrievalOutcome(chunks=[], no_matches=True)

        document_ids, owner_scoped = _parse_scope(filters)
        retrieval_mode = str((filters or {}).get("retrieval_mode") or "hybrid").lower()
        plan = plan_retrieval(
            intent, self.config, top_k=top_k, retrieval_mode=retrieval_mode
        )

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
                fusion_method=getattr(self.config, "fusion_method", "rrf"),
                coverage=coverage,
                scope_hash=scope_hash_for_document_ids(document_ids),
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
            f":{'rerank' if getattr(self.config, 'rerank_enabled', False) else 'norerank'}"
            f":mode={retrieval_mode}"
            f":parent={int(bool(getattr(self.config, 'parent_context_enabled', False)))}"
        )
        started = perf_counter()
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
                outcome = RetrievalOutcome(
                    chunks=filtered,
                    injection_chunks_filtered=removed,
                    no_matches=len(filtered) == 0,
                    intent=plan.intent,
                    fusion_method=getattr(self.config, "fusion_method", "rrf"),
                    coverage=coverage,
                    scope_hash=scope_hash_for_document_ids(document_ids),
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
                    )
                return outcome

            dense: list[RetrievedChunk] = []
            if plan.dense_candidates > 0:
                query_embedding = (await self.embeddings.embed_texts([query]))[0]
                if not can_index_embedding(
                    query_embedding,
                    expected_dimensions=self.config.embedding_dimensions,
                ):
                    metrics.rag_vector_unavailable_total.inc()
                    metrics.rag_retrieval_degraded_total.inc()
                    outcome = RetrievalOutcome(
                        chunks=[],
                        degraded=True,
                        degradation_reason="embedding_dimension_mismatch",
                        intent=plan.intent,
                        fusion_method=getattr(self.config, "fusion_method", "rrf"),
                        scope_hash=scope_hash_for_document_ids(document_ids),
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
                        )
                    return outcome

                search_filters = {
                    "owner_scoped": owner_scoped,
                }
                if document_ids is not None:
                    search_filters["document_ids"] = document_ids

                dense = await self.vector_store.similarity_search(
                    query,
                    user_id=user_id,
                    project_id=project_id,
                    top_k=plan.dense_candidates,
                    filters=search_filters,
                    query_embedding=query_embedding,
                )

            lexical: list[RetrievedChunk] = []
            if plan.lexical_candidates > 0:
                lexical = await self.repo.lexical_search(
                    user_id=user_id,
                    project_id=project_id,
                    document_ids=document_ids,
                    query=query,
                    top_k=plan.lexical_candidates,
                    owner_scoped=owner_scoped,
                )

            dense = filter_to_allow_list(dense, document_ids)
            lexical = filter_to_allow_list(lexical, document_ids)

            if dense and lexical:
                fused = reciprocal_rank_fusion(
                    [dense, lexical],
                    k=getattr(self.config, "rrf_k", 60),
                    limit=max(plan.top_k * 3, plan.top_k),
                )
            elif dense:
                fused = dense
            else:
                fused = lexical
            fused = filter_to_allow_list(fused, document_ids)

            filtered, removed = exclude_injection_flagged_chunks(fused)
            if removed:
                metrics.rag_injection_chunks_filtered_total.inc(removed)

            if getattr(self.config, "parent_context_enabled", False) and filtered:
                filtered = await expand_parent_chunks(
                    filtered, repo=self.repo, document_ids=document_ids
                )
                filtered = filter_to_allow_list(filtered, document_ids)

            if getattr(self.config, "rerank_enabled", False) and filtered:
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
                fusion_method=getattr(self.config, "fusion_method", "rrf"),
                coverage=coverage,
                dense_candidate_count=len(dense),
                lexical_candidate_count=len(lexical),
                fused_candidate_count=len(fused),
                scope_hash=scope_hash_for_document_ids(document_ids),
            )
            if outcome.no_matches and fused and removed == len(fused):
                outcome.degradation_reason = "injection_filtered_all_matches"

            if removed == 0:
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
                fusion_method=getattr(self.config, "fusion_method", "rrf"),
                scope_hash=scope_hash_for_document_ids(document_ids),
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
                    )
                except Exception:
                    logger.exception("Failed to persist degraded retrieval trace")
            return outcome
        finally:
            metrics.rag_retrieval_latency_ms.observe((perf_counter() - started) * 1000)

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
    ) -> RetrievalOutcome:
        coverage = outcome.coverage
        trace = await self.repo.create_retrieval_trace(
            user_id=user_id,
            project_id=project_id,
            conversation_id=conversation_id,
            query=query,
            intent=outcome.intent.value if outcome.intent else None,
            scope_hash=outcome.scope_hash,
            document_ids=document_ids,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "score": c.score,
                    "rank": c.rank,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "snippet": c.content[:400],
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
                "retrieval_algorithm_version": getattr(self.config, "retrieval_algorithm_version", "hybrid-rrf-v1"),
                "index_version": getattr(self.config, "index_version", "pgvector-fts-v1"),
                "dense_candidate_count": outcome.dense_candidate_count,
                "lexical_candidate_count": outcome.lexical_candidate_count,
                "fused_candidate_count": outcome.fused_candidate_count,
            },
            degraded=outcome.degraded,
            degradation_reason=outcome.degradation_reason,
            no_matches=outcome.no_matches,
            injection_chunks_filtered=outcome.injection_chunks_filtered,
            latency_ms=latency_ms,
        )
        outcome.retrieval_trace_id = trace.id
        return outcome
