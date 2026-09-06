from __future__ import annotations

import logging
from time import perf_counter

from backend.core.config import settings
from backend.lib.retrieval_cache import get_cached_retrieval, set_cached_retrieval
from backend.lib.vectors import can_index_embedding
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.retrieval_filters import exclude_injection_flagged_chunks
from backend.modules.rag.application.retrieval_ranker import HybridRetrievalRanker
from backend.modules.rag.domain.models import RetrievalOutcome
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.vector_store_adapter import build_vector_store
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.vector_store = build_vector_store(db, self.config)
        self.embeddings = EmbeddingService(self.config)
        self.ranker = HybridRetrievalRanker()

    async def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> RetrievalOutcome:
        if not self.config.enabled:
            return RetrievalOutcome(chunks=[])

        resolved_top_k = top_k or self.config.top_k
        rerank_enabled = getattr(self.config, "rerank_enabled", False)
        cache_variant = "hybrid-v1" if rerank_enabled else "vector-v1"
        started = perf_counter()
        try:
            cached = await get_cached_retrieval(
                user_id=user_id,
                project_id=project_id,
                query=query,
                top_k=resolved_top_k,
                filters=filters,
                variant=cache_variant,
            )
            if cached is not None:
                filtered, removed = exclude_injection_flagged_chunks(cached)
                if removed:
                    metrics.rag_injection_chunks_filtered_total.inc(removed)
                metrics.rag_retrieved_chunks.observe(len(filtered))
                logger.debug(
                    "RAG retrieval cache_hit user=%s chunks=%s injection_filtered=%s",
                    user_id,
                    len(filtered),
                    removed,
                )
                return RetrievalOutcome(
                    chunks=filtered,
                    injection_chunks_filtered=removed,
                    no_matches=len(filtered) == 0,
                )

            query_embedding = (await self.embeddings.embed_texts([query]))[0]
            if not can_index_embedding(
                query_embedding,
                expected_dimensions=self.config.embedding_dimensions,
            ):
                logger.warning(
                    "Query embedding dimension mismatch for user=%s (got %s, expected %s)",
                    user_id,
                    len(query_embedding),
                    self.config.embedding_dimensions,
                )
                metrics.rag_vector_unavailable_total.inc()
                metrics.rag_retrieval_degraded_total.inc()
                return RetrievalOutcome(
                    chunks=[],
                    degraded=True,
                    degradation_reason="embedding_dimension_mismatch",
                )

            multiplier = getattr(self.config, "rerank_candidate_multiplier", 1)
            candidate_limit = (
                min(50, resolved_top_k * multiplier) if rerank_enabled else resolved_top_k
            )
            raw_results = await self.vector_store.similarity_search(
                query,
                user_id=user_id,
                project_id=project_id,
                top_k=candidate_limit,
                filters=filters,
                query_embedding=query_embedding,
            )
            filtered, removed = exclude_injection_flagged_chunks(raw_results)
            if removed:
                metrics.rag_injection_chunks_filtered_total.inc(removed)
            if rerank_enabled:
                rerank_started = perf_counter()
                filtered = self.ranker.rerank(query, filtered, limit=resolved_top_k)
                metrics.rag_rerank_latency_ms.observe((perf_counter() - rerank_started) * 1000)

            outcome = RetrievalOutcome(
                chunks=filtered,
                injection_chunks_filtered=removed,
                no_matches=len(filtered) == 0,
            )
            if outcome.no_matches and raw_results and removed == len(raw_results):
                outcome.degradation_reason = "injection_filtered_all_matches"

            if removed == 0:
                await set_cached_retrieval(
                    user_id=user_id,
                    project_id=project_id,
                    query=query,
                    top_k=resolved_top_k,
                    filters=filters,
                    chunks=filtered,
                    variant=cache_variant,
                )
            metrics.rag_retrieved_chunks.observe(len(filtered))
            if outcome.no_matches:
                metrics.rag_retrieval_no_match_total.inc()
            duration_ms = (perf_counter() - started) * 1000
            if outcome.degraded:
                logger.warning(
                    "RAG retrieval degraded user=%s duration_ms=%.2f reason=%s chunks=%s",
                    user_id,
                    duration_ms,
                    outcome.degradation_reason,
                    len(filtered),
                )
            elif outcome.no_matches:
                logger.info(
                    "RAG retrieval no_matches user=%s duration_ms=%.2f chunks=%s",
                    user_id,
                    duration_ms,
                    len(filtered),
                )
            else:
                logger.info(
                    "RAG retrieval completed user=%s duration_ms=%.2f "
                    "chunks=%s injection_filtered=%s",
                    user_id,
                    duration_ms,
                    len(filtered),
                    removed,
                )
            if duration_ms >= settings.SLOW_EXTERNAL_CALL_MS:
                logger.warning(
                    "slow_rag_retrieval user=%s duration_ms=%.2f",
                    user_id,
                    duration_ms,
                )
            return outcome
        except Exception:
            logger.exception("RAG retrieval failed for user=%s", user_id)
            metrics.rag_vector_unavailable_total.inc()
            metrics.rag_retrieval_degraded_total.inc()
            return RetrievalOutcome(
                chunks=[],
                degraded=True,
                degradation_reason="retrieval_failed",
            )
        finally:
            metrics.rag_retrieval_latency_ms.observe((perf_counter() - started) * 1000)
