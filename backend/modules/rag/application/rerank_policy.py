"""Bounded reranking that preserves useful retrieval results on ranker failure."""

from __future__ import annotations

from dataclasses import dataclass

from backend.modules.rag.application.reranker_port import RerankerPort
from backend.modules.rag.domain.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RerankResult:
    chunks: list[RetrievedChunk]
    failed: bool = False


def apply_rerank_policy(
    ranker: RerankerPort,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    depth: int,
) -> RerankResult:
    """Rerank only a bounded leading window; retain the remainder in original order."""
    if depth <= 0 or not chunks:
        return RerankResult(chunks=chunks)
    bounded_depth = min(depth, len(chunks))
    try:
        reranked = ranker.rerank(query, chunks[:bounded_depth], limit=bounded_depth)
    except Exception:
        return RerankResult(chunks=chunks, failed=True)
    return RerankResult(chunks=[*reranked, *chunks[bounded_depth:]])
