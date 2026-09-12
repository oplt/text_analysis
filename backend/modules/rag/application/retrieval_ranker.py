"""Backward-compatible alias; prefer ``reranker_port.build_reranker``."""

from __future__ import annotations

from backend.modules.rag.application.reranker_port import (
    NoOpReranker,
    UnicodeHeuristicReranker,
    build_reranker,
)
from backend.modules.rag.domain.models import RetrievedChunk


class HybridRetrievalRanker:
    """Legacy name kept for eval baselines; Unicode-aware, no RRF score mixing."""

    def __init__(self) -> None:
        self._impl = UnicodeHeuristicReranker()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        return self._impl.rerank(query, chunks, limit=limit)


__all__ = [
    "HybridRetrievalRanker",
    "NoOpReranker",
    "UnicodeHeuristicReranker",
    "build_reranker",
]
