"""Reranker port — keep post-RRF scoring pluggable and scale-safe."""

from __future__ import annotations

import re
from typing import Protocol

from backend.modules.rag.domain.models import RetrievedChunk

# Unicode-aware tokens (letters/numbers across scripts), min length 2.
_UNICODE_TOKEN = re.compile(r"[^\W_]{2,}", re.UNICODE)


class RerankerPort(Protocol):
    name: str
    version: str

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]: ...


class NoOpReranker:
    """Safe default: preserve RRF/dense order without mixing incomparable scales."""

    name = "noop"
    version = "v1"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        return chunks[:limit]


class UnicodeHeuristicReranker:
    """Optional lexical nudge using Unicode tokens; does not blend raw RRF scores."""

    name = "unicode_heuristic"
    version = "v1"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        terms = set(_UNICODE_TOKEN.findall(query.lower()))
        if not terms or len(chunks) < 2:
            return chunks[:limit]

        def key(chunk: RetrievedChunk) -> tuple[float, float, int]:
            content_terms = set(_UNICODE_TOKEN.findall(chunk.content.lower()))
            overlap = len(terms & content_terms) / len(terms)
            # Sort by overlap first, then original score, then original rank.
            return (overlap, chunk.score, -(chunk.rank or 0))

        return sorted(chunks, key=key, reverse=True)[:limit]


def build_reranker(*, enabled: bool, heuristic: bool = False) -> RerankerPort:
    if not enabled:
        return NoOpReranker()
    if heuristic:
        return UnicodeHeuristicReranker()
    return NoOpReranker()
