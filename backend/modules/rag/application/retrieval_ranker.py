from __future__ import annotations

import re

from backend.modules.rag.domain.models import RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


class HybridRetrievalRanker:
    """Dependency-free lexical/vector fusion over a bounded vector candidate set."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        terms = set(_TOKEN_PATTERN.findall(query.lower()))
        if not terms or len(chunks) < 2:
            return chunks[:limit]

        def score(chunk: RetrievedChunk) -> tuple[float, float]:
            content_terms = set(_TOKEN_PATTERN.findall(chunk.content.lower()))
            lexical = len(terms & content_terms) / len(terms)
            return (chunk.score * 0.8) + (lexical * 0.2), chunk.score

        return sorted(chunks, key=score, reverse=True)[:limit]
