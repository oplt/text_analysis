from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.lib.vectors import can_index_embedding, cosine_similarity, vector_literal

logger = logging.getLogger(__name__)


class DenseFallbackScopeTooLarge(RuntimeError):
    """JSON vectors cannot provide exact retrieval for this eligible scope."""


def json_fallback_max_candidates(top_k: int) -> int:
    return min(5000, max(250, top_k * 50))


def rank_embedding_matches(
    query_embedding: list[float],
    rows: list[dict],
    *,
    top_k: int,
    score_threshold: float,
    build_match,
) -> list:
    scored: list[tuple[float, dict]] = []
    for row in rows:
        embedding = row.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_embedding, embedding)
        if score >= score_threshold:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [build_match(row, score) for score, row in scored[:top_k]]


def parse_embedding_json(raw: str | list[float] | None) -> list[float]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def embedding_is_indexable(
    embedding: list[float],
    *,
    expected_dimensions: int | None = None,
) -> bool:
    return can_index_embedding(
        embedding,
        expected_dimensions=expected_dimensions or settings.RAG_EMBEDDING_DIMENSIONS,
    )


async def pgvector_is_available(db: AsyncSession) -> bool:
    try:
        result = await db.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
        )
        return result.scalar() is not None
    except Exception:
        logger.debug("pgvector availability check failed", exc_info=True)
        return False


async def store_chunk_embeddings_batch(
    db: AsyncSession,
    *,
    table: str,
    items: list[tuple[str, list[float]]],
) -> None:
    if not items:
        return
    if not await pgvector_is_available(db):
        return

    indexable = [
        (chunk_id, embedding) for chunk_id, embedding in items if embedding_is_indexable(embedding)
    ]
    if not indexable:
        return

    for chunk_id, embedding in indexable:
        await db.execute(
            text(f"UPDATE {table} SET embedding = CAST(:embedding AS vector) WHERE id = :chunk_id"),
            {
                "embedding": vector_literal(embedding),
                "chunk_id": chunk_id,
            },
        )
