from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from backend.core.cache import cache_get_json, cache_set_json, embedding_cache_key
from backend.core.config import settings


async def embed_texts_with_cache(
    *,
    provider: str,
    model: str,
    texts: list[str],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    dimensions: int | None = None,
    ttl_seconds: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    resolved_ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_EMBEDDING_TTL_SECONDS
    max_chars = settings.CACHE_EMBEDDING_MAX_TEXT_CHARS
    results: list[list[float] | None] = [None] * len(texts)
    pending: list[tuple[int, str]] = []

    for index, text in enumerate(texts):
        if len(text) > max_chars or not settings.CACHE_ENABLED:
            pending.append((index, text))
            continue

        cache_key = embedding_cache_key(
            provider,
            model,
            text,
            dimensions=dimensions,
        )
        cached = await cache_get_json(cache_key)
        if cached is not None:
            results[index] = cached
        else:
            pending.append((index, text))

    if pending:
        embedded = await embed_fn([text for _, text in pending])
        for (index, text), vector in zip(pending, embedded, strict=True):
            results[index] = vector
            if len(text) <= max_chars and settings.CACHE_ENABLED:
                cache_key = embedding_cache_key(
                    provider,
                    model,
                    text,
                    dimensions=dimensions,
                )
                await cache_set_json(cache_key, vector, ttl_seconds=resolved_ttl)

    return cast(list[list[float]], results)
