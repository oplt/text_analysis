from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import cast

from prometheus_client import Counter, Histogram

from backend.core.cache import (
    cache_mget_json,
    cache_mset_json,
    embedding_cache_key,
)
from backend.core.config import settings

logger = logging.getLogger(__name__)

embedding_cache_hits = Counter(
    "embedding_cache_hits",
    "Embedding cache hits",
)
embedding_cache_misses = Counter(
    "embedding_cache_misses",
    "Embedding cache misses",
)
embedding_cache_hit_ratio = Histogram(
    "embedding_cache_hit_ratio",
    "Per-call embedding cache hit ratio (hits / (hits + misses))",
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
provider_batch_size = Histogram(
    "provider_batch_size",
    "Texts sent per embedding provider batch",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
embedding_provider_latency = Histogram(
    "embedding_provider_latency",
    "Embedding provider batch latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0),
)
redis_embedding_cache_latency = Histogram(
    "redis_embedding_cache_latency",
    "Redis embedding cache MGET/MSET latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

_FLIGHT_GUARD = asyncio.Lock()
_INFLIGHT: OrderedDict[str, asyncio.Future[list[float]]] = OrderedDict()


def _flight_max() -> int:
    try:
        return max(16, int(settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX))
    except Exception:
        return 1024


def _batch_size() -> int:
    try:
        return max(1, int(settings.RAG_EMBEDDING_BATCH_SIZE))
    except Exception:
        return 32


def _batch_concurrency() -> int:
    try:
        return max(1, int(settings.RAG_EMBEDDING_BATCH_CONCURRENCY))
    except Exception:
        return 2


def reset_embedding_flight_locks_for_tests() -> None:
    """Test helper: clear process-local single-flight futures."""
    _INFLIGHT.clear()


def _trim_inflight() -> None:
    limit = _flight_max()
    while len(_INFLIGHT) > limit:
        key, future = _INFLIGHT.popitem(last=False)
        if not future.done():
            # Keep in-progress futures; reinsert and stop if everything is live.
            _INFLIGHT[key] = future
            if all(not item.done() for item in _INFLIGHT.values()):
                break


async def _embed_batches(
    texts: list[str],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
) -> list[list[float]]:
    if not texts:
        return []

    size = _batch_size()
    batches = [texts[index : index + size] for index in range(0, len(texts), size)]
    if len(batches) == 1:
        provider_batch_size.observe(len(batches[0]))
        started = perf_counter()
        vectors = await embed_fn(batches[0])
        embedding_provider_latency.observe(perf_counter() - started)
        return vectors

    semaphore = asyncio.Semaphore(_batch_concurrency())
    results: list[list[list[float]] | None] = [None] * len(batches)

    async def _run(batch_index: int, batch: list[str]) -> None:
        async with semaphore:
            provider_batch_size.observe(len(batch))
            started = perf_counter()
            results[batch_index] = await embed_fn(batch)
            embedding_provider_latency.observe(perf_counter() - started)

    await asyncio.gather(*(_run(index, batch) for index, batch in enumerate(batches)))
    flattened: list[list[float]] = []
    for batch_vectors in results:
        if batch_vectors is None:
            raise RuntimeError("embedding provider batch returned no vectors")
        flattened.extend(batch_vectors)
    return flattened


async def embed_texts_with_cache(
    *,
    provider: str,
    model: str,
    texts: list[str],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    dimensions: int | None = None,
    ttl_seconds: int | None = None,
) -> list[list[float]]:
    """Embed texts with batched Redis cache lookup/write and bounded provider calls.

    Flow: compute keys → MGET → identify misses → dedupe → embed in batches →
    pipeline MSET → reconstruct original order. Identical missing cache keys use
    single-flight futures so concurrent callers share one provider embed.
    """
    if not texts:
        return []

    resolved_ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_EMBEDDING_TTL_SECONDS
    max_chars = settings.CACHE_EMBEDDING_MAX_TEXT_CHARS
    results: list[list[float] | None] = [None] * len(texts)

    cacheable: list[tuple[int, str, str]] = []
    bypass: list[tuple[int, str]] = []

    for index, text in enumerate(texts):
        if len(text) > max_chars or not settings.CACHE_ENABLED:
            bypass.append((index, text))
            continue
        cache_key = embedding_cache_key(
            provider,
            model,
            text,
            dimensions=dimensions,
        )
        cacheable.append((index, text, cache_key))

    hits = 0
    misses = 0
    pending_cacheable: list[tuple[int, str, str]] = []

    if cacheable:
        started = perf_counter()
        cached_values = await cache_mget_json([key for _, _, key in cacheable])
        redis_embedding_cache_latency.observe(perf_counter() - started)
        for (index, text, cache_key), cached in zip(cacheable, cached_values, strict=True):
            if isinstance(cached, list) and cached:
                results[index] = cast(list[float], cached)
                hits += 1
            else:
                pending_cacheable.append((index, text, cache_key))
                misses += 1

    if hits or misses:
        embedding_cache_hits.inc(hits)
        embedding_cache_misses.inc(misses)
        total = hits + misses
        if total:
            embedding_cache_hit_ratio.observe(hits / total)

    # Map original indices that still need vectors → text (+ optional cache key).
    need: list[tuple[int, str, str | None]] = list(pending_cacheable)
    need.extend((index, text, None) for index, text in bypass)
    if not need:
        return cast(list[list[float]], results)

    # Unique texts; for cacheable keys, register single-flight ownership.
    unique_order: list[str] = []
    unique_meta: dict[str, str | None] = {}  # text -> cache_key (or None)
    wait_futures: list[tuple[list[int], asyncio.Future[list[float]]]] = []
    owned_futures: list[tuple[str, asyncio.Future[list[float]]]] = []
    text_to_indices: dict[str, list[int]] = {}

    for index, text, _cache_key in need:
        text_to_indices.setdefault(text, []).append(index)

    for text, indices in text_to_indices.items():
        # Prefer any cache key from the first matching need entry.
        cache_key = next(key for idx, candidate, key in need if candidate == text)
        if cache_key is None:
            unique_order.append(text)
            unique_meta[text] = None
            continue

        async with _FLIGHT_GUARD:
            existing = _INFLIGHT.get(cache_key)
            if existing is not None and not existing.done():
                wait_futures.append((indices, existing))
                _INFLIGHT.move_to_end(cache_key)
                continue
            future: asyncio.Future[list[float]] = asyncio.get_running_loop().create_future()
            _INFLIGHT[cache_key] = future
            _trim_inflight()
            owned_futures.append((cache_key, future))
            unique_order.append(text)
            unique_meta[text] = cache_key

    if unique_order:
        try:
            embedded_unique = await _embed_batches(unique_order, embed_fn)
        except Exception as exc:
            for cache_key, future in owned_futures:
                if not future.done():
                    future.set_exception(exc)
                async with _FLIGHT_GUARD:
                    current = _INFLIGHT.get(cache_key)
                    if current is future:
                        _INFLIGHT.pop(cache_key, None)
            raise

        if len(embedded_unique) != len(unique_order):
            error = RuntimeError(
                f"embedding provider returned {len(embedded_unique)} vectors "
                f"for {len(unique_order)} texts"
            )
            for cache_key, future in owned_futures:
                if not future.done():
                    future.set_exception(error)
                async with _FLIGHT_GUARD:
                    current = _INFLIGHT.get(cache_key)
                    if current is future:
                        _INFLIGHT.pop(cache_key, None)
            raise error

        cache_writes: list[tuple[str, list[float]]] = []
        for text, vector in zip(unique_order, embedded_unique, strict=True):
            for index in text_to_indices[text]:
                results[index] = vector
            cache_key = unique_meta[text]
            if cache_key is not None:
                cache_writes.append((cache_key, vector))
                for owned_key, future in owned_futures:
                    if owned_key == cache_key and not future.done():
                        future.set_result(vector)

        if cache_writes:
            started = perf_counter()
            await cache_mset_json(cache_writes, ttl_seconds=resolved_ttl)
            redis_embedding_cache_latency.observe(perf_counter() - started)

        async with _FLIGHT_GUARD:
            for cache_key, future in owned_futures:
                current = _INFLIGHT.get(cache_key)
                if current is future:
                    _INFLIGHT.pop(cache_key, None)

    for indices, future in wait_futures:
        vector = await future
        for index in indices:
            results[index] = vector

    return cast(list[list[float]], results)
