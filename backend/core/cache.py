from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any, TypeVar

import redis.asyncio as redis
from pydantic import BaseModel

from backend.core.config import settings

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

CACHE_NAMESPACE = "ga"
PLATFORM_CONFIG_CACHE_KEY = f"{CACHE_NAMESPACE}:platform:config"
PLATFORM_FEATURE_FLAGS_CACHE_KEY = f"{CACHE_NAMESPACE}:platform:feature_flags"
SETTINGS_DATABASE_CACHE_KEY = f"{CACHE_NAMESPACE}:settings:database:all"
SETTINGS_CONFIG_ENTRIES_CACHE_KEY = f"{CACHE_NAMESPACE}:settings:config_entries"
PLATFORM_EMAIL_TEMPLATES_CACHE_KEY = f"{CACHE_NAMESPACE}:platform:email_templates"
OBSERVABILITY_STATUS_CACHE_KEY = f"{CACHE_NAMESPACE}:observability:status"
LOCAL_CACHE_KEYS = frozenset(
    {
        PLATFORM_CONFIG_CACHE_KEY,
        PLATFORM_FEATURE_FLAGS_CACHE_KEY,
        SETTINGS_DATABASE_CACHE_KEY,
        SETTINGS_CONFIG_ENTRIES_CACHE_KEY,
        PLATFORM_EMAIL_TEMPLATES_CACHE_KEY,
        OBSERVABILITY_STATUS_CACHE_KEY,
    }
)
LOCAL_ONLY_CACHE_KEYS = frozenset(
    {
        SETTINGS_CONFIG_ENTRIES_CACHE_KEY,
        OBSERVABILITY_STATUS_CACHE_KEY,
    }
)

T = TypeVar("T")


class _LocalTTLCache:
    __slots__ = ("_entries", "_maxsize")

    def __init__(self, maxsize: int = 64):
        self._maxsize = maxsize
        self._entries: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if monotonic() >= expires_at:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        self._entries[key] = (value, monotonic() + ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._entries.pop(key, None)


_local_cache = _LocalTTLCache()


def _uses_local_cache(key: str) -> bool:
    return key in LOCAL_CACHE_KEYS


def _uses_redis_cache(key: str) -> bool:
    if key in LOCAL_ONLY_CACHE_KEYS:
        return False
    return settings.CACHE_ENABLED


def _local_ttl_for_key(key: str) -> int:
    if key == OBSERVABILITY_STATUS_CACHE_KEY:
        return settings.CACHE_OBSERVABILITY_STATUS_TTL_SECONDS
    if key in {
        SETTINGS_DATABASE_CACHE_KEY,
        SETTINGS_CONFIG_ENTRIES_CACHE_KEY,
        PLATFORM_EMAIL_TEMPLATES_CACHE_KEY,
    }:
        return settings.CACHE_SETTINGS_TTL_SECONDS
    if key.startswith(f"{CACHE_NAMESPACE}:platform:"):
        return settings.CACHE_PLATFORM_TTL_SECONDS
    return 60


def get_local_cached_json(key: str) -> Any | None:
    if not _uses_local_cache(key):
        return None
    return _local_cache.get(key)


def set_local_cached_json(key: str, value: Any, *, ttl_seconds: int) -> None:
    if _uses_local_cache(key):
        _local_cache.set(key, value, ttl_seconds=ttl_seconds)


def delete_local_cached_json(*keys: str) -> None:
    if keys:
        _local_cache.delete(*keys)


def cache_key(*parts: str) -> str:
    return ":".join((CACHE_NAMESPACE, *parts))


def embedding_cache_key(
    provider: str,
    model: str,
    text: str,
    *,
    dimensions: int | None = None,
) -> str:
    resolved_dimensions = dimensions if dimensions is not None else settings.RAG_EMBEDDING_DIMENSIONS
    digest = hashlib.sha256(
        f"{provider}\0{model}\0{resolved_dimensions}\0{text}".encode()
    ).hexdigest()
    return cache_key("embed", digest)


async def cache_get_json(key: str) -> Any | None:
    if _uses_local_cache(key):
        local_value = _local_cache.get(key)
        if local_value is not None:
            return local_value

    if not _uses_redis_cache(key):
        return None
    try:
        raw = await redis_client.get(key)
        if raw is None:
            return None
        value = json.loads(raw)
        if _uses_local_cache(key):
            _local_cache.set(key, value, ttl_seconds=_local_ttl_for_key(key))
        return value
    except Exception:
        logger.debug("cache get failed for key=%s", key, exc_info=True)
        return None


async def cache_set_json(key: str, value: Any, *, ttl_seconds: int) -> None:
    if _uses_local_cache(key):
        _local_cache.set(key, value, ttl_seconds=ttl_seconds)
    if not _uses_redis_cache(key):
        return
    try:
        await redis_client.setex(
            key,
            ttl_seconds,
            json.dumps(value, ensure_ascii=True, default=str),
        )
    except Exception:
        logger.debug("cache set failed for key=%s", key, exc_info=True)


async def cache_delete(*keys: str) -> None:
    if keys:
        _local_cache.delete(*keys)
    redis_keys = [key for key in keys if _uses_redis_cache(key)]
    if not redis_keys:
        return
    try:
        await redis_client.delete(*redis_keys)
    except Exception:
        logger.debug("cache delete failed for keys=%s", redis_keys, exc_info=True)


async def cache_delete_pattern(pattern: str, *, batch_size: int = 100) -> None:
    if not settings.CACHE_ENABLED:
        return
    try:
        batch: list[str] = []
        async for key in redis_client.scan_iter(match=pattern, count=batch_size):
            batch.append(key)
            if len(batch) >= batch_size:
                await redis_client.delete(*batch)
                batch.clear()
        if batch:
            await redis_client.delete(*batch)
    except Exception:
        logger.debug("cache delete pattern failed for pattern=%s", pattern, exc_info=True)


async def cache_get_model(key: str, model: type[T]) -> T | None:  # noqa: UP047
    payload = await cache_get_json(key)
    if payload is None:
        return None
    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_validate(payload)
    return payload


async def cache_set_model(key: str, value: BaseModel, *, ttl_seconds: int) -> None:
    await cache_set_json(key, value.model_dump(mode="json"), ttl_seconds=ttl_seconds)


async def cache_get_or_load_model(
    key: str,
    model: type[T],
    *,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[T]],
) -> T:
    cached = await cache_get_model(key, model)
    if cached is not None:
        return cached
    loaded = await loader()
    if isinstance(loaded, BaseModel):
        await cache_set_model(key, loaded, ttl_seconds=ttl_seconds)
    return loaded


async def cache_get_or_load_json(
    key: str,
    *,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[Any]],
) -> Any:
    cached = await cache_get_json(key)
    if cached is not None:
        return cached
    loaded = await loader()
    await cache_set_json(key, loaded, ttl_seconds=ttl_seconds)
    return loaded


async def invalidate_platform_caches() -> None:
    await cache_delete(PLATFORM_CONFIG_CACHE_KEY, PLATFORM_FEATURE_FLAGS_CACHE_KEY)


async def invalidate_platform_email_template_cache() -> None:
    await cache_delete(PLATFORM_EMAIL_TEMPLATES_CACHE_KEY)


async def invalidate_settings_database_cache() -> None:
    await cache_delete(SETTINGS_DATABASE_CACHE_KEY, SETTINGS_CONFIG_ENTRIES_CACHE_KEY)


def invalidate_settings_config_cache() -> None:
    delete_local_cached_json(SETTINGS_CONFIG_ENTRIES_CACHE_KEY)


async def invalidate_settings_related_caches(setting_key: str | None = None) -> None:
    await invalidate_settings_database_cache()
    if setting_key is None or setting_key.startswith("platform."):
        await invalidate_platform_caches()
