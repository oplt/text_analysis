from __future__ import annotations

from datetime import date

from pydantic import BaseModel

import backend.core.cache as core_cache
from backend.core.cache import cache_get_json, cache_key, cache_set_json


def user_profile_cache_key(user_id: str) -> str:
    return cache_key("user", "profile", user_id)


def project_list_cache_key(user_id: str, limit: int, offset: int) -> str:
    return cache_key("projects", "list", user_id, str(limit), str(offset))


def calendar_items_cache_key(user_id: str, start_date: date, end_date: date) -> str:
    return cache_key(
        "calendar",
        "items",
        user_id,
        start_date.isoformat(),
        end_date.isoformat(),
    )


def user_directory_cache_key(limit: int, offset: int) -> str:
    return cache_key("users", "directory", str(limit), str(offset))


async def get_cached_model_list(
    key: str,
    model: type[BaseModel],
) -> tuple[list[BaseModel], int] | None:
    payload = await cache_get_json(key)
    if payload is None:
        return None
    items = [model.model_validate(item) for item in payload.get("items", [])]
    return items, int(payload.get("total", 0))


async def set_cached_model_list(
    key: str,
    items: list[BaseModel],
    *,
    total: int,
    ttl_seconds: int,
) -> None:
    await cache_set_json(
        key,
        {
            "items": [item.model_dump(mode="json") for item in items],
            "total": total,
        },
        ttl_seconds=ttl_seconds,
    )


async def get_cached_model_items(
    key: str,
    model: type[BaseModel],
) -> list[BaseModel] | None:
    payload = await cache_get_json(key)
    if payload is None:
        return None
    return [model.model_validate(item) for item in payload.get("items", [])]


async def set_cached_model_items(
    key: str,
    items: list[BaseModel],
    *,
    ttl_seconds: int,
) -> None:
    await cache_set_json(
        key,
        {"items": [item.model_dump(mode="json") for item in items]},
        ttl_seconds=ttl_seconds,
    )


async def invalidate_user_profile_cache(user_id: str) -> None:
    await core_cache.cache_delete(user_profile_cache_key(user_id))


async def invalidate_project_list_cache(user_id: str) -> None:
    await core_cache.cache_delete_pattern(cache_key("projects", "list", user_id, "*"))


async def invalidate_calendar_cache(user_id: str) -> None:
    await core_cache.cache_delete_pattern(cache_key("calendar", "items", user_id, "*"))


async def invalidate_user_directory_cache() -> None:
    await core_cache.cache_delete_pattern(cache_key("users", "directory", "*"))
