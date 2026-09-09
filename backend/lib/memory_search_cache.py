from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from backend.core.cache import cache_delete_pattern, cache_get_json, cache_key, cache_set_json
from backend.core.config import settings
from backend.modules.memory.domain.models import MemoryItem, MemoryMetadata


def _scope_token(value: str | None) -> str:
    return value or "_"


def _stable_digest(*parts: str) -> str:
    payload = "\0".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def memory_search_cache_key(
    *,
    user_id: str,
    agent_id: str,
    query: str,
    run_id: str | None,
    project_id: str | None,
    memory_levels: list[str],
    limit: int,
) -> str:
    return cache_key(
        "memory",
        "search",
        user_id,
        agent_id,
        _scope_token(run_id),
        _scope_token(project_id),
        _stable_digest(query, ",".join(sorted(memory_levels)), str(limit)),
    )


def memory_search_cache_pattern(*, user_id: str) -> str:
    return cache_key("memory", "search", user_id, "*")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def serialize_memory_items(items: list[MemoryItem]) -> list[dict]:
    return [
        {
            "id": item.id,
            "content": item.content,
            "score": item.score,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            "metadata": item.metadata.to_dict(),
        }
        for item in items
    ]


def deserialize_memory_items(payload: list[dict] | None) -> list[MemoryItem] | None:
    if payload is None:
        return None

    items: list[MemoryItem] = []
    for raw in payload:
        metadata = MemoryMetadata.from_dict(raw.get("metadata") or {})
        meta_payload = raw.get("metadata") or {}
        metadata.created_at = _parse_datetime(meta_payload.get("created_at"))
        metadata.last_seen_at = _parse_datetime(meta_payload.get("last_seen_at"))
        metadata.last_confirmed_at = _parse_datetime(meta_payload.get("last_confirmed_at"))
        metadata.occurred_at = _parse_datetime(meta_payload.get("occurred_at"))
        metadata.expires_at = _parse_datetime(meta_payload.get("expires_at"))
        items.append(
            MemoryItem(
                id=str(raw["id"]),
                content=str(raw["content"]),
                metadata=metadata,
                score=raw.get("score"),
                created_at=_parse_datetime(raw.get("created_at")),
                updated_at=_parse_datetime(raw.get("updated_at")),
            )
        )
    return items


async def get_cached_memory_search(
    *,
    user_id: str,
    agent_id: str,
    query: str,
    run_id: str | None,
    project_id: str | None,
    memory_levels: list[str],
    limit: int,
) -> list[MemoryItem] | None:
    payload = await cache_get_json(
        memory_search_cache_key(
            user_id=user_id,
            agent_id=agent_id,
            query=query,
            run_id=run_id,
            project_id=project_id,
            memory_levels=memory_levels,
            limit=limit,
        )
    )
    return deserialize_memory_items(payload)


async def set_cached_memory_search(
    *,
    user_id: str,
    agent_id: str,
    query: str,
    run_id: str | None,
    project_id: str | None,
    memory_levels: list[str],
    limit: int,
    items: list[MemoryItem],
) -> None:
    await cache_set_json(
        memory_search_cache_key(
            user_id=user_id,
            agent_id=agent_id,
            query=query,
            run_id=run_id,
            project_id=project_id,
            memory_levels=memory_levels,
            limit=limit,
        ),
        serialize_memory_items(items),
        ttl_seconds=settings.CACHE_MEMORY_SEARCH_TTL_SECONDS,
    )


async def invalidate_memory_search_cache(*, user_id: str) -> None:
    await cache_delete_pattern(memory_search_cache_pattern(user_id=user_id))
