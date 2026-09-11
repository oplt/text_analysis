from __future__ import annotations

import hashlib
import json

from backend.core.cache import cache_delete_pattern, cache_get_json, cache_key, cache_set_json
from backend.core.config import settings
from backend.modules.rag.domain.models import RetrievedChunk


def _scope_token(value: str | None) -> str:
    return value or "_"


def _stable_digest(*parts: str) -> str:
    payload = "\0".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _filters_digest(filters: dict | None) -> str:
    if not filters:
        return "_"
    normalized = json.dumps(filters, sort_keys=True, separators=(",", ":"), default=str)
    return _stable_digest(normalized)


def retrieval_cache_key(
    *,
    user_id: str,
    project_id: str | None,
    query: str,
    top_k: int,
    filters: dict | None,
    variant: str = "vector-v1",
) -> str:
    return cache_key(
        "retrieval",
        user_id,
        _scope_token(project_id),
        _stable_digest(query, str(top_k), variant),
        _filters_digest(filters),
    )


def retrieval_cache_pattern(*, user_id: str, project_id: str | None) -> str:
    return cache_key("retrieval", user_id, _scope_token(project_id), "*")


def serialize_retrieved_chunks(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "score": chunk.score,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "metadata": chunk.metadata,
            "index_revision_id": chunk.index_revision_id,
        }
        for chunk in chunks
    ]


def deserialize_retrieved_chunks(payload: list[dict] | None) -> list[RetrievedChunk] | None:
    if payload is None:
        return None
    return [
        RetrievedChunk(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            content=item["content"],
            score=float(item["score"]),
            filename=item["filename"],
            chunk_index=int(item["chunk_index"]),
            page_number=item.get("page_number"),
            metadata=item.get("metadata") or {},
            index_revision_id=item.get("index_revision_id"),
        )
        for item in payload
    ]


async def get_cached_retrieval(
    *,
    user_id: str,
    project_id: str | None,
    query: str,
    top_k: int,
    filters: dict | None,
    variant: str = "vector-v1",
) -> list[RetrievedChunk] | None:
    payload = await cache_get_json(
        retrieval_cache_key(
            user_id=user_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
            filters=filters,
            variant=variant,
        )
    )
    return deserialize_retrieved_chunks(payload)


async def set_cached_retrieval(
    *,
    user_id: str,
    project_id: str | None,
    query: str,
    top_k: int,
    filters: dict | None,
    chunks: list[RetrievedChunk],
    variant: str = "vector-v1",
) -> None:
    await cache_set_json(
        retrieval_cache_key(
            user_id=user_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
            filters=filters,
            variant=variant,
        ),
        serialize_retrieved_chunks(chunks),
        ttl_seconds=settings.CACHE_RETRIEVAL_TTL_SECONDS,
    )


async def invalidate_retrieval_cache(*, user_id: str, project_id: str | None) -> None:
    await cache_delete_pattern(retrieval_cache_pattern(user_id=user_id, project_id=project_id))


async def invalidate_retrieval_cache_for_document(
    *,
    user_id: str,
    project_id: str | None,
) -> None:
    await invalidate_retrieval_cache(user_id=user_id, project_id=project_id)
    await invalidate_retrieval_cache(user_id=user_id, project_id=None)
