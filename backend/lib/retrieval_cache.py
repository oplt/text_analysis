from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from backend.core.cache import cache_delete_pattern, cache_get_json, cache_key, cache_set_json
from backend.core.config import settings
from backend.modules.rag.domain.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class CachedRetrieval:
    """A retrieval result together with the algorithm run that produced it."""

    chunks: list[RetrievedChunk]
    provenance: dict
    artifact_version: str = "v2"


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
    identity: dict | None = None,
) -> str:
    canonical_identity = json.dumps(
        identity or {"variant": variant}, sort_keys=True, separators=(",", ":"), default=str
    )
    return cache_key(
        "retrieval",
        user_id,
        _scope_token(project_id),
        _stable_digest(query, str(top_k), canonical_identity),
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
    identity: dict | None = None,
    expected_revision_ids: list[str] | None = None,
) -> CachedRetrieval | None:
    payload = await cache_get_json(
        retrieval_cache_key(
            user_id=user_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
            filters=filters,
            variant=variant,
            identity=identity,
        )
    )
    # Entries created before provenance was persisted are deliberately cache
    # misses: presenting them as reproducible would be misleading.
    if not isinstance(payload, dict):
        return None
    chunks = deserialize_retrieved_chunks(payload.get("chunks"))
    provenance = payload.get("provenance")
    artifact_version = payload.get("artifact_version")
    if chunks is None or not isinstance(provenance, dict) or not isinstance(artifact_version, str):
        return None
    if expected_revision_ids is not None:
        expected = set(expected_revision_ids)
        cached_revisions = set(provenance.get("index_revision_ids") or ())
        if cached_revisions != expected or any(
            chunk.index_revision_id not in expected for chunk in chunks
        ):
            return None
    return CachedRetrieval(
        chunks=chunks, provenance=provenance, artifact_version=artifact_version
    )


async def set_cached_retrieval(
    *,
    user_id: str,
    project_id: str | None,
    query: str,
    top_k: int,
    filters: dict | None,
    chunks: list[RetrievedChunk],
    provenance: dict,
    variant: str = "vector-v1",
    identity: dict | None = None,
    artifact_version: str = "v2",
) -> None:
    await cache_set_json(
        retrieval_cache_key(
            user_id=user_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
            filters=filters,
            variant=variant,
            identity=identity,
        ),
        {
            "artifact_version": artifact_version,
            "chunks": serialize_retrieved_chunks(chunks),
            "provenance": provenance,
        },
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
