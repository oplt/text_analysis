from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.lib.retrieval_cache import (
    deserialize_retrieved_chunks,
    get_cached_retrieval,
    invalidate_retrieval_cache,
    invalidate_retrieval_cache_for_document,
    retrieval_cache_key,
    retrieval_cache_pattern,
    serialize_retrieved_chunks,
    set_cached_retrieval,
)
from backend.modules.rag.domain.models import RetrievedChunk


def _sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="hello world",
        score=0.91,
        filename="notes.txt",
        chunk_index=0,
    )


def test_retrieval_cache_key_changes_with_query_and_filters() -> None:
    base = retrieval_cache_key(
        user_id="user-1",
        project_id=None,
        query="hello",
        top_k=5,
        filters=None,
    )
    other_query = retrieval_cache_key(
        user_id="user-1",
        project_id=None,
        query="goodbye",
        top_k=5,
        filters=None,
    )
    filtered = retrieval_cache_key(
        user_id="user-1",
        project_id=None,
        query="hello",
        top_k=5,
        filters={"document_ids": ["doc-1"]},
    )

    assert base.startswith("ga:retrieval:user-1:_:")
    assert base != other_query
    assert base != filtered


def test_retrieval_cache_round_trip() -> None:
    chunk = _sample_chunk()
    payload = serialize_retrieved_chunks([chunk])
    restored = deserialize_retrieved_chunks(payload)

    assert restored is not None
    assert len(restored) == 1
    assert restored[0].chunk_id == chunk.chunk_id
    assert restored[0].score == chunk.score


def test_get_and_set_cached_retrieval() -> None:
    chunk = _sample_chunk()

    async def _run() -> None:
        with (
            patch("backend.lib.retrieval_cache.cache_get_json", AsyncMock(return_value=None)) as cache_get,
            patch("backend.lib.retrieval_cache.cache_set_json", AsyncMock()) as cache_set,
            patch("backend.lib.retrieval_cache.settings") as mock_settings,
        ):
            mock_settings.CACHE_RETRIEVAL_TTL_SECONDS = 180

            await set_cached_retrieval(
                user_id="user-1",
                project_id="proj-1",
                query="hello",
                top_k=5,
                filters=None,
                chunks=[chunk],
            )
            cache_set.assert_awaited_once()

            cache_get.return_value = serialize_retrieved_chunks([chunk])
            cached = await get_cached_retrieval(
                user_id="user-1",
                project_id="proj-1",
                query="hello",
                top_k=5,
                filters=None,
            )

        assert cached is not None
        assert cached[0].chunk_id == "chunk-1"

    asyncio.run(_run())


def test_invalidate_retrieval_cache_deletes_scope_pattern() -> None:
    async def _run() -> None:
        with patch("backend.lib.retrieval_cache.cache_delete_pattern", AsyncMock()) as delete_pattern:
            await invalidate_retrieval_cache(user_id="user-1", project_id="proj-1")

        delete_pattern.assert_awaited_once_with(
            retrieval_cache_pattern(user_id="user-1", project_id="proj-1")
        )

    asyncio.run(_run())


def test_invalidate_retrieval_cache_for_document_clears_project_and_global_scopes() -> None:
    async def _run() -> None:
        with patch("backend.lib.retrieval_cache.cache_delete_pattern", AsyncMock()) as delete_pattern:
            await invalidate_retrieval_cache_for_document(
                user_id="user-1",
                project_id="proj-1",
            )

        assert delete_pattern.await_count == 2
        delete_pattern.assert_any_await(
            retrieval_cache_pattern(user_id="user-1", project_id="proj-1")
        )
        delete_pattern.assert_any_await(
            retrieval_cache_pattern(user_id="user-1", project_id=None)
        )

    asyncio.run(_run())


def test_retrieval_service_uses_cached_results() -> None:
    from backend.modules.rag.application.retrieval_service import RetrievalService

    chunk = _sample_chunk()
    db = MagicMock()
    service = RetrievalService(db)
    service.config = SimpleNamespace(enabled=True, top_k=5)
    service.embeddings = MagicMock()
    service.embeddings.embed_texts = AsyncMock()
    service.vector_store = MagicMock()
    service.vector_store.similarity_search = AsyncMock()

    async def _run() -> None:
        with patch(
            "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
            AsyncMock(return_value=[chunk]),
        ) as cache_get:
            results = await service.retrieve(
                "hello",
                user_id="user-1",
                project_id=None,
                top_k=5,
            )

        assert results.chunks == [chunk]
        cache_get.assert_awaited_once()
        service.embeddings.embed_texts.assert_not_called()
        service.vector_store.similarity_search.assert_not_called()

    asyncio.run(_run())
