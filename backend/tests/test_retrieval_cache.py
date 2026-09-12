from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.lib.retrieval_cache import (
    CachedRetrieval,
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
            patch(
                "backend.lib.retrieval_cache.cache_get_json", AsyncMock(return_value=None)
            ) as cache_get,
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
                provenance={"fusion_method": "rrf", "result_chunk_ids": ["chunk-1"]},
            )
            cache_set.assert_awaited_once()

            cache_get.return_value = {
                "artifact_version": "v2",
                "chunks": serialize_retrieved_chunks([chunk]),
                "provenance": {"fusion_method": "rrf", "result_chunk_ids": ["chunk-1"]},
            }
            cached = await get_cached_retrieval(
                user_id="user-1",
                project_id="proj-1",
                query="hello",
                top_k=5,
                filters=None,
            )

        assert cached is not None
        assert cached.chunks[0].chunk_id == "chunk-1"
        assert cached.provenance["fusion_method"] == "rrf"

    asyncio.run(_run())


def test_legacy_payload_and_wrong_revision_are_safe_cache_misses() -> None:
    chunk = _sample_chunk()

    async def _run() -> None:
        with patch(
            "backend.lib.retrieval_cache.cache_get_json",
            AsyncMock(
                return_value={
                    "chunks": serialize_retrieved_chunks([chunk]),
                    "provenance": {"index_revision_ids": ["old"]},
                }
            ),
        ):
            assert await get_cached_retrieval(
                user_id="user-1", project_id=None, query="q", top_k=5, filters=None
            ) is None

        with patch(
            "backend.lib.retrieval_cache.cache_get_json",
            AsyncMock(
                return_value={
                    "artifact_version": "v2",
                    "chunks": serialize_retrieved_chunks([chunk]),
                    "provenance": {"index_revision_ids": ["old"]},
                }
            ),
        ):
            assert await get_cached_retrieval(
                user_id="user-1",
                project_id=None,
                query="q",
                top_k=5,
                filters=None,
                expected_revision_ids=["new"],
            ) is None

    asyncio.run(_run())


def test_cache_identity_changes_for_plan_and_evidence_revision() -> None:
    base = retrieval_cache_key(
        user_id="user-1", project_id=None, query="q", top_k=5, filters=None,
        identity={"planner": {"dense_candidates": 40}, "evidence_revision_hash": "one"},
    )
    changed = retrieval_cache_key(
        user_id="user-1", project_id=None, query="q", top_k=5, filters=None,
        identity={"planner": {"dense_candidates": 20}, "evidence_revision_hash": "two"},
    )
    assert base != changed


def test_invalidate_retrieval_cache_deletes_scope_pattern() -> None:
    async def _run() -> None:
        with patch(
            "backend.lib.retrieval_cache.cache_delete_pattern", AsyncMock()
        ) as delete_pattern:
            await invalidate_retrieval_cache(user_id="user-1", project_id="proj-1")

        delete_pattern.assert_awaited_once_with(
            retrieval_cache_pattern(user_id="user-1", project_id="proj-1")
        )

    asyncio.run(_run())


def test_invalidate_retrieval_cache_for_document_clears_project_and_global_scopes() -> None:
    async def _run() -> None:
        with patch(
            "backend.lib.retrieval_cache.cache_delete_pattern", AsyncMock()
        ) as delete_pattern:
            await invalidate_retrieval_cache_for_document(
                user_id="user-1",
                project_id="proj-1",
            )

        assert delete_pattern.await_count == 2
        delete_pattern.assert_any_await(
            retrieval_cache_pattern(user_id="user-1", project_id="proj-1")
        )
        delete_pattern.assert_any_await(retrieval_cache_pattern(user_id="user-1", project_id=None))

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
    service.repo = MagicMock()
    service.repo.create_retrieval_trace = AsyncMock(return_value=SimpleNamespace(id="trace-1"))
    service.repo.list_current_revision_ids = AsyncMock(return_value=["revision-current"])

    async def _run() -> None:
        with patch(
            "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
            AsyncMock(
                return_value=CachedRetrieval(
                    chunks=[chunk],
                    provenance={
                        "intent": "fact",
                        "fusion_method": "rrf",
                        "dense_candidate_count": 4,
                        "lexical_candidate_count": 3,
                        "fused_candidate_count": 5,
                        "result_chunk_ids": ["chunk-1"],
                        "index_revision_ids": ["revision-cached"],
                        "query_variants": [{"label": "original", "text": "hello"}],
                    },
                )
            ),
        ) as cache_get:
            results = await service.retrieve(
                "hello",
                user_id="user-1",
                project_id=None,
                top_k=5,
                persist_trace=True,
            )

        assert [result.chunk_id for result in results.chunks] == [chunk.chunk_id]
        assert results.cache_hit is True
        assert results.fusion_method == "rrf"
        assert results.retrieval_provenance["result_chunk_ids"] == ["chunk-1"]
        assert results.index_revision_ids == ["revision-cached"]
        trace_config = service.repo.create_retrieval_trace.await_args.kwargs["config"]
        assert trace_config["cache_hit"] is True
        assert trace_config["original_fusion_method"] == "rrf"
        assert trace_config["query_variants"] == [{"label": "original", "text": "hello"}]
        cache_get.assert_awaited_once()
        service.embeddings.embed_texts.assert_not_called()
        service.vector_store.similarity_search.assert_not_called()

    asyncio.run(_run())
