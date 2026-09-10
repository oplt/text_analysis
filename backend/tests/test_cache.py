import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.cache import (
    PLATFORM_CONFIG_CACHE_KEY,
    PLATFORM_FEATURE_FLAGS_CACHE_KEY,
    cache_get_json,
    cache_mget_json,
    cache_mset_json,
    cache_set_json,
    embedding_cache_key,
    invalidate_platform_caches,
)
from backend.lib.embedding_cache import reset_embedding_flight_locks_for_tests


class CacheHelpersTest(unittest.IsolatedAsyncioTestCase):
    async def test_cache_get_returns_none_when_disabled(self):
        with patch("backend.core.cache.settings") as mock_settings:
            mock_settings.CACHE_ENABLED = False
            value = await cache_get_json("ga:test")
        self.assertIsNone(value)

    async def test_cache_set_skips_when_disabled(self):
        client = AsyncMock()
        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", client),
        ):
            mock_settings.CACHE_ENABLED = False
            await cache_set_json("ga:test", {"ok": True}, ttl_seconds=60)
        client.setex.assert_not_called()

    async def test_cache_round_trip_json(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", client),
        ):
            mock_settings.CACHE_ENABLED = True
            await cache_set_json("ga:test", {"count": 2}, ttl_seconds=30)
            client.setex.assert_awaited_once()

            client.get = AsyncMock(return_value='{"count": 2}')
            value = await cache_get_json("ga:test")
        self.assertEqual(value, {"count": 2})

    async def test_cache_mget_and_mset_use_redis_batch_apis(self):
        client = AsyncMock()
        pipe = MagicMock()
        pipe.setex = MagicMock()
        pipe.execute = AsyncMock(return_value=[True, True])
        client.pipeline = MagicMock(return_value=pipe)
        client.mget = AsyncMock(return_value=['[0.1]', None])

        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", client),
        ):
            mock_settings.CACHE_ENABLED = True
            values = await cache_mget_json(["ga:a", "ga:b"])
            await cache_mset_json(
                [("ga:a", [0.1]), ("ga:b", [0.2])],
                ttl_seconds=30,
            )

        self.assertEqual(values, [[0.1], None])
        client.mget.assert_awaited_once_with(["ga:a", "ga:b"])
        self.assertEqual(pipe.setex.call_count, 2)
        pipe.execute.assert_awaited_once()

    def test_embedding_cache_key_is_stable_and_namespaced(self):
        first = embedding_cache_key(
            "openai", "text-embedding-3-small", "hello world", dimensions=1536
        )
        second = embedding_cache_key(
            "openai", "text-embedding-3-small", "hello world", dimensions=1536
        )
        third = embedding_cache_key("openai", "text-embedding-3-small", "other", dimensions=1536)
        different_dims = embedding_cache_key(
            "openai", "text-embedding-3-small", "hello world", dimensions=768
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertNotEqual(first, different_dims)
        self.assertTrue(first.startswith("ga:embed:"))

    async def test_invalidate_platform_caches_deletes_known_keys(self):
        client = AsyncMock()
        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", client),
        ):
            mock_settings.CACHE_ENABLED = True
            await invalidate_platform_caches()

        client.delete.assert_awaited_once_with(
            PLATFORM_CONFIG_CACHE_KEY,
            PLATFORM_FEATURE_FLAGS_CACHE_KEY,
        )

    async def test_cache_delete_pattern_unlinks_in_batches(self):
        from backend.core.cache import cache_delete_pattern

        class FakeScanIter:
            def __init__(self, keys):
                self._keys = iter(keys)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._keys)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        client = AsyncMock()
        client.scan_iter = lambda **kwargs: FakeScanIter(
            [f"ga:projects:list:user-1:{index}" for index in range(5)]
        )
        client.unlink = AsyncMock()
        client.delete = AsyncMock()

        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", client),
        ):
            mock_settings.CACHE_ENABLED = True
            await cache_delete_pattern("ga:projects:list:user-1:*", batch_size=2)

        self.assertEqual(client.unlink.await_count, 3)
        client.delete.assert_not_called()


class EmbeddingCacheTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_embedding_flight_locks_for_tests()

    async def test_embed_texts_uses_cache_for_repeat_query(self):
        from backend.modules.rag.application.embedding_service import EmbeddingService

        service = EmbeddingService()
        service._adapter = AsyncMock()
        service._adapter.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])

        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch(
                "backend.lib.embedding_cache.cache_mget_json",
                AsyncMock(side_effect=[[None], [[0.1, 0.2]]]),
            ) as cache_mget,
            patch(
                "backend.lib.embedding_cache.cache_mset_json",
                AsyncMock(),
            ) as cache_mset,
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 4000
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 32
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2

            first = await service.embed_texts(["hello"])
            second = await service.embed_texts(["hello"])

        self.assertEqual(first, [[0.1, 0.2]])
        self.assertEqual(second, [[0.1, 0.2]])
        service._adapter.embed_texts.assert_awaited_once_with(["hello"])
        cache_mset.assert_awaited_once()
        self.assertEqual(cache_mget.await_count, 2)

    async def test_embed_texts_batches_mget_and_dedupes_provider_calls(self):
        from backend.lib.embedding_cache import embed_texts_with_cache

        embed_fn = AsyncMock(return_value=[[0.1]])
        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch(
                "backend.lib.embedding_cache.cache_mget_json",
                AsyncMock(return_value=[None, [0.9], None]),
            ) as cache_mget,
            patch(
                "backend.lib.embedding_cache.cache_mset_json",
                AsyncMock(),
            ) as cache_mset,
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 4000
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 32
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2

            vectors = await embed_texts_with_cache(
                provider="local",
                model="local-heuristic",
                texts=["a", "cached", "a"],
                embed_fn=embed_fn,
            )

        self.assertEqual(vectors, [[0.1], [0.9], [0.1]])
        # One unique miss ("a") after dedupe; provider sees a single-text batch.
        embed_fn.assert_awaited_once_with(["a"])
        cache_mget.assert_awaited()
        cache_mset.assert_awaited_once()
        written = cache_mset.await_args.args[0]
        self.assertEqual(len(written), 1)

    async def test_embed_texts_respects_provider_batch_size(self):
        from backend.lib.embedding_cache import embed_texts_with_cache

        calls: list[list[str]] = []

        async def embed_fn(batch: list[str]) -> list[list[float]]:
            calls.append(list(batch))
            return [[float(index)] for index, _ in enumerate(batch)]

        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch(
                "backend.lib.embedding_cache.cache_mget_json",
                AsyncMock(return_value=[None, None, None, None, None]),
            ),
            patch("backend.lib.embedding_cache.cache_mset_json", AsyncMock()),
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 4000
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 2
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2

            vectors = await embed_texts_with_cache(
                provider="local",
                model="local-heuristic",
                texts=["t0", "t1", "t2", "t3", "t4"],
                embed_fn=embed_fn,
            )

        self.assertEqual(len(vectors), 5)
        self.assertEqual([len(batch) for batch in calls], [2, 2, 1])

    async def test_embed_texts_skips_cache_for_overlong_text(self):
        from backend.lib.embedding_cache import embed_texts_with_cache

        embed_fn = AsyncMock(return_value=[[0.3, 0.4]])
        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch("backend.lib.embedding_cache.cache_mget_json", AsyncMock()) as cache_mget,
            patch("backend.lib.embedding_cache.cache_mset_json", AsyncMock()) as cache_mset,
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 5
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 32
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2
            vectors = await embed_texts_with_cache(
                provider="local",
                model="local-heuristic",
                texts=["0123456789"],
                embed_fn=embed_fn,
            )

        self.assertEqual(vectors, [[0.3, 0.4]])
        embed_fn.assert_awaited_once_with(["0123456789"])
        cache_mget.assert_not_awaited()
        cache_mset.assert_not_awaited()

    async def test_ai_embed_retrieval_queries_uses_cache(self):
        from backend.modules.ai.providers import AiProviderRegistry

        registry = AiProviderRegistry()
        provider = AsyncMock()
        provider.key = "local"
        provider.embed_texts = AsyncMock(return_value=[[0.5, 0.6]])
        registry.embedding_provider_and_model = MagicMock(
            return_value=(provider, "local-heuristic")
        )

        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch(
                "backend.lib.embedding_cache.cache_mget_json",
                AsyncMock(side_effect=[[None], [[0.5, 0.6]]]),
            ),
            patch("backend.lib.embedding_cache.cache_mset_json", AsyncMock()) as cache_mset,
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 4000
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.CACHE_QUERY_EMBEDDING_TTL_SECONDS = 3600
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 32
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2
            mock_settings.RAG_EMBEDDING_DIMENSIONS = 1536

            first = await registry.embed_retrieval_queries(["hello"])
            second = await registry.embed_retrieval_queries(["hello"])

        self.assertEqual(first, [[0.5, 0.6]])
        self.assertEqual(second, [[0.5, 0.6]])
        provider.embed_texts.assert_awaited_once_with(["hello"], model="local-heuristic")
        cache_mset.assert_awaited_once()

    async def test_embed_texts_single_flight_shares_provider_call(self):
        import asyncio

        from backend.lib.embedding_cache import embed_texts_with_cache

        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def embed_fn(batch: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            started.set()
            await release.wait()
            return [[0.7, 0.8] for _ in batch]

        with (
            patch("backend.lib.embedding_cache.settings") as mock_settings,
            patch(
                "backend.lib.embedding_cache.cache_mget_json",
                AsyncMock(return_value=[None]),
            ),
            patch("backend.lib.embedding_cache.cache_mset_json", AsyncMock()),
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_EMBEDDING_TTL_SECONDS = 600
            mock_settings.CACHE_EMBEDDING_MAX_TEXT_CHARS = 4000
            mock_settings.CACHE_EMBEDDING_FLIGHT_LOCKS_MAX = 1024
            mock_settings.RAG_EMBEDDING_BATCH_SIZE = 32
            mock_settings.RAG_EMBEDDING_BATCH_CONCURRENCY = 2

            first = asyncio.create_task(
                embed_texts_with_cache(
                    provider="local",
                    model="local-heuristic",
                    texts=["shared"],
                    embed_fn=embed_fn,
                )
            )
            await started.wait()
            second = asyncio.create_task(
                embed_texts_with_cache(
                    provider="local",
                    model="local-heuristic",
                    texts=["shared"],
                    embed_fn=embed_fn,
                )
            )
            await asyncio.sleep(0)
            release.set()
            vectors_a, vectors_b = await asyncio.gather(first, second)

        self.assertEqual(call_count, 1)
        self.assertEqual(vectors_a, [[0.7, 0.8]])
        self.assertEqual(vectors_b, [[0.7, 0.8]])
