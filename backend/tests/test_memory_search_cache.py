import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.storage import AVATAR_CACHE_CONTROL, PRIVATE_UPLOAD_CACHE_CONTROL, ObjectStorage
from backend.lib.memory_search_cache import (
    deserialize_memory_items,
    get_cached_memory_search,
    invalidate_memory_search_cache,
    memory_search_cache_key,
    serialize_memory_items,
    set_cached_memory_search,
)
from backend.modules.memory.domain.enums import MemoryLevel, MemoryType
from backend.modules.memory.domain.models import MemoryItem, MemoryMetadata


class StorageCacheControlTest(unittest.TestCase):
    def test_upload_bytes_applies_requested_cache_control(self):
        storage = ObjectStorage()
        storage._client = MagicMock()

        import asyncio

        async def run_inline(func):
            return func()

        with (
            patch("backend.core.storage.settings") as mock_settings,
            patch("backend.core.storage.asyncio.to_thread", side_effect=run_inline) as to_thread,
        ):
            mock_settings.STORAGE_BUCKET = "app-assets"
            mock_settings.STORAGE_PUBLIC_BASE_URL = "http://localhost:9000/app-assets"
            asyncio.run(
                storage.upload_bytes(
                    object_key="avatars/user/file.jpg",
                    body=b"abc",
                    content_type="image/jpeg",
                    cache_control=AVATAR_CACHE_CONTROL,
                )
            )

        to_thread.assert_awaited_once()
        storage._client.put_object.assert_called_once()
        self.assertEqual(
            storage._client.put_object.call_args.kwargs["CacheControl"],
            AVATAR_CACHE_CONTROL,
        )

    def test_private_upload_cache_control_constant(self):
        self.assertIn("private", PRIVATE_UPLOAD_CACHE_CONTROL)


class MemorySearchCacheTest(unittest.IsolatedAsyncioTestCase):
    def test_cache_key_changes_with_query(self):
        first = memory_search_cache_key(
            user_id="user-1",
            agent_id="agent-1",
            query="hello",
            run_id=None,
            project_id=None,
            memory_levels=["user"],
            limit=5,
        )
        second = memory_search_cache_key(
            user_id="user-1",
            agent_id="agent-1",
            query="goodbye",
            run_id=None,
            project_id=None,
            memory_levels=["user"],
            limit=5,
        )
        self.assertNotEqual(first, second)

    def test_serialize_and_deserialize_memory_items(self):
        item = MemoryItem(
            id="mem-1",
            content="prefers dark mode",
            metadata=MemoryMetadata(
                memory_level=MemoryLevel.USER,
                memory_type=MemoryType.PREFERENCE,
                user_id="user-1",
                agent_id="agent-1",
                confidence=0.9,
            ),
            score=0.88,
            created_at=datetime.now(UTC),
        )
        restored = deserialize_memory_items(serialize_memory_items([item]))
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored[0].id, "mem-1")
        self.assertEqual(restored[0].metadata.memory_level, MemoryLevel.USER)

    async def test_get_and_set_cached_memory_search(self):
        item = MemoryItem(
            id="mem-1",
            content="hello",
            metadata=MemoryMetadata(
                memory_level=MemoryLevel.USER,
                memory_type=MemoryType.FACT,
                user_id="user-1",
                agent_id="agent-1",
            ),
        )
        with (
            patch("backend.lib.memory_search_cache.cache_get_json", AsyncMock(return_value=None)),
            patch("backend.lib.memory_search_cache.cache_set_json", AsyncMock()) as cache_set,
        ):
            await set_cached_memory_search(
                user_id="user-1",
                agent_id="agent-1",
                query="hello",
                run_id=None,
                project_id=None,
                memory_levels=["user"],
                limit=5,
                items=[item],
            )
        cache_set.assert_awaited_once()

        with patch(
            "backend.lib.memory_search_cache.cache_get_json",
            AsyncMock(return_value=serialize_memory_items([item])),
        ):
            cached = await get_cached_memory_search(
                user_id="user-1",
                agent_id="agent-1",
                query="hello",
                run_id=None,
                project_id=None,
                memory_levels=["user"],
                limit=5,
            )
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(cached[0].content, "hello")

    async def test_invalidate_memory_search_cache_deletes_user_pattern(self):
        with patch(
            "backend.lib.memory_search_cache.cache_delete_pattern",
            AsyncMock(),
        ) as delete_pattern:
            await invalidate_memory_search_cache(user_id="user-1")
        delete_pattern.assert_awaited_once()
