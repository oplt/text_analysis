import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

from backend.lib.resource_cache import (
    calendar_items_cache_key,
    get_cached_model_list,
    invalidate_user_profile_cache,
    project_list_cache_key,
    set_cached_model_list,
    user_directory_cache_key,
    user_profile_cache_key,
)
from backend.modules.users.schemas import UserDirectoryResponse


class ResourceCacheKeyTest(unittest.TestCase):
    def test_cache_keys_are_namespaced(self):
        self.assertEqual(user_profile_cache_key("user-1"), "ga:user:profile:user-1")
        self.assertEqual(project_list_cache_key("user-1", 50, 0), "ga:projects:list:user-1:50:0")
        self.assertEqual(
            calendar_items_cache_key("user-1", date(2026, 1, 1), date(2026, 1, 31)),
            "ga:calendar:items:user-1:2026-01-01:2026-01-31",
        )
        self.assertEqual(user_directory_cache_key(50, 0), "ga:users:directory:50:0")


class ResourceCacheRoundTripTest(unittest.IsolatedAsyncioTestCase):
    async def test_set_and_get_cached_model_list(self):
        item = UserDirectoryResponse(id="user-1", email="a@example.com", full_name="A")
        with patch("backend.lib.resource_cache.cache_set_json", AsyncMock()) as cache_set:
            await set_cached_model_list(
                user_directory_cache_key(50, 0),
                [item],
                total=1,
                ttl_seconds=60,
            )
        cache_set.assert_awaited_once()

        with patch(
            "backend.lib.resource_cache.cache_get_json",
            AsyncMock(
                return_value={
                    "items": [item.model_dump(mode="json")],
                    "total": 1,
                }
            ),
        ):
            loaded = await get_cached_model_list(user_directory_cache_key(50, 0), UserDirectoryResponse)

        self.assertIsNotNone(loaded)
        items, total = loaded
        self.assertEqual(total, 1)
        self.assertEqual(items[0].email, "a@example.com")

    async def test_profile_cache_invalidation_deletes_key(self):
        with patch("backend.core.cache.cache_delete", AsyncMock()) as cache_delete:
            await invalidate_user_profile_cache("user-1")
        cache_delete.assert_awaited_once_with(user_profile_cache_key("user-1"))
