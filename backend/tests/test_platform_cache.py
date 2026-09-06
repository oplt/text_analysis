import unittest
from unittest.mock import AsyncMock, patch

from backend.core.cache import (
    PLATFORM_CONFIG_CACHE_KEY,
    PLATFORM_FEATURE_FLAGS_CACHE_KEY,
    _local_cache,
    cache_get_json,
    cache_set_json,
    invalidate_platform_caches,
)
from backend.modules.platform.router import router as platform_router
from backend.modules.platform.schemas import PlatformConfigResponse
from backend.modules.platform.service import PlatformService


class PlatformConfigCacheTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _local_cache.delete(PLATFORM_CONFIG_CACHE_KEY, PLATFORM_FEATURE_FLAGS_CACHE_KEY)

    async def test_get_platform_config_uses_cache_loader_once(self):
        db = AsyncMock()
        service = PlatformService(db)
        expected = PlatformConfigResponse(
            app_name="Demo",
            core_domain_singular="Project",
            core_domain_plural="Projects",
            module_pack="full_platform",
            enabled_modules=["ai"],
            module_catalog=[],
            available_module_packs=[],
            module_overrides={},
            mfa_enabled=False,
        )
        service._load_platform_config = AsyncMock(return_value=expected)

        with patch(
            "backend.modules.platform.config_service.cache_get_or_load_model",
            AsyncMock(side_effect=[expected, expected]),
        ) as cache_get:
            first = await service.get_platform_config()
            second = await service.get_platform_config()

        self.assertEqual(first.app_name, "Demo")
        self.assertEqual(second.app_name, "Demo")
        self.assertEqual(cache_get.await_count, 2)
        service._load_platform_config.assert_not_called()

    async def test_local_cache_serves_platform_config_without_second_redis_read(self):
        payload = {"app_name": "Demo", "enabled_modules": ["ai"]}
        redis_client = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)

        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", redis_client),
        ):
            mock_settings.CACHE_ENABLED = True
            mock_settings.CACHE_PLATFORM_TTL_SECONDS = 300

            await cache_set_json(PLATFORM_CONFIG_CACHE_KEY, payload, ttl_seconds=300)
            redis_client.get.reset_mock()

            first = await cache_get_json(PLATFORM_CONFIG_CACHE_KEY)
            second = await cache_get_json(PLATFORM_CONFIG_CACHE_KEY)

        self.assertEqual(first, payload)
        self.assertEqual(second, payload)
        redis_client.get.assert_not_awaited()

    def test_all_capability_routes_are_registered_once(self):
        expected = {
            ("GET", "/metadata"),
            ("GET", "/billing/plans"),
            ("GET", "/billing/subscription"),
            ("PUT", "/billing/subscription"),
            ("GET", "/api-keys"),
            ("POST", "/api-keys"),
            ("DELETE", "/api-keys/{api_key_id}"),
            ("GET", "/webhooks"),
            ("POST", "/webhooks"),
            ("PATCH", "/webhooks/{webhook_id}"),
            ("DELETE", "/webhooks/{webhook_id}"),
            ("POST", "/webhooks/{webhook_id}/test"),
            ("GET", "/feature-flags"),
            ("GET", "/admin/config"),
            ("PUT", "/admin/config"),
            ("GET", "/admin/plans"),
            ("POST", "/admin/plans"),
            ("PATCH", "/admin/plans/{plan_id}"),
            ("GET", "/admin/feature-flags"),
            ("POST", "/admin/feature-flags"),
            ("PATCH", "/admin/feature-flags/{feature_flag_id}"),
            ("GET", "/admin/email-templates"),
            ("POST", "/admin/email-templates"),
            ("PATCH", "/admin/email-templates/{template_id}"),
        }
        registered = [
            (method, route.path)
            for route in platform_router.routes
            for method in route.methods or set()
        ]

        self.assertEqual(set(registered), expected)
        self.assertEqual(len(registered), len(expected))

    async def test_list_feature_flags_uses_shared_cache_loader(self):
        db = AsyncMock()
        service = PlatformService(db)
        cached_payload = [
            {
                "id": "flag-1",
                "key": "ai_studio",
                "name": "AI Studio",
                "description": None,
                "module_key": "ai",
                "is_enabled": True,
                "rollout_percentage": 100,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
        service._load_feature_flags_for_cache = AsyncMock(return_value=cached_payload)

        with patch(
            "backend.modules.platform.feature_flag_service.cache_get_or_load_json",
            AsyncMock(side_effect=[cached_payload, cached_payload]),
        ) as cache_get:
            first = await service.list_feature_flags()
            second = await service.list_feature_flags()

        self.assertEqual(first[0].key, "ai_studio")
        self.assertEqual(second[0].key, "ai_studio")
        self.assertEqual(cache_get.await_count, 2)
        service._load_feature_flags_for_cache.assert_not_called()

    async def test_invalidate_platform_caches_clears_local_entries(self):
        with (
            patch("backend.core.cache.settings") as mock_settings,
            patch("backend.core.cache.redis_client", AsyncMock()) as redis_client,
        ):
            mock_settings.CACHE_ENABLED = True
            await cache_set_json(PLATFORM_FEATURE_FLAGS_CACHE_KEY, [{"key": "ai"}], ttl_seconds=300)
            await invalidate_platform_caches()

            value = await cache_get_json(PLATFORM_FEATURE_FLAGS_CACHE_KEY)

        self.assertIsNone(value)
        redis_client.delete.assert_awaited_once_with(
            PLATFORM_CONFIG_CACHE_KEY,
            PLATFORM_FEATURE_FLAGS_CACHE_KEY,
        )
