import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.observability.schemas import ObservabilityStatus, ObservabilityStatusItem
from backend.observability.service import ObservabilityService, build_public_url


def make_settings(**overrides):
    defaults = {
        "GRAFANA_PUBLIC_URL": "http://localhost:3001/",
        "PROMETHEUS_PUBLIC_URL": "http://localhost:9090/",
        "TEMPO_PUBLIC_URL": "http://localhost:3200",
        "GRAFANA_APP_OVERVIEW_DASHBOARD_PATH": "/d/app-overview/application-overview",
        "GRAFANA_API_DASHBOARD_PATH": "/d/api-observability/backend-api",
        "GRAFANA_FRONTEND_DASHBOARD_PATH": "",
        "GRAFANA_DATABASE_DASHBOARD_PATH": "/d/database/database",
        "GRAFANA_CACHE_DASHBOARD_PATH": "/d/cache/cache-redis",
        "GRAFANA_WORKERS_DASHBOARD_PATH": "/d/workers/background-workers",
        "GRAFANA_SCHEDULED_TASKS_DASHBOARD_PATH": "/d/scheduled-tasks/scheduled-tasks",
        "GRAFANA_ERRORS_DASHBOARD_PATH": "/d/errors/error-investigation",
        "GRAFANA_TEMPO_EXPLORE_PATH": "/explore",
        "CACHE_OBSERVABILITY_STATUS_TTL_SECONDS": 30,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_status_item(**overrides):
    payload = {"status": "healthy", "detail": "ok", "last_checked_at": "2026-01-01T00:00:00+00:00"}
    payload.update(overrides)
    return ObservabilityStatusItem(**payload)


def make_observability_status():
    item = make_status_item()
    return ObservabilityStatus(
        api=item,
        frontend=item,
        database=item,
        cache=item,
        workers=item,
        background_jobs=item,
        error_rate=item,
        request_latency=item,
        prometheus=item,
        grafana=item,
        tempo=item,
    )


class ObservabilityServiceTest(unittest.TestCase):
    def test_url_joining_trims_duplicate_slashes(self):
        url = build_public_url("http://localhost:3001/", "/d/api-observability/backend-api")

        self.assertEqual(url, "http://localhost:3001/d/api-observability/backend-api")

    def test_links_return_configured_urls_for_admin(self):
        user = SimpleNamespace(is_admin=True)
        links = ObservabilityService(make_settings()).get_links(is_admin=True)

        self.assertEqual(
            links.dashboards.api.url,
            "http://localhost:3001/d/api-observability/backend-api",
        )
        self.assertEqual(links.prometheus_url.url, "http://localhost:9090/graph")
        self.assertTrue(links.tempo_explore_url.allowed)

    def test_missing_optional_urls_are_not_configured(self):
        user = SimpleNamespace(is_admin=True)
        links = ObservabilityService(
            make_settings(GRAFANA_FRONTEND_DASHBOARD_PATH="")
        ).get_links(is_admin=True)

        self.assertIsNone(links.dashboards.frontend.url)
        self.assertFalse(links.dashboards.frontend.configured)

    def test_non_admin_does_not_receive_technical_urls(self):
        links = ObservabilityService(make_settings()).get_links(is_admin=False)

        self.assertIsNone(links.prometheus_url.url)
        self.assertFalse(links.prometheus_url.allowed)


class ObservabilityStatusTest(unittest.IsolatedAsyncioTestCase):
    async def test_status_does_not_crash_when_tools_are_unconfigured(self):
        service = ObservabilityService(
            make_settings(
                GRAFANA_PUBLIC_URL="",
                PROMETHEUS_PUBLIC_URL="",
                TEMPO_PUBLIC_URL="",
            )
        )
        unknown = ObservabilityStatusItem(status="unknown", detail="check unavailable")

        async def passthrough_get_or_load_model(key, model, *, ttl_seconds, loader):
            return await loader()

        with (
            patch(
                "backend.observability.service.cache_get_or_load_model",
                passthrough_get_or_load_model,
            ),
            patch.object(service, "_database_status", AsyncMock(return_value=unknown)),
            patch.object(service, "_cache_status", AsyncMock(return_value=unknown)),
        ):
            status = await service.get_status()

        self.assertIn(status.prometheus.status, {"not_configured", "unknown", "down"})
        self.assertIn(status.grafana.status, {"not_configured", "unknown", "down"})
        self.assertIn(status.tempo.status, {"not_configured", "unknown", "down"})

    async def test_status_uses_cached_payload_on_repeat_calls(self):
        service = ObservabilityService(make_settings())
        built = make_observability_status()
        build_mock = AsyncMock(return_value=built)

        with (
            patch.object(service, "_build_status", build_mock),
            patch(
                "backend.observability.service.cache_get_or_load_model",
                AsyncMock(side_effect=[built, built]),
            ),
        ):
            first = await service.get_status()
            second = await service.get_status()

        self.assertEqual(first.api.status, "healthy")
        self.assertEqual(second.api.status, "healthy")
        build_mock.assert_not_called()

    async def test_http_status_reuses_shared_client(self):
        service = ObservabilityService(make_settings())
        created_clients: list[MagicMock] = []

        class FakeAsyncClient:
            is_closed = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created_clients.append(self)

            async def get(self, url):
                response = MagicMock()
                response.status_code = 200
                return response

        with patch("backend.observability.service.httpx.AsyncClient", FakeAsyncClient):
            await service._http_status("http://localhost:9090/graph", "Prometheus", "2026-01-01T00:00:00+00:00")
            await service._http_status("http://localhost:3001/", "Grafana", "2026-01-01T00:00:00+00:00")

        self.assertEqual(len(created_clients), 1)


if __name__ == "__main__":
    unittest.main()
