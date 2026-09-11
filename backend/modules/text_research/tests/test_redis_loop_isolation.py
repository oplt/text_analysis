"""Regression coverage for Redis clients used by API and eager worker loops."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Event
from unittest.mock import MagicMock, patch

from starlette.responses import Response

from backend.api.middleware.public_rate_limit import PublicRateLimitMiddleware
from backend.core import cache
from backend.modules.text_research.infrastructure import run_events
from backend.workers import async_dispatch


@dataclass
class _LoopBoundRedis:
    loop: asyncio.AbstractEventLoop
    calls: list[str] = field(default_factory=list)
    counter: int = 0
    closed: bool = False

    def _assert_owner(self, operation: str) -> None:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError(f"{operation} used from a different event loop")
        self.calls.append(operation)

    async def publish(self, _channel: str, _message: str) -> int:
        self._assert_owner("publish")
        return 1

    async def incr(self, _key: str) -> int:
        self._assert_owner("incr")
        self.counter += 1
        return self.counter

    async def expire(self, _key: str, _seconds: int) -> bool:
        self._assert_owner("expire")
        return True

    async def setex(self, _key: str, _seconds: int, _value: str) -> bool:
        self._assert_owner("setex")
        return True

    async def aclose(self) -> None:
        self._assert_owner("aclose")
        self.closed = True


class _ApiRequest:
    url = type("Url", (), {"path": "/api/v1/research/runs"})()
    client = type("Client", (), {"host": "127.0.0.1"})()


async def _allow_request(_request: _ApiRequest) -> Response:
    return Response(status_code=204)


def test_eager_worker_events_do_not_reuse_api_loop_redis_client() -> None:
    """Repeated API cache/rate-limit calls remain isolated from worker publications."""
    clients: list[_LoopBoundRedis] = []

    def create_client(*_args, **_kwargs) -> _LoopBoundRedis:
        client = _LoopBoundRedis(asyncio.get_running_loop())
        clients.append(client)
        return client

    async def publish_from_worker(index: int) -> None:
        await run_events.publish_run_event_envelope({"run": {"id": f"run-{index}"}})

    async def exercise_api_loop() -> None:
        api_client = cache.get_async_redis_client()
        middleware = PublicRateLimitMiddleware(app=None)
        for index in range(3):
            published = Event()
            errors: list[BaseException] = []

            def publish_research_operation(
                *,
                run_index: int,
                done: Event = published,
                error_bucket: list[BaseException] = errors,
            ) -> None:
                try:
                    async_dispatch.run_async_in_sync_context(publish_from_worker(run_index))
                except BaseException as exc:  # noqa: BLE001
                    error_bucket.append(exc)
                finally:
                    done.set()

            async_dispatch.dispatch_background_sync_job(
                target=publish_research_operation,
                kwargs={"run_index": index},
                celery_task=None,
                celery_kwargs={},
                queue="research_light",
                job_name="research-loop-isolation",
            )
            assert published.wait(timeout=3)
            assert not errors, f"eager worker publish failed: {errors[0]!r}"
            await cache.cache_set_json(
                f"ga:loop-isolation:{index}",
                {"index": index},
                ttl_seconds=60,
            )
            response = await middleware.dispatch(_ApiRequest(), _allow_request)
            assert response.status_code == 204
        assert cache.get_async_redis_client() is api_client
        await cache.close_current_async_redis_client()

    async_dispatch.shutdown_worker_async_runtime()
    try:
        rate_limit_settings = MagicMock()
        rate_limit_settings.effective_public_rate_limit_requests = 10
        rate_limit_settings.PUBLIC_RATE_LIMIT_WINDOW_SECONDS = 60
        with (
            patch("backend.core.cache.redis.from_url", side_effect=create_client),
            patch("backend.core.cache._uses_redis_cache", return_value=True),
            patch("backend.core.config.settings.CACHE_ENABLED", True),
            patch(
                "backend.api.middleware.public_rate_limit.settings",
                rate_limit_settings,
            ),
            patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True),
        ):
            asyncio.run(exercise_api_loop())
    finally:
        async_dispatch.shutdown_worker_async_runtime()

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.closed for client in clients)
    assert clients[1].calls.count("publish") == 3
    assert clients[0].calls.count("setex") == 3
    assert clients[0].calls.count("incr") == 3
