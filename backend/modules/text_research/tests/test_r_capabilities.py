"""R worker liveness and API capability aggregation tests."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.infrastructure.r_runtime import capabilities


class _AsyncRedis:
    def __init__(self, records: dict[str, str]) -> None:
        self.records = records

    async def get(self, key: str) -> str | None:
        return self.records.get(key)

    async def scan_iter(self, *, match: str):
        del match
        for key in self.records:
            yield key


def _record(
    worker_id: str,
    *,
    ready: bool = True,
    queues: list[str] | None = None,
    heartbeat_at: datetime | None = None,
) -> tuple[str, str]:
    payload = {
        "worker_id": worker_id,
        "queues": queues or ["research_r"],
        "ready": ready,
        "implementation": "quanteda",
        "implementation_version": "r-quanteda-2",
        "analyses": ["frequencies"],
        "runtime_version": "R version 4.4.0",
        "package_versions": {"quanteda": "4.2.0"},
        "heartbeat_at": (heartbeat_at or datetime.now(UTC)).isoformat(),
    }
    return capabilities.r_worker_capabilities_key(worker_id), json.dumps(payload)


def _live(records: dict[str, str]) -> list[dict[str, object]]:
    client = _AsyncRedis(records)
    with patch("backend.core.cache.get_async_redis_client", return_value=client):
        return asyncio.run(capabilities.get_live_r_worker_capabilities())


def test_r_feature_enabled_follows_settings_only() -> None:
    with patch.object(capabilities.settings, "RESEARCH_R_ENABLED", True):
        assert capabilities.r_feature_enabled() is True
    with patch.object(capabilities.settings, "RESEARCH_R_ENABLED", False):
        assert capabilities.r_feature_enabled() is False


def test_publish_creates_a_per_worker_record_with_runtime_metadata() -> None:
    store: dict[str, str] = {}
    fake = MagicMock()
    fake.setex.side_effect = lambda key, _ttl, value: store.__setitem__(key, value)

    with (
        patch.object(capabilities, "_sync_redis", return_value=fake),
        patch.object(
            capabilities,
            "_local_r_runtime_metadata",
            return_value=("R version 4.4.0", {"quanteda": "4.2.0"}),
        ),
    ):
        capabilities.publish_r_worker_capabilities(
            ready=True,
            analyses=["dfm", "frequencies"],
            worker_id="r@one",
            queues=["research_r"],
        )

    payload = json.loads(store[capabilities.r_worker_capabilities_key("r@one")])
    assert payload["worker_id"] == "r@one"
    assert payload["queues"] == ["research_r"]
    assert payload["runtime_version"] == "R version 4.4.0"
    assert payload["package_versions"] == {"quanteda": "4.2.0"}
    assert payload["heartbeat_at"]


def test_non_r_worker_cannot_advertise_r_readiness() -> None:
    with (
        patch.object(capabilities.settings, "RESEARCH_R_ENABLED", True),
        patch.object(capabilities, "publish_r_worker_capabilities") as publish,
        patch(
            "backend.modules.text_research.infrastructure.r_runtime.runner.r_runtime_available"
        ) as runtime_ready,
    ):
        ready = capabilities.refresh_r_worker_heartbeat_if_local_runtime_ready(
            worker_id="python@one",
            queues=["research_cpu"],
        )

    assert ready is False
    publish.assert_not_called()
    runtime_ready.assert_not_called()


def test_worker_without_observed_queues_cannot_advertise_r_readiness() -> None:
    with (
        patch.object(capabilities.settings, "RESEARCH_R_ENABLED", True),
        patch.object(capabilities, "publish_r_worker_capabilities") as publish,
        patch(
            "backend.modules.text_research.infrastructure.r_runtime.runner.r_runtime_available"
        ) as runtime_ready,
    ):
        ready = capabilities.refresh_r_worker_heartbeat_if_local_runtime_ready(
            worker_id="unknown@one",
        )

    assert ready is False
    publish.assert_not_called()
    runtime_ready.assert_not_called()


def test_live_aggregation_filters_expired_and_non_r_workers() -> None:
    live_key, live_payload = _record("r@one")
    expired_key, expired_payload = _record(
        "r@expired",
        heartbeat_at=datetime.now(UTC)
        - timedelta(seconds=capabilities.R_WORKER_CAPABILITIES_TTL_SECONDS + 1),
    )
    python_key, python_payload = _record("python@one", queues=["research_cpu"])

    live = _live(
        {
            live_key: live_payload,
            expired_key: expired_payload,
            python_key: python_payload,
        }
    )

    assert [item["worker_id"] for item in live] == ["r@one"]


def test_live_aggregation_keeps_remaining_worker_when_one_dies() -> None:
    first_key, first_payload = _record("r@one")
    second_key, second_payload = _record("r@two")
    records = {first_key: first_payload, second_key: second_payload}

    assert {item["worker_id"] for item in _live(records)} == {"r@one", "r@two"}
    del records[first_key]
    assert [item["worker_id"] for item in _live(records)] == ["r@two"]


def test_redis_unavailability_is_reported_as_no_live_workers() -> None:
    with patch(
        "backend.core.cache.get_async_redis_client",
        side_effect=RuntimeError("redis unavailable"),
    ):
        assert asyncio.run(capabilities.get_live_r_worker_capabilities()) == []


def test_analysis_engines_reports_enabled_r_as_unready_without_live_workers() -> None:
    from backend.modules.text_research.api import routes

    with (
        patch.object(capabilities, "r_feature_enabled", return_value=True),
        patch.object(capabilities, "get_live_r_worker_capabilities", AsyncMock(return_value=[])),
    ):
        payload = asyncio.run(routes.analysis_engines(current_user=MagicMock()))

    r_engine = next(engine for engine in payload["engines"] if engine["name"] == "r")
    assert r_engine["available"] is True
    assert r_engine["ready"] is False


def test_analysis_engines_reports_ready_when_a_live_r_worker_exists() -> None:
    from backend.modules.text_research.api import routes

    live = [{"worker_id": "r@one", "ready": True, "queues": ["research_r"]}]
    with (
        patch.object(capabilities, "r_feature_enabled", return_value=True),
        patch.object(capabilities, "get_live_r_worker_capabilities", AsyncMock(return_value=live)),
    ):
        payload = asyncio.run(routes.analysis_engines(current_user=MagicMock()))

    r_engine = next(engine for engine in payload["engines"] if engine["name"] == "r")
    assert r_engine["ready"] is True


def test_r_submission_guard_requires_a_live_worker_for_queued_execution() -> None:
    from backend.modules.text_research.application.quantitative_analysis_service import (
        QuantitativeAnalysisService,
    )

    service = object.__new__(QuantitativeAnalysisService)
    with (
        patch.object(capabilities, "r_feature_enabled", return_value=True),
        patch.object(capabilities, "get_live_r_worker_capabilities", AsyncMock(return_value=[])),
    ):
        try:
            asyncio.run(service._ensure_r_execution_capability(inline=False))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "worker" in str(exc.detail).lower()
        else:
            raise AssertionError("queued R execution must require a live worker")


def test_r_submission_guard_requires_local_runtime_for_inline_execution() -> None:
    from backend.modules.text_research.application.quantitative_analysis_service import (
        QuantitativeAnalysisService,
    )

    service = object.__new__(QuantitativeAnalysisService)
    with (
        patch.object(capabilities, "r_feature_enabled", return_value=True),
        patch(
            "backend.modules.text_research.application.quantitative_analysis_service.RAnalysisEngine.runtime_ready",
            return_value=False,
        ),
    ):
        try:
            asyncio.run(service._ensure_r_execution_capability(inline=True))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "local runtime" in str(exc.detail).lower()
        else:
            raise AssertionError("inline R execution must require the local runtime")


def test_r_submission_guard_rejects_disabled_feature_before_redis_lookup() -> None:
    from backend.modules.text_research.application.quantitative_analysis_service import (
        QuantitativeAnalysisService,
    )

    service = object.__new__(QuantitativeAnalysisService)
    with (
        patch.object(capabilities, "r_feature_enabled", return_value=False),
        patch.object(capabilities, "get_live_r_worker_capabilities", AsyncMock()) as live,
    ):
        try:
            asyncio.run(service._ensure_r_execution_capability(inline=False))
        except HTTPException as exc:
            assert exc.status_code == 503
        else:
            raise AssertionError("disabled R feature must reject submission")
    live.assert_not_awaited()


def test_idle_r_worker_timer_refreshes_and_cleanup_removes_its_record() -> None:
    from backend.workers import logging_hooks

    timer = MagicMock()
    handle = MagicMock()
    timer.call_repeatedly.return_value = handle
    queue = type("Queue", (), {"name": "research_r"})()
    sender = type(
        "Worker",
        (),
        {
            "hostname": "r@one",
            "task_consumer": type("Consumer", (), {"queues": [queue]})(),
            "timer": timer,
        },
    )()
    logging_hooks._r_worker_heartbeat_handles.clear()

    with (
        patch(
            "backend.modules.text_research.infrastructure.r_runtime.capabilities.r_feature_enabled",
            return_value=True,
        ),
        patch(
            "backend.modules.text_research.infrastructure.r_runtime.capabilities.refresh_r_worker_heartbeat_if_local_runtime_ready",
            return_value=True,
        ) as refresh,
        patch(
            "backend.modules.text_research.infrastructure.r_runtime.capabilities.remove_r_worker_capabilities"
        ) as remove,
    ):
        logging_hooks._start_r_worker_heartbeat(sender)
        callback = timer.call_repeatedly.call_args.args[1]
        callback(**timer.call_repeatedly.call_args.kwargs["kwargs"])
        logging_hooks.close_worker_async_runtime()

    assert refresh.call_count == 2
    handle.cancel.assert_called_once()
    remove.assert_called_once_with("r@one")
