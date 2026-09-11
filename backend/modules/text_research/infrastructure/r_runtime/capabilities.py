"""Per-worker liveness records for the optional R/quanteda execution runtime."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import redis

from backend.core.config import settings

logger = logging.getLogger(__name__)

R_WORKER_CAPABILITIES_PREFIX = "research:r_worker:"
R_WORKER_CAPABILITIES_SUFFIX = ":capabilities"
R_WORKER_CAPABILITIES_PATTERN = f"{R_WORKER_CAPABILITIES_PREFIX}*{R_WORKER_CAPABILITIES_SUFFIX}"
R_WORKER_HEARTBEAT_INTERVAL_SECONDS = 30
R_WORKER_CAPABILITIES_TTL_SECONDS = 90


def _r_engine_descriptor():
    from backend.modules.text_research.infrastructure.plugin_registry import (
        resolve_execution_engine,
    )

    return resolve_execution_engine("r")


def r_feature_enabled() -> bool:
    """True when the deployment intends to offer R analyses (API-side flag)."""
    return bool(settings.RESEARCH_R_ENABLED)


def r_worker_capabilities_key(worker_id: str) -> str:
    return f"{R_WORKER_CAPABILITIES_PREFIX}{worker_id}{R_WORKER_CAPABILITIES_SUFFIX}"


def worker_id_for_runtime(worker_id: str | None = None) -> str:
    """Resolve the stable Celery node name used to scope a liveness record."""
    return worker_id or socket.gethostname()


def worker_consumes_r_queue(queues: Iterable[str]) -> bool:
    return settings.RESEARCH_QUEUE_R in set(queues)


def _sync_redis() -> redis.Redis:
    """Dedicated synchronous Redis client for Celery signal/timer callbacks."""
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


@lru_cache(maxsize=1)
def _local_r_runtime_metadata() -> tuple[str | None, dict[str, str]]:
    """Read stable local R/quanteda versions once, without delaying every heartbeat."""
    command = (
        "cat(R.version.string, '\\n', as.character(utils::packageVersion('quanteda')), sep = '')"
    )
    try:
        completed = subprocess.run(
            [settings.RESEARCH_RSCRIPT_PATH, "--vanilla", "-e", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        runtime_version = lines[0] if lines else None
        package_versions = {"quanteda": lines[1]} if len(lines) > 1 else {}
        return runtime_version, package_versions
    except Exception:
        logger.debug("Unable to determine local R runtime metadata", exc_info=True)
        return None, {}


def publish_r_worker_capabilities(
    *,
    ready: bool,
    analyses: list[str] | None = None,
    worker_id: str | None = None,
    queues: Iterable[str] | None = None,
) -> None:
    """Write one TTL-bound capability record for an R-queue worker."""
    # An omitted queue list means that the caller has not established what this
    # process actually consumes.  Never infer R-queue membership from intent.
    queue_names = sorted(set(queues or []))
    if not worker_consumes_r_queue(queue_names):
        return
    runtime_version, package_versions = _local_r_runtime_metadata() if ready else (None, {})
    resolved_worker_id = worker_id_for_runtime(worker_id)
    descriptor = _r_engine_descriptor()
    if analyses is not None:
        resolved_analyses = analyses
    elif ready:
        resolved_analyses = descriptor.supported_analyses
    else:
        resolved_analyses = []
    payload = {
        "worker_id": resolved_worker_id,
        "queues": queue_names,
        "ready": ready,
        "implementation": descriptor.implementation,
        "implementation_version": descriptor.implementation_version,
        "analyses": sorted(set(resolved_analyses)),
        "runtime_version": runtime_version,
        "package_versions": package_versions,
        "heartbeat_at": datetime.now(UTC).isoformat(),
    }
    try:
        client = _sync_redis()
        client.setex(
            r_worker_capabilities_key(resolved_worker_id),
            R_WORKER_CAPABILITIES_TTL_SECONDS,
            json.dumps(payload),
        )
    except Exception:
        logger.debug("Failed to publish R worker capabilities", exc_info=True)


def remove_r_worker_capabilities(worker_id: str) -> None:
    """Best-effort cleanup of this worker's own liveness record on shutdown."""
    try:
        _sync_redis().delete(r_worker_capabilities_key(worker_id))
    except Exception:
        logger.debug("Failed to remove R worker capabilities", exc_info=True)


def _is_fresh(payload: dict[str, Any]) -> bool:
    raw_timestamp = payload.get("heartbeat_at")
    if not isinstance(raw_timestamp, str):
        return False
    try:
        timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - timestamp).total_seconds() < R_WORKER_CAPABILITIES_TTL_SECONDS


def _is_compatible_r_worker(payload: dict[str, Any]) -> bool:
    queues = payload.get("queues")
    descriptor = _r_engine_descriptor()
    return (
        bool(payload.get("ready"))
        and payload.get("implementation") == descriptor.implementation
        and payload.get("implementation_version") == descriptor.implementation_version
        and isinstance(queues, list)
        and worker_consumes_r_queue(str(queue) for queue in queues)
        and _is_fresh(payload)
    )


async def get_live_r_worker_capabilities() -> list[dict[str, Any]]:
    """Aggregate fresh, compatible R-worker records without blocking the API loop."""
    try:
        from backend.core.cache import get_async_redis_client

        client = get_async_redis_client()
        capabilities: list[dict[str, Any]] = []
        async for key in client.scan_iter(match=R_WORKER_CAPABILITIES_PATTERN):
            raw = await client.get(key)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and _is_compatible_r_worker(payload):
                capabilities.append(payload)
        return capabilities
    except Exception:
        logger.debug("Failed to aggregate R worker capabilities", exc_info=True)
        return []


def refresh_r_worker_heartbeat_if_local_runtime_ready(
    *,
    worker_id: str | None = None,
    queues: Iterable[str] | None = None,
) -> bool:
    """Refresh this R worker's record only when it consumes the R queue locally."""
    from backend.modules.text_research.infrastructure.r_runtime.runner import r_runtime_available

    # Queue membership must come from the live Celery consumer.  A default of
    # ``research_r`` would let an arbitrary worker masquerade as R-capable.
    queue_names = list(queues or [])
    if not r_feature_enabled() or not worker_consumes_r_queue(queue_names):
        return False
    ready = r_runtime_available()
    descriptor = _r_engine_descriptor()
    publish_r_worker_capabilities(
        ready=ready,
        analyses=sorted(descriptor.supported_analyses) if ready else [],
        worker_id=worker_id,
        queues=queue_names,
    )
    return ready
