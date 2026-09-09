"""Redis Pub/Sub transport for analysis-run SSE events.

PostgreSQL remains the durable source of truth. Redis carries ephemeral
notifications so SSE endpoints do not poll the database every second.

Channel: ``research:run:{run_id}:events``

Envelope::

    {"event": "progress", "run": {<AnalysisRunResponse JSON>}, "published_at": "..."}
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.domain.models import AnalysisRun, loads

logger = logging.getLogger(__name__)

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
RUN_EVENT_CHANNEL_PREFIX = "research:run:"
RUN_EVENT_CHANNEL_SUFFIX = ":events"

_sync_redis: Any | None = None
_sync_redis_failed = False


def run_events_channel(run_id: str) -> str:
    return f"{RUN_EVENT_CHANNEL_PREFIX}{run_id}{RUN_EVENT_CHANNEL_SUFFIX}"


def reset_run_events_redis_for_tests() -> None:
    global _sync_redis, _sync_redis_failed
    _sync_redis = None
    _sync_redis_failed = False


def _get_sync_redis() -> Any | None:
    global _sync_redis, _sync_redis_failed
    if _sync_redis_failed:
        return None
    if _sync_redis is not None:
        return _sync_redis
    try:
        import redis

        from backend.core.config import settings

        if not getattr(settings, "CACHE_ENABLED", True):
            _sync_redis_failed = True
            return None
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        _sync_redis = client
        return _sync_redis
    except Exception:
        logger.debug("run events: Redis unavailable", exc_info=True)
        _sync_redis_failed = True
        return None


def serialize_run(run: AnalysisRun) -> dict[str, Any]:
    """Build the JSON body streamed to SSE clients (matches AnalysisRunResponse)."""

    def _dt(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()

    return {
        "id": run.id,
        "project_id": run.project_id,
        "corpus_id": run.corpus_id,
        "run_type": run.run_type,
        "status": run.status,
        "progress_stage": run.progress_stage,
        "parameters": loads(run.parameters_json),
        "metrics": loads(run.metrics_json),
        "results": loads(run.results_json),
        "artifact_path": run.artifact_path,
        "random_seed": run.random_seed,
        "created_by": run.created_by,
        "started_at": _dt(run.started_at),
        "completed_at": _dt(run.completed_at),
        "error_message": run.error_message,
        "created_at": _dt(run.created_at) or datetime.now(UTC).isoformat(),
    }


def run_event_name(
    *,
    status: str,
    progress_stage: str | None,
    artifact_path: str | None,
    previous: dict[str, Any] | None,
) -> str:
    """Name the SSE event for a transition from ``previous`` → current fields."""
    if previous is None or status != previous.get("status"):
        if status in TERMINAL_RUN_STATUSES | {"queued"}:
            return status
        return "started" if status in {"running", "pending"} else "progress"
    if artifact_path and artifact_path != previous.get("artifact_path"):
        return "artifact-created"
    if progress_stage != previous.get("progress_stage"):
        return "progress"
    return "progress"


def publish_run_event(
    run: AnalysisRun,
    *,
    previous: dict[str, Any] | None = None,
    event: str | None = None,
) -> str | None:
    """Publish a run snapshot to Redis. Returns event name, or None if skipped."""
    event_name = event or run_event_name(
        status=run.status,
        progress_stage=run.progress_stage,
        artifact_path=run.artifact_path,
        previous=previous,
    )
    envelope = {
        "event": event_name,
        "run": serialize_run(run),
        "published_at": datetime.now(UTC).isoformat(),
    }
    client = _get_sync_redis()
    if client is None:
        return event_name
    channel = run_events_channel(run.id)
    try:
        client.publish(channel, json.dumps(envelope, ensure_ascii=True, default=str))
    except Exception:
        logger.debug("run events: publish failed run_id=%s", run.id, exc_info=True)
    return event_name


def snapshot_fields(run: AnalysisRun) -> dict[str, Any]:
    return {
        "status": run.status,
        "progress_stage": run.progress_stage,
        "artifact_path": run.artifact_path,
    }
