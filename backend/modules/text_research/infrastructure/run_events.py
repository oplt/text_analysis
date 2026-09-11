"""Redis Pub/Sub transport for analysis-run SSE events.

PostgreSQL remains the durable source of truth. Redis carries ephemeral
notifications so SSE endpoints do not poll the database every second.

Channel: ``research:run:{run_id}:events``

Envelope::

    {"event": "progress", "run": {<AnalysisRunResponse JSON>}, "published_at": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.domain.models import AnalysisRun, loads

logger = logging.getLogger(__name__)

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
RUN_EVENT_CHANNEL_PREFIX = "research:run:"
RUN_EVENT_CHANNEL_SUFFIX = ":events"


def run_events_channel(run_id: str) -> str:
    return f"{RUN_EVENT_CHANNEL_PREFIX}{run_id}{RUN_EVENT_CHANNEL_SUFFIX}"


def reset_run_events_redis_for_tests() -> None:
    """Compatibility hook for tests; the async Redis client is stateless here."""


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
        "run_version": int(getattr(run, "run_version", 1) or 1),
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


def run_event_envelope(
    run: AnalysisRun,
    *,
    previous: dict[str, Any] | None = None,
    event: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a post-commit-safe SSE envelope without performing I/O."""
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
    return event_name, envelope


async def publish_run_event_envelope(envelope: dict[str, Any]) -> None:
    """Publish through the async Redis client; failures leave DB reconciliation intact."""
    try:
        from backend.core.cache import get_async_redis_client
        from backend.core.config import settings

        if not getattr(settings, "CACHE_ENABLED", True):
            return
        run_id = str((envelope.get("run") or {}).get("id") or "")
        if not run_id:
            return
        await get_async_redis_client().publish(
            run_events_channel(run_id),
            json.dumps(envelope, ensure_ascii=True, default=str),
        )
    except Exception:
        # Do not latch failures: the next durable commit retries Redis, while
        # SSE clients continue to reconcile from PostgreSQL.
        logger.debug("run events: async publish failed", exc_info=True)


def _schedule_run_event_publish(envelope: dict[str, Any]) -> None:
    """Publish on the current loop, or the dedicated worker loop when sync.

    Never borrows another loop's Redis client: ``get_async_redis_client`` is
    always resolved on the loop that actually awaits the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        from backend.workers.async_dispatch import run_async_in_sync_context

        run_async_in_sync_context(publish_run_event_envelope(envelope))
        return
    loop.create_task(publish_run_event_envelope(envelope))


def publish_run_event(
    run: AnalysisRun,
    *,
    previous: dict[str, Any] | None = None,
    event: str | None = None,
) -> str:
    """Schedule publication on the owning event loop (API or worker)."""
    event_name, envelope = run_event_envelope(run, previous=previous, event=event)
    _schedule_run_event_publish(envelope)
    return event_name


def queue_run_event(session: Any, run: AnalysisRun, *, previous: dict[str, Any] | None) -> None:
    """Queue a snapshot for the SQLAlchemy session's post-commit hook."""
    _event_name, envelope = run_event_envelope(run, previous=previous)
    session.info.setdefault("research_run_events", []).append(envelope)


def snapshot_fields(run: AnalysisRun) -> dict[str, Any]:
    return {
        "status": run.status,
        "progress_stage": run.progress_stage,
        "artifact_path": run.artifact_path,
    }
