"""Cancel helpers and cooperative cancellation for analysis runs."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import AnalysisRun

TERMINAL_RUN_STATUSES = frozenset(
    {
        AnalysisRunStatus.COMPLETED.value,
        AnalysisRunStatus.FAILED.value,
        AnalysisRunStatus.CANCELLED.value,
    }
)

ACTIVE_RUN_STATUSES = frozenset(
    {
        AnalysisRunStatus.QUEUED.value,
        AnalysisRunStatus.RUNNING.value,
        "pending",
    }
)


class RunCancelledError(RuntimeError):
    """Raised when a worker observes a cancelled AnalysisRun."""


def is_terminal_status(status: str | None) -> bool:
    """Return True when ``status`` is a terminal AnalysisRun state."""
    return status in TERMINAL_RUN_STATUSES


async def ensure_not_cancelled(repo, run: AnalysisRun) -> AnalysisRun:
    """Reload run status; raise if the run was cancelled."""
    refreshed = await repo.get_run(run.id)
    if refreshed is None:
        raise ValueError(f"AnalysisRun {run.id} not found")
    if refreshed.status == AnalysisRunStatus.CANCELLED.value:
        raise RunCancelledError(f"AnalysisRun {run.id} was cancelled")
    return refreshed


async def complete_if_active(repo, run: AnalysisRun, **fields: Any) -> AnalysisRun | None:
    """Mark COMPLETED only while the run is still active.

    Refuses to overwrite CANCELLED / COMPLETED / FAILED. Returns ``None`` when
    the terminal write was skipped because another terminal state already won.
    """
    payload = dict(fields)
    payload["status"] = AnalysisRunStatus.COMPLETED.value
    updated = await repo.update_run_if_active(run, **payload)
    if updated is not None:
        _observe_terminal(updated)
    return updated


async def fail_if_active(repo, run: AnalysisRun, **fields: Any) -> AnalysisRun | None:
    """Mark FAILED only while the run is still active.

    Refuses to overwrite CANCELLED / COMPLETED / FAILED so cancellation remains
    terminal when a worker races an exception path after cancel.
    """
    payload = dict(fields)
    payload["status"] = AnalysisRunStatus.FAILED.value
    updated = await repo.update_run_if_active(run, **payload)
    if updated is not None:
        _observe_terminal(updated)
    return updated


async def cancel_if_active(repo, run: AnalysisRun, **fields: Any) -> AnalysisRun | None:
    """Mark CANCELLED only while the run is still active."""
    payload = dict(fields)
    payload["status"] = AnalysisRunStatus.CANCELLED.value
    updated = await repo.update_run_if_active(run, **payload)
    if updated is not None:
        _observe_terminal(updated)
    return updated


def _observe_terminal(run: AnalysisRun) -> None:
    try:
        from backend.modules.text_research.application.research_observability import (
            log_terminal_transition,
            result_payload_bytes,
        )

        artifact_bytes = None
        artifact_path = getattr(run, "artifact_path", None)
        if artifact_path:
            from pathlib import Path

            try:
                artifact_bytes = Path(str(artifact_path)).stat().st_size
            except OSError:
                artifact_bytes = None
        log_terminal_transition(
            run_id=str(run.id),
            analysis_type=getattr(run, "run_type", None),
            status=str(run.status),
            artifact_bytes=artifact_bytes,
            result_bytes=result_payload_bytes(
                getattr(run, "results", None),
                getattr(run, "metrics", None),
            ),
        )
    except Exception:
        # Observability must never break lifecycle transitions.
        return
