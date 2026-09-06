"""Cancel helpers and cooperative cancellation for analysis runs."""

from __future__ import annotations

from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import AnalysisRun


class RunCancelledError(RuntimeError):
    """Raised when a worker observes a cancelled AnalysisRun."""


async def ensure_not_cancelled(repo, run: AnalysisRun) -> AnalysisRun:
    """Reload run status; raise if the run was cancelled."""
    refreshed = await repo.get_run(run.id)
    if refreshed is None:
        raise ValueError(f"AnalysisRun {run.id} not found")
    if refreshed.status == AnalysisRunStatus.CANCELLED.value:
        raise RunCancelledError(f"AnalysisRun {run.id} was cancelled")
    return refreshed
