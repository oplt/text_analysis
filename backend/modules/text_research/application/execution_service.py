"""Single application boundary for submitting long-running research runs."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import AnalysisRun, loads

logger = logging.getLogger(__name__)

ResearchOperation = Literal[
    "segmentation",
    "classification",
    "topic_training",
    "topic_k_sweep",
    "topic_seed_stability",
    "robustness",
    "prediction",
    "quantitative",
    "corpus_synthesis",
]


class ExecutionService:
    """Submit a persisted run to the matching background-worker operation.

    Services create and commit their ``AnalysisRun`` before calling this class.
    Keeping dispatch here prevents request handlers from deciding how CPU-heavy
    work is scheduled.

    Dispatch contract (TASK-014):
    1. Persist execution identity + QUEUED metadata and commit durably.
    2. Publish Celery work keyed by ``execution_key`` (idempotent task id).
    3. Persist ``celery_task_id`` in a guarded follow-up commit.
    4. Retries reuse the same execution key / task id without double-publishing.
    """

    @staticmethod
    def _compute_execution_key(run: AnalysisRun) -> str:
        params = loads(run.parameters_json, {}) or {}
        identity = {
            "run_id": run.id,
            "run_type": run.run_type,
            "corpus_id": run.corpus_id,
            "spec_hash": params.get("analysis_spec_hash"),
            "parameters": params,
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    @staticmethod
    async def submit(
        *, db: AsyncSession, run: AnalysisRun, operation: ResearchOperation, user_id: str
    ) -> None:
        """Commit dispatch metadata before Celery publish; recover on retry."""
        from backend.modules.text_research.workers import queue_research_operation

        # Idempotent re-entry: already dispatched successfully.
        if run.celery_task_id:
            logger.info(
                "execution submit skipped; celery_task_id already set run=%s task=%s",
                run.id,
                run.celery_task_id,
            )
            return

        if not run.execution_key:
            run.execution_key = ExecutionService._compute_execution_key(run)
        run.artifact_namespace = run.artifact_namespace or f"runs/{run.project_id}/{run.id}"
        if run.status not in {
            AnalysisRunStatus.QUEUED.value,
            AnalysisRunStatus.RUNNING.value,
        }:
            run.status = AnalysisRunStatus.QUEUED.value
            run.progress_stage = run.progress_stage or "queued"

        # Durable commit BEFORE publish — survives publish/DB failure windows.
        await db.commit()
        await db.refresh(run)

        if run.celery_task_id:
            return

        task_id = queue_research_operation(
            operation=operation,
            run_id=run.id,
            user_id=user_id,
            task_id=run.execution_key,
        )
        # Eager mode returns None; store execution_key as stable dispatch marker.
        run.celery_task_id = task_id or f"eager:{run.execution_key}"
        try:
            await db.commit()
        except Exception:
            logger.exception(
                "failed to persist celery_task_id after publish run=%s task=%s",
                run.id,
                run.celery_task_id,
            )
            # Best-effort recovery on next submit: identity already durable;
            # celery_task_id may be missing — recover_queued_dispatch will republish
            # with the same task_id (Celery idempotent).
            raise

    @staticmethod
    async def recover_queued_dispatch(
        *, db: AsyncSession, run: AnalysisRun, operation: ResearchOperation, user_id: str
    ) -> None:
        """Re-publish for QUEUED runs that never stored a celery_task_id."""
        if run.status != AnalysisRunStatus.QUEUED.value:
            return
        if run.celery_task_id:
            return
        await ExecutionService.submit(db=db, run=run, operation=operation, user_id=user_id)
