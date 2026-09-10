"""Single application boundary for submitting long-running research runs."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.text_research.domain.models import AnalysisRun, loads

ResearchOperation = Literal[
    "segmentation",
    "classification",
    "topic_training",
    "topic_k_sweep",
    "topic_seed_stability",
    "robustness",
    "prediction",
    "quantitative",
]


class ExecutionService:
    """Submit a persisted run to the matching background-worker operation.

    Services create and commit their ``AnalysisRun`` before calling this class.
    Keeping dispatch here prevents request handlers from deciding how CPU-heavy
    work is scheduled.
    """

    @staticmethod
    async def submit(
        *, db: AsyncSession, run: AnalysisRun, operation: ResearchOperation, user_id: str
    ) -> None:
        """Persist execution identity and Celery task id around a single enqueue."""
        from backend.modules.text_research.workers import queue_research_operation

        params = loads(run.parameters_json, {}) or {}
        identity = {
            "run_type": run.run_type,
            "corpus_id": run.corpus_id,
            "spec_hash": params.get("analysis_spec_hash"),
            "parameters": params,
        }
        run.execution_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        run.artifact_namespace = f"runs/{run.project_id}/{run.id}"
        await db.flush()
        task_id = queue_research_operation(operation=operation, run_id=run.id, user_id=user_id)
        run.celery_task_id = task_id
        await db.commit()
