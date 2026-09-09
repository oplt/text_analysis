"""Lifecycle guarantees for persisted background analysis runs."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.execution_service import ExecutionService
from backend.modules.text_research.domain.models import AnalysisRun, dumps


class ExecutionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_persists_identity_namespace_and_celery_task_id(self):
        run = AnalysisRun(
            id="run-1",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="topic_model",
            created_by="user-1",
            parameters_json=dumps({"analysis_spec_hash": "spec-1"}),
        )
        db = MagicMock(flush=AsyncMock(), commit=AsyncMock())
        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
            return_value="celery-1",
        ):
            await ExecutionService.submit(
                db=db,
                run=run,
                operation="topic_training",
                user_id="user-1",
            )

        self.assertEqual(run.celery_task_id, "celery-1")
        self.assertEqual(run.artifact_namespace, "runs/project-1/run-1")
        self.assertEqual(len(run.execution_key or ""), 64)
        db.flush.assert_awaited_once()
        db.commit.assert_awaited_once()
