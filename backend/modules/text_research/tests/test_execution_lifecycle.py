"""Lifecycle guarantees for persisted background analysis runs."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.execution_service import ExecutionService
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import AnalysisRun, dumps


class ExecutionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_commits_identity_before_celery_publish(self):
        run = AnalysisRun(
            id="run-1",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="topic_model",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            parameters_json=dumps({"analysis_spec_hash": "spec-1"}),
        )
        commit_order: list[str] = []

        async def _commit():
            commit_order.append("commit")

        async def _refresh(_run):
            commit_order.append("refresh")

        db = MagicMock(commit=AsyncMock(side_effect=_commit), refresh=AsyncMock(side_effect=_refresh))

        def _queue(**kwargs):
            commit_order.append(f"publish:{kwargs.get('task_id')}")
            return "celery-1"

        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
            side_effect=_queue,
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
        self.assertEqual(commit_order[0], "commit")
        self.assertIn("refresh", commit_order)
        publish_steps = [step for step in commit_order if step.startswith("publish:")]
        self.assertEqual(len(publish_steps), 1)
        self.assertLess(commit_order.index("commit"), commit_order.index(publish_steps[0]))
        self.assertEqual(commit_order.count("commit"), 2)

    async def test_submit_skips_republish_when_task_id_present(self):
        run = AnalysisRun(
            id="run-2",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="topic_model",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            parameters_json=dumps({}),
            celery_task_id="existing-task",
            execution_key="abc",
        )
        db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
        ) as queue:
            await ExecutionService.submit(
                db=db, run=run, operation="topic_training", user_id="user-1"
            )
        queue.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_publish_then_task_id_commit_failure_leaves_recoverable_state(self):
        run = AnalysisRun(
            id="run-3",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="quantitative",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            parameters_json=dumps({"op": "frequencies"}),
        )
        commits = {"n": 0}

        async def _commit():
            commits["n"] += 1
            if commits["n"] == 2:
                raise RuntimeError("db down after publish")

        db = MagicMock(commit=AsyncMock(side_effect=_commit), refresh=AsyncMock())
        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
            return_value="celery-recover",
        ):
            with self.assertRaises(RuntimeError):
                await ExecutionService.submit(
                    db=db, run=run, operation="quantitative", user_id="user-1"
                )
        # Identity was committed first; celery_task_id was assigned in memory.
        self.assertTrue(run.execution_key)
        self.assertEqual(run.celery_task_id, "celery-recover")

    async def test_recover_queued_dispatch_republishes_missing_task_id(self):
        run = AnalysisRun(
            id="run-4",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="quantitative",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            parameters_json=dumps({}),
            execution_key="fixed-key",
        )
        db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
            return_value="celery-2",
        ) as queue:
            await ExecutionService.recover_queued_dispatch(
                db=db, run=run, operation="quantitative", user_id="user-1"
            )
        queue.assert_called_once()
        self.assertEqual(queue.call_args.kwargs["task_id"], "fixed-key")
        self.assertEqual(run.celery_task_id, "celery-2")

    async def test_publish_failure_does_not_set_celery_task_id(self):
        run = AnalysisRun(
            id="run-5",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="topic_model",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            parameters_json=dumps({}),
        )
        db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
        with patch(
            "backend.modules.text_research.workers.queue_research_operation",
            side_effect=RuntimeError("broker down"),
        ):
            with self.assertRaises(RuntimeError):
                await ExecutionService.submit(
                    db=db, run=run, operation="topic_training", user_id="user-1"
                )
        self.assertIsNone(run.celery_task_id)
        self.assertTrue(run.execution_key)
