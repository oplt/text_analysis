"""Lifecycle guarantees for persisted background analysis runs."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
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

        db = MagicMock(
            commit=AsyncMock(side_effect=_commit),
            refresh=AsyncMock(side_effect=_refresh),
        )

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
        with (
            patch(
                "backend.modules.text_research.workers.queue_research_operation",
                return_value="celery-recover",
            ),
            self.assertRaises(RuntimeError),
        ):
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
        with (
            patch(
                "backend.modules.text_research.workers.queue_research_operation",
                side_effect=RuntimeError("broker down"),
            ),
            self.assertRaises(RuntimeError),
        ):
            await ExecutionService.submit(
                db=db, run=run, operation="topic_training", user_id="user-1"
            )
        self.assertIsNone(run.celery_task_id)
        self.assertTrue(run.execution_key)

    async def test_claim_allows_only_one_worker_to_execute(self):
        db = MagicMock(commit=AsyncMock())
        queued = SimpleNamespace(
            id="run-claim",
            status=AnalysisRunStatus.QUEUED.value,
            execution_key="execution-key",
        )
        with patch(
            "backend.modules.text_research.infrastructure.repositories.ResearchRepository"
        ) as repo_cls:
            repo = repo_cls.return_value
            repo.get_run = AsyncMock(return_value=queued)
            repo.claim_run_for_execution = AsyncMock(
                side_effect=[SimpleNamespace(id="run-claim"), None]
            )
            first = await ExecutionService.claim_run_for_execution(
                db=db, run_id="run-claim", worker_id="w1"
            )
            second = await ExecutionService.claim_run_for_execution(
                db=db, run_id="run-claim", worker_id="w2"
            )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(repo.claim_run_for_execution.await_count, 2)
        self.assertEqual(
            repo.claim_run_for_execution.await_args_list[0].kwargs["execution_key"],
            "execution-key",
        )
        self.assertEqual(
            repo.claim_run_for_execution.await_args_list[0].kwargs["worker_id"],
            "w1",
        )
        db.commit.assert_awaited_once()

    async def test_claim_misses_when_run_missing(self):
        db = MagicMock(commit=AsyncMock())
        with patch(
            "backend.modules.text_research.infrastructure.repositories.ResearchRepository"
        ) as repo_cls:
            repo = repo_cls.return_value
            repo.get_run = AsyncMock(return_value=None)
            repo.claim_run_for_execution = AsyncMock()
            claimed = await ExecutionService.claim_run_for_execution(db=db, run_id="missing")
        self.assertFalse(claimed)
        repo.claim_run_for_execution.assert_not_awaited()
        db.commit.assert_not_awaited()


class PredictionWorkerDeliveryTests(unittest.TestCase):
    def test_duplicate_prediction_delivery_executes_only_the_claimed_worker(self) -> None:
        from backend.modules.text_research import workers

        def _run_in_test_session(factory):
            asyncio.run(factory(MagicMock()))

        with (
            patch.object(workers, "_run_with_session", side_effect=_run_in_test_session),
            patch.object(
                workers,
                "_claim_run_for_execution",
                new=AsyncMock(side_effect=[True, False]),
            ),
            patch(
                "backend.modules.text_research.application.prediction_service.PredictionService"
            ) as service_cls,
        ):
            service_cls.return_value.execute_prediction = AsyncMock()
            workers.prediction_sync(run_id="prediction-run", user_id="user-1")
            workers.prediction_sync(run_id="prediction-run", user_id="user-1")

        service_cls.return_value.execute_prediction.assert_awaited_once_with("prediction-run")

    def test_duplicate_topic_and_classifier_deliveries_respect_claim(self) -> None:
        from backend.modules.text_research import workers

        def _run_in_test_session(factory):
            asyncio.run(factory(MagicMock()))

        cases = [
            (
                "classifier_training_sync",
                "backend.modules.text_research.application.classification_service.ClassificationService",
                "execute_training",
            ),
            (
                "topic_model_training_sync",
                "backend.modules.text_research.application.topic_model_service.TopicModelService",
                "execute_training",
            ),
            (
                "robustness_sweep_sync",
                "backend.modules.text_research.application.robustness_service.RobustnessService",
                "execute_sweep",
            ),
            (
                "corpus_synthesis_sync",
                "backend.modules.text_research.application.corpus_synthesis_service.CorpusSynthesisService",
                "execute_synthesis",
            ),
            (
                "segmentation_sync",
                "backend.modules.text_research.application.segmentation_service.SegmentationService",
                "execute_segmentation",
            ),
        ]
        for sync_name, service_path, method_name in cases:
            with (
                self.subTest(sync_name=sync_name),
                patch.object(workers, "_run_with_session", side_effect=_run_in_test_session),
                patch.object(
                    workers,
                    "_claim_run_for_execution",
                    new=AsyncMock(side_effect=[True, False]),
                ),
                patch(service_path) as service_cls,
            ):
                setattr(service_cls.return_value, method_name, AsyncMock())
                sync_fn = getattr(workers, sync_name)
                sync_fn(run_id=f"{sync_name}-run", user_id="user-1")
                sync_fn(run_id=f"{sync_name}-run", user_id="user-1")
                getattr(service_cls.return_value, method_name).assert_awaited_once()


class RepositoryClaimSqlTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_update_requires_queued_status(self) -> None:
        from backend.modules.text_research.infrastructure.repositories import ResearchRepository

        db = MagicMock()
        claimed = SimpleNamespace(id="run-1", status="running")
        execute_result = MagicMock()
        execute_result.scalar_one_or_none = MagicMock(return_value=claimed)
        db.execute = AsyncMock(return_value=execute_result)

        repo = ResearchRepository(db)
        result = await repo.claim_run_for_execution(
            "run-1", execution_key="key-1", worker_id="worker-a"
        )
        self.assertIs(result, claimed)
        statement = db.execute.await_args.args[0]
        compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))
        self.assertIn("research_analysis_runs", compiled.lower())
        # Conditional claim must gate on queued status.
        self.assertTrue(
            any(
                getattr(clause, "left", None) is not None
                and getattr(getattr(clause, "left", None), "key", None) == "status"
                for clause in statement.whereclause.get_children()
            )
            or "queued" in compiled.lower()
            or "status" in compiled.lower()
        )
