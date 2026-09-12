"""TASK-003: cancellation is a terminal AnalysisRun state across workers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.application.prediction_service import PredictionService
from backend.modules.text_research.application.run_lifecycle import (
    ACTIVE_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    RunCancelledError,
    cancel_if_active,
    complete_if_active,
    ensure_not_cancelled,
    fail_if_active,
)
from backend.modules.text_research.application.run_service import RunService
from backend.modules.text_research.application.topic_model_service import TopicModelService
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps


class RunLifecycleGuardsTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_if_active_refuses_terminal_statuses(self) -> None:
        for status in TERMINAL_RUN_STATUSES:
            run = SimpleNamespace(id="run-1", status=status)
            repo = MagicMock()
            repo.update_run_if_active = AsyncMock(return_value=None)
            result = await complete_if_active(repo, run, progress_stage="completed")
            self.assertIsNone(result)
            kwargs = repo.update_run_if_active.await_args.kwargs
            self.assertEqual(kwargs["status"], AnalysisRunStatus.COMPLETED.value)

    async def test_fail_if_active_refuses_cancelled(self) -> None:
        run = SimpleNamespace(id="run-1", status=AnalysisRunStatus.CANCELLED.value)
        repo = MagicMock()
        repo.update_run_if_active = AsyncMock(return_value=None)
        result = await fail_if_active(repo, run, error_message="boom")
        self.assertIsNone(result)
        self.assertEqual(
            repo.update_run_if_active.await_args.kwargs["status"],
            AnalysisRunStatus.FAILED.value,
        )

    async def test_ensure_not_cancelled_raises(self) -> None:
        run = SimpleNamespace(id="run-1")
        repo = MagicMock()
        repo.get_run = AsyncMock(
            return_value=SimpleNamespace(
                id="run-1", status=AnalysisRunStatus.CANCELLED.value
            )
        )
        with self.assertRaises(RunCancelledError):
            await ensure_not_cancelled(repo, run)

    async def test_concurrent_cancel_and_complete_race(self) -> None:
        """Whichever terminal transition wins first blocks the other."""
        run = SimpleNamespace(
            id="run-race",
            status=AnalysisRunStatus.RUNNING.value,
            run_version=1,
        )
        state = {"status": AnalysisRunStatus.RUNNING.value}

        async def _update_if_active(locked_run, **fields):
            if state["status"] not in ACTIVE_RUN_STATUSES:
                return None
            state["status"] = fields["status"]
            for key, value in fields.items():
                setattr(locked_run, key, value)
            return locked_run

        repo = MagicMock()
        repo.update_run_if_active = AsyncMock(side_effect=_update_if_active)

        cancelled = await cancel_if_active(
            repo,
            run,
            cancellation_requested=True,
            progress_stage="cancelled",
            error_message="Cancelled by user",
        )
        completed = await complete_if_active(
            repo,
            run,
            progress_stage="completed",
        )

        self.assertIsNotNone(cancelled)
        self.assertIsNone(completed)
        self.assertEqual(state["status"], AnalysisRunStatus.CANCELLED.value)

        # Reverse order: completion first then cancel.
        state["status"] = AnalysisRunStatus.RUNNING.value
        run.status = AnalysisRunStatus.RUNNING.value
        completed = await complete_if_active(repo, run, progress_stage="completed")
        cancelled = await cancel_if_active(
            repo,
            run,
            cancellation_requested=True,
            progress_stage="cancelled",
        )
        self.assertIsNotNone(completed)
        self.assertIsNone(cancelled)
        self.assertEqual(state["status"], AnalysisRunStatus.COMPLETED.value)


class CancelQueuedCeleryRevokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_queued_run_revokes_celery_task_softly(self) -> None:
        run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.QUEUED.value,
            artifact_namespace=None,
            celery_task_id="celery-task-1",
        )
        cancelled_run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.CANCELLED.value,
            cancellation_requested=True,
            artifact_namespace=None,
        )
        service = RunService(MagicMock())
        service.get_run_or_404 = AsyncMock(return_value=run)
        service.repo = MagicMock()
        service.repo.update_run_if_active = AsyncMock(return_value=cancelled_run)
        service.repo.get_run = AsyncMock(return_value=cancelled_run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        with patch("backend.workers.celery_app.celery_app") as celery_app:
            celery_app.control.revoke = MagicMock()
            result = await service.cancel_run("run-1", user_id="user-1")

        self.assertEqual(result.status, AnalysisRunStatus.CANCELLED.value)
        celery_app.control.revoke.assert_called_once_with(
            "celery-task-1", terminate=False
        )

    async def test_cancel_uses_guarded_transition(self) -> None:
        run = SimpleNamespace(
            id="run-2",
            status=AnalysisRunStatus.RUNNING.value,
            artifact_namespace=None,
            celery_task_id=None,
        )
        service = RunService(MagicMock())
        service.get_run_or_404 = AsyncMock(return_value=run)
        service.repo = MagicMock()
        service.repo.update_run_if_active = AsyncMock(return_value=None)
        service.repo.get_run = AsyncMock(
            return_value=SimpleNamespace(
                id="run-2", status=AnalysisRunStatus.COMPLETED.value
            )
        )
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        with self.assertRaises(HTTPException) as ctx:
            await service.cancel_run("run-2", user_id="user-1")
        self.assertEqual(ctx.exception.status_code, 409)


class TopicCancelDuringTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_mocked_training_stays_cancelled(self) -> None:
        run = SimpleNamespace(
            id="topic-run",
            status=AnalysisRunStatus.QUEUED.value,
            parameters_json=dumps(
                {
                    "unit_type": "paragraph",
                    "algorithm": "lda",
                    "n_topics": 2,
                    "max_iterations": 5,
                    "random_seed": 1,
                    "filters": {},
                }
            ),
            corpus_id="corpus-1",
            created_by="user-1",
        )
        cancelled = SimpleNamespace(
            id="topic-run",
            status=AnalysisRunStatus.CANCELLED.value,
            parameters_json=run.parameters_json,
            corpus_id="corpus-1",
            created_by="user-1",
        )
        service = TopicModelService(MagicMock())
        service.repo = MagicMock()
        service.db = MagicMock()
        service.db.commit = AsyncMock()
        service.repo.get_run = AsyncMock(side_effect=[run, cancelled])
        service.repo.update_run = AsyncMock(return_value=run)
        service.repo.update_run_if_active = AsyncMock(return_value=None)
        service._select_texts = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="u1",
                    text="alpha beta gamma",
                    corpus_document_id="d1",
                )
            ]
        )
        service.repo.get_preprocessing_profile = AsyncMock(return_value=None)

        prepared = SimpleNamespace(
            texts_joined=["alpha beta gamma"],
            original_units=["alpha beta gamma"],
            unit_ids=["u1"],
            corpus_checksum="c1",
            pipeline_checksum="p1",
        )

        with (
            patch(
                "backend.modules.text_research.application.topic_model_service.prepare_texts_cached_async",
                AsyncMock(return_value=prepared),
            ),
            patch(
                "backend.modules.text_research.application.topic_model_service.train_topic_model",
                side_effect=AssertionError("training should not run after cancel"),
            ),
            patch(
                "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
                new=AsyncMock(side_effect=RunCancelledError("cancelled")),
            ),
            patch(
                "backend.modules.text_research.application.run_lifecycle.complete_if_active",
                new=AsyncMock(side_effect=AssertionError("must not complete")),
            ),
        ):
            result = await service.execute_training("topic-run")

        self.assertEqual(result.status, AnalysisRunStatus.CANCELLED.value)


class PredictionCancelAfterBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_after_first_batch_skips_later_completion(self) -> None:
        run = SimpleNamespace(
            id="pred-run",
            status=AnalysisRunStatus.QUEUED.value,
            parameters_json=dumps(
                {
                    "model_id": "model-1",
                    "unit_type": "paragraph",
                    "only_unannotated": False,
                    "filters": {},
                }
            ),
            corpus_id="corpus-1",
            created_by="user-1",
            project_id="project-1",
        )
        cancelled = SimpleNamespace(
            id="pred-run",
            status=AnalysisRunStatus.CANCELLED.value,
            parameters_json=run.parameters_json,
            corpus_id="corpus-1",
            created_by="user-1",
            project_id="project-1",
        )
        model = SimpleNamespace(
            id="model-1",
            corpus_id="corpus-1",
            task_type="binary",
            vectorizer_artifact_path="vec.joblib",
            model_artifact_path="model.joblib",
            label_ids_json=dumps(["neg", "pos"]),
            metrics_json=dumps({}),
        )
        units = [
            SimpleNamespace(id="u1", text="one", corpus_document_id="d1"),
            SimpleNamespace(id="u2", text="two", corpus_document_id="d1"),
            SimpleNamespace(id="u3", text="three", corpus_document_id="d1"),
        ]

        service = PredictionService(MagicMock())
        service.repo = MagicMock()
        service.db = MagicMock()
        service.db.commit = AsyncMock()
        service.repo.get_run = AsyncMock(side_effect=[run, cancelled])
        service.repo.update_run = AsyncMock(return_value=run)
        service.repo.update_run_if_active = AsyncMock(return_value=None)
        service.repo.get_model = AsyncMock(return_value=model)
        service.repo.list_documents = AsyncMock(return_value=[])
        service.repo.count_text_units_for_corpus = AsyncMock(return_value=3)

        async def _iter_units(*_args, **_kwargs):
            for unit in units:
                yield [unit]

        service.repo.iter_text_units_for_corpus = _iter_units
        service.repo.bulk_upsert_predictions = AsyncMock()

        checkpoint_calls = {"n": 0}

        async def _checkpoint(_repo, current):
            checkpoint_calls["n"] += 1
            # 1: after mark running / load model path
            # 2: before artifact load
            # 3: before first batch — allow
            # 4: before second batch — cancel
            if checkpoint_calls["n"] >= 4:
                raise RunCancelledError("cancelled after first batch")
            return current

        with (
            patch(
                "backend.modules.text_research.application.prediction_service.model_storage.load_artifact",
                return_value=object(),
            ),
            patch(
                "backend.modules.text_research.application.prediction_service.predict_with_uncertainty",
                return_value=[
                    {"prediction": 1, "probability": 0.9, "uncertainty": 0.1}
                ],
            ),
            patch(
                "backend.modules.text_research.application.prediction_service.resolve_batch_size",
                return_value=1,
            ),
            patch(
                "backend.modules.text_research.application.prediction_service.should_use_out_of_core",
                return_value=True,
            ),
            patch(
                "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
                new=AsyncMock(side_effect=_checkpoint),
            ),
            patch(
                "backend.modules.text_research.application.run_lifecycle.complete_if_active",
                new=AsyncMock(side_effect=AssertionError("must not complete")),
            ),
            patch(
                "backend.modules.text_research.application.prediction_set_service.PredictionSetService"
            ) as prediction_set_cls,
        ):
            prediction_set_cls.return_value.create_from_run = AsyncMock()
            result = await service.execute_prediction("pred-run")

        self.assertEqual(result.status, AnalysisRunStatus.CANCELLED.value)
        self.assertEqual(service.repo.bulk_upsert_predictions.await_count, 1)
        prediction_set_cls.return_value.create_from_run.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
