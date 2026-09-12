"""Phase 21: cancelled background run + active-learning blind campaign hooks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.application.active_learning_service import ActiveLearningService
from backend.modules.text_research.application.run_service import RunService
from backend.modules.text_research.domain.enums import AnalysisRunStatus


class CancelledRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_queued_run_sets_cancelled_status(self) -> None:
        run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.QUEUED.value,
            artifact_namespace=None,
            celery_task_id=None,
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

        cancelled = await service.cancel_run("run-1", user_id="user-1")

        self.assertEqual(cancelled.status, AnalysisRunStatus.CANCELLED.value)
        update_kwargs = service.repo.update_run_if_active.await_args.kwargs
        self.assertEqual(update_kwargs["status"], AnalysisRunStatus.CANCELLED.value)
        self.assertTrue(update_kwargs["cancellation_requested"])
        service.db.commit.assert_awaited_once()

    async def test_cancel_completed_run_conflicts(self) -> None:
        run = SimpleNamespace(
            id="run-2",
            status=AnalysisRunStatus.COMPLETED.value,
            artifact_namespace=None,
        )
        service = RunService(MagicMock())
        service.get_run_or_404 = AsyncMock(return_value=run)
        with self.assertRaises(HTTPException) as ctx:
            await service.cancel_run("run-2", user_id="user-1")
        self.assertEqual(ctx.exception.status_code, 409)


class ActiveLearningBlindCampaignTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_to_annotation_delegates_without_breaking_blind_campaign(self) -> None:
        service = ActiveLearningService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=SimpleNamespace(id="model-1"))
        assigned = [SimpleNamespace(id="task-1")]

        with patch(
            "backend.modules.text_research.application.active_learning_service.AnnotationService"
        ) as annotation_cls:
            annotation_cls.return_value.assign_tasks = AsyncMock(return_value=assigned)
            tasks = await service.send_to_annotation(
                "model-1",
                user_id="user-1",
                text_unit_ids=["u1", "u2"],
                annotator_ids=["a1"],
            )

        self.assertEqual(tasks, assigned)
        annotation_cls.return_value.assign_tasks.assert_awaited_once_with(
            user_id="user-1",
            text_unit_ids=["u1", "u2"],
            annotator_ids=["a1"],
        )

    async def test_uncertain_queue_orders_by_uncertainty(self) -> None:
        service = ActiveLearningService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=SimpleNamespace(id="model-1"))
        predictions = [
            SimpleNamespace(text_unit_id="u2", uncertainty=0.9),
            SimpleNamespace(text_unit_id="u1", uncertainty=0.8),
        ]
        service.repo = MagicMock()
        service.repo.list_predictions_for_model = AsyncMock(return_value=(predictions, 2))
        service.repo.list_text_units_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(id="u1", text="one"),
                SimpleNamespace(id="u2", text="two"),
            ]
        )

        items, total = await service.uncertain_queue("model-1", user_id="user-1", limit=10)
        self.assertEqual(total, 2)
        self.assertEqual(len(items), 2)
        service.repo.list_predictions_for_model.assert_awaited_once()
        kwargs = service.repo.list_predictions_for_model.await_args.kwargs
        self.assertTrue(kwargs.get("order_by_uncertainty"))


if __name__ == "__main__":
    unittest.main()
