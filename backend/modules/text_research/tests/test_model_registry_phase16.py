"""Phase 16: model registry lifecycle + drift service persistence."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.text_research.application.drift_service import DriftService
from backend.modules.text_research.application.model_lifecycle_service import (
    ModelLifecycleService,
)
from backend.modules.text_research.domain.models import TrainedModel


def _model(*, model_id: str = "model-1", status: str = "candidate") -> TrainedModel:
    model = TrainedModel(
        project_id="proj-1",
        corpus_id="corpus-1",
        analysis_run_id="run-1",
        training_dataset_snapshot_id="snap-1",
        model_family="logistic_regression",
        task_type="binary",
        label_ids_json='["no","yes"]',
        feature_config_json="{}",
        training_config_json="{}",
        metrics_json="{}",
        model_artifact_path="/tmp/model.joblib",
        vectorizer_artifact_path="/tmp/vec.joblib",
        created_by="user-1",
    )
    model.id = model_id
    model.lifecycle_status = status
    return model


class ModelLifecycleTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_to_approved_to_deprecated(self) -> None:
        model = _model()
        service = ModelLifecycleService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=model)
        service.repo.get_model = AsyncMock(return_value=model)
        service.db.flush = AsyncMock()
        service.db.commit = AsyncMock()

        approved = await service.set_status("model-1", user_id="user-1", status="approved")
        self.assertEqual(approved.lifecycle_status, "approved")

        deprecated = await service.set_status(
            "model-1", user_id="user-1", status="deprecated", notes="retired"
        )
        self.assertEqual(deprecated.lifecycle_status, "deprecated")
        self.assertEqual(deprecated.lifecycle_notes, "retired")

    async def test_invalid_lifecycle_status_rejected(self) -> None:
        service = ModelLifecycleService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=_model())
        with self.assertRaises(HTTPException) as ctx:
            await service.set_status("model-1", user_id="user-1", status="archived")
        self.assertEqual(ctx.exception.status_code, 422)


class DriftServiceEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_distributions_persists_report(self) -> None:
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        created: dict = {}

        async def create_run(run):
            created["run"] = run
            run.id = "drift-run-1"
            return run

        service = DriftService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.repo = MagicMock()
        service.repo.create_run = AsyncMock(side_effect=create_run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        report = await service.compare_distributions(
            "corp-1",
            user_id="user-1",
            baseline={"label_counts": {"yes": 50, "no": 50}},
            current={"label_counts": {"yes": 80, "no": 20}},
            baseline_run_id="run-a",
            current_run_id="run-b",
        )
        self.assertEqual(report["analysis_run_id"], "drift-run-1")
        self.assertIn("summary", report)
        params = json.loads(created["run"].parameters_json)
        self.assertEqual(params.get("baseline_run_id"), "run-a")
        self.assertEqual(params.get("current_run_id"), "run-b")


class DriftRouteRegistrationTests(unittest.TestCase):
    def test_monitoring_drift_route_registered(self) -> None:
        from backend.modules.text_research.api.routes import router

        paths = {getattr(route, "path", "") for route in router.routes}
        self.assertIn("/corpora/{corpus_id}/monitoring/drift", paths)
        methods = {
            method
            for route in router.routes
            if getattr(route, "path", None) == "/corpora/{corpus_id}/monitoring/drift"
            for method in (getattr(route, "methods", None) or set())
        }
        self.assertIn("POST", methods)


if __name__ == "__main__":
    unittest.main()
