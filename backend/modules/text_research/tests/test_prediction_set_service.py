"""Unit tests for PredictionSet service (mocked repository)."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from backend.modules.text_research.application.prediction_set_service import PredictionSetService
from backend.modules.text_research.domain.enums import ResearchArtifactKind
from backend.modules.text_research.domain.models import (
    AnalysisRun,
    ModelPrediction,
    PredictionSet,
    TrainedModel,
    dumps,
    loads,
)


class PredictionSetModelTests(unittest.TestCase):
    def test_prediction_set_table_has_required_columns(self) -> None:
        column_names = {column.key for column in PredictionSet.__table__.columns}
        for name in (
            "project_id",
            "corpus_id",
            "trained_model_id",
            "model_version",
            "dataset_snapshot_id",
            "analysis_run_id",
            "metadata_json",
        ):
            self.assertIn(name, column_names)

    def test_research_artifact_kind_constant(self) -> None:
        self.assertEqual(ResearchArtifactKind.PREDICTION_SET.value, "prediction_set")


class PredictionSetServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = MagicMock()
        self.service = PredictionSetService(self.db)
        self.service.repo = AsyncMock()
        self.service.ensure_project_access = AsyncMock()

        self.run = AnalysisRun(
            id="run-1",
            project_id="project-1",
            corpus_id="corpus-1",
            run_type="classifier_prediction",
            status="completed",
            created_by="user-1",
        )
        self.model = TrainedModel(
            id="model-1",
            project_id="project-1",
            corpus_id="corpus-1",
            analysis_run_id="train-run-1",
            training_dataset_snapshot_id="snapshot-1",
            model_family="logistic_regression",
            task_type="binary",
            label_ids_json=dumps(["no", "yes"]),
            feature_config_json=dumps({}),
            training_config_json=dumps({}),
            metrics_json=dumps({}),
            model_artifact_path="/tmp/model.joblib",
            vectorizer_artifact_path="/tmp/vectorizer.joblib",
            version=2,
            created_by="user-1",
        )

    async def test_create_from_run_persists_header_without_touching_annotations(self) -> None:
        created = PredictionSet(
            id="ps-1",
            project_id="project-1",
            corpus_id="corpus-1",
            trained_model_id="model-1",
            model_version=2,
            dataset_snapshot_id="snapshot-1",
            analysis_run_id="run-1",
            created_by="user-1",
            created_at=datetime.now(UTC),
            metadata_json=dumps(
                {
                    "kind": ResearchArtifactKind.PREDICTION_SET.value,
                    "unit_ids": ["u-1", "u-2"],
                    "unit_count": 2,
                }
            ),
        )
        self.service.repo.create_prediction_set.return_value = created

        result = await self.service.create_from_run(
            run=self.run,
            model=self.model,
            unit_ids=["u-1", "u-2"],
            created_by="user-1",
        )

        self.service.repo.create_prediction_set.assert_awaited_once()
        saved = self.service.repo.create_prediction_set.await_args.args[0]
        metadata = loads(saved.metadata_json, {})
        self.assertEqual(result.id, "ps-1")
        self.assertEqual(saved.trained_model_id, "model-1")
        self.assertEqual(saved.model_version, 2)
        self.assertEqual(metadata["kind"], ResearchArtifactKind.PREDICTION_SET.value)
        self.assertEqual(metadata["unit_ids"], ["u-1", "u-2"])

    async def test_get_returns_predictions_for_set_units(self) -> None:
        prediction_set = PredictionSet(
            id="ps-1",
            project_id="project-1",
            corpus_id="corpus-1",
            trained_model_id="model-1",
            model_version=2,
            dataset_snapshot_id="snapshot-1",
            analysis_run_id="run-1",
            created_by="user-1",
            created_at=datetime.now(UTC),
            metadata_json=dumps(
                {
                    "kind": ResearchArtifactKind.PREDICTION_SET.value,
                    "unit_ids": ["u-1"],
                    "unit_count": 1,
                }
            ),
        )
        prediction = ModelPrediction(
            id="pred-1",
            trained_model_id="model-1",
            text_unit_id="u-1",
            predicted_labels_json=dumps(["yes"]),
            scores_json=dumps({"yes": 0.88}),
            uncertainty=0.12,
            created_at=datetime.now(UTC),
        )
        self.service.repo.get_prediction_set.return_value = prediction_set
        self.service.repo.list_predictions_for_units.return_value = [prediction]

        payload = await self.service.get("ps-1", user_id="user-1")

        self.service.ensure_project_access.assert_awaited_once_with(
            user_id="user-1",
            project_id="project-1",
        )
        self.service.repo.list_predictions_for_units.assert_awaited_once_with(
            "model-1",
            ["u-1"],
        )
        self.assertEqual(payload["prediction_set"].id, "ps-1")
        self.assertEqual(payload["predictions"][0].text_unit_id, "u-1")

    async def test_list_checks_corpus_access(self) -> None:
        corpus = MagicMock(project_id="project-1")
        self.service.repo.get_corpus.return_value = corpus
        self.service.repo.list_prediction_sets_for_corpus.return_value = ([], 0)

        items, total = await self.service.list("corpus-1", user_id="user-1")

        self.service.ensure_project_access.assert_awaited_once()
        self.assertEqual(items, [])
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
