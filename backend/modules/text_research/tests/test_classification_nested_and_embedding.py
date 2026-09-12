"""Tests for nested grouped CV, embedding classifiers, and custom utility thresholds."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from backend.modules.text_research.application.classification_service import (
    ClassificationService,
    build_fit_kwargs,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps
from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.transformer_classifier import (
    fit_transformer_classifier,
)


class NestedGroupedCVTests(unittest.TestCase):
    def test_nested_grouped_cv_returns_outer_folds(self):
        texts = [f"document {index} about topic" for index in range(12)]
        labels = [0, 1] * 6
        groups = [f"doc-{index // 2}" for index in range(12)]
        result = classifiers.nested_grouped_cv_evaluation(
            texts,
            labels,
            groups,
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(max_features=100),
            preprocessing_config={"language": "en"},
            label_names=None,
            class_weight=None,
            C=1.0,
            random_seed=1,
            outer_splits=3,
            inner_splits=2,
        )
        self.assertEqual(result["strategy"], "nested_grouped_cv")
        self.assertGreaterEqual(len(result["folds"]), 2)
        self.assertIn("aggregate_metrics", result)


class EmbeddingClassifierTests(unittest.TestCase):
    def test_embedding_logistic_resolves_and_fits(self):
        texts = ["alpha beta", "gamma delta", "alpha gamma", "beta delta"] * 3
        labels = [0, 1] * 6
        groups = [index // 2 for index in range(len(texts))]
        split = classifiers.grouped_train_val_test_split(
            texts, labels, groups, test_size=0.25, val_size=0.25, random_seed=1
        )
        result = classifiers.fit_embedding_text_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            "binary",
            algorithm="embedding_logistic",
            embedding_provider="hashing",
            X_val_texts=split.get("X_val") or None,
            y_val=split.get("y_val") or None,
            groups_test=split["groups_test"],
            tune_thresholds=False,
        )
        self.assertIn("metrics", result)
        self.assertIn("model", result)
        self.assertGreater(result["n_test"], 0)


class ClassificationFitKwargsTests(unittest.TestCase):
    def test_sparse_and_embedding_fit_calls_receive_task_type_once(self):
        params = {
            "class_weight": None,
            "regularization_c": 1.0,
            "random_seed": 42,
        }
        split = {"X_val": [], "y_val": [], "groups_test": ["doc-1"]}
        kwargs = build_fit_kwargs(
            params, label_names=["negative", "positive"], task_type="binary", split=split
        )
        self.assertNotIn("task_type", kwargs)

        sparse_fit = MagicMock()
        sparse_fit([], [], [], [], task_type="binary", **kwargs)
        self.assertEqual(sparse_fit.call_args.kwargs["task_type"], "binary")

        embedding_fit = MagicMock()
        embedding_fit([], [], [], [], "binary", **kwargs)
        self.assertEqual(embedding_fit.call_args.args[4], "binary")
        self.assertNotIn("task_type", embedding_fit.call_args.kwargs)


class ExecuteTrainingSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def _execute(self, algorithm: str) -> None:
        unit_ids = [f"u-{index}" for index in range(16)]
        labels = {
            unit_id: ["negative" if index % 2 == 0 else "positive"]
            for index, unit_id in enumerate(unit_ids)
        }
        snapshot = SimpleNamespace(
            id="snapshot-1",
            unit_ids_json=dumps(unit_ids),
            class_distribution_json=dumps(
                {"distribution": {"negative": 8, "positive": 8}, "unit_labels": labels}
            ),
            annotation_campaign_id=None,
            annotation_campaign_snapshot_hash=None,
            adjudication_policy=None,
            gold_source=None,
        )
        params = {
            "snapshot_id": snapshot.id,
            "algorithm": algorithm,
            "task_type": "binary",
            "min_df": 1,
            "max_df": 1.0,
            "max_features": None,
            "test_size": 0.25,
            "val_size": 0.25,
            "random_seed": 7,
            "ngram_max": 1,
            "class_weight": None,
            "regularization_c": 1.0,
            "n_bootstrap": 0,
            "tune_thresholds": False,
            "calibration_method": "none",
            "embedding_provider": "hashing",
        }
        run = SimpleNamespace(
            id="run-1",
            project_id="project-1",
            corpus_id="corpus-1",
            created_by="user-1",
            status=AnalysisRunStatus.QUEUED.value,
            parameters_json=dumps(params),
            artifact_namespace="runs/project-1/run-1",
        )
        service = ClassificationService(MagicMock())
        service.db = MagicMock(commit=AsyncMock())
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run = AsyncMock(return_value=run)
        service.repo.get_snapshot = AsyncMock(return_value=snapshot)
        service.repo.list_text_units_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=unit_id,
                    text=(
                        "market freedom individual choice"
                        if index % 2 == 0
                        else "equality solidarity universal welfare"
                    ),
                    corpus_document_id=f"doc-{index}",
                )
                for index, unit_id in enumerate(unit_ids)
            ]
        )
        service.repo.list_documents_by_ids = AsyncMock(return_value=[])
        service.repo.next_model_version = AsyncMock(return_value=1)
        service.repo.create_model = AsyncMock(return_value=SimpleNamespace(id="model-1"))

        with (
            patch(
                "backend.modules.text_research.application.classification_service.model_storage.save_artifact_with_metadata",
                side_effect=[
                    ("model.joblib", {"sha256": "model-checksum"}),
                    ("vectorizer.joblib", {"sha256": "vectorizer-checksum"}),
                ],
            ),
            patch(
                "backend.modules.text_research.application.run_lifecycle.complete_if_active",
                new=AsyncMock(
                    side_effect=lambda *_args, **_kwargs: (
                        setattr(run, "status", AnalysisRunStatus.COMPLETED.value) or run
                    )
                ),
            ),
        ):
            result = await service.execute_training(run.id)

        self.assertEqual(result.status, AnalysisRunStatus.COMPLETED.value)
        self.assertTrue(service.repo.create_model.awaited)

    async def test_execute_training_real_sparse_fit_completes(self) -> None:
        await self._execute("logistic_regression")

    async def test_execute_training_real_hashing_embedding_fit_completes(self) -> None:
        await self._execute("embedding_logistic")


class CustomUtilityThresholdTests(unittest.TestCase):
    def test_custom_utility_objective(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.25]),
                np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.25]),
            ]
        )
        result = classifiers.optimize_thresholds(
            y_true,
            y_proba,
            "binary",
            ["neg", "pos"],
            objective="custom_utility",
            utility_tp=2.0,
            utility_tn=1.0,
            utility_fp=-3.0,
            utility_fn=-4.0,
        )
        self.assertEqual(result["objective"], "custom_utility")
        self.assertEqual(result["utility_tp"], 2.0)
        self.assertIn("threshold", result)


class TransformerClassifierTests(unittest.TestCase):
    def test_transformer_raises_honestly_without_deps(self):
        with self.assertRaises(ValueError) as ctx:
            fit_transformer_classifier(["hello"], ["a"])
        message = str(ctx.exception).lower()
        self.assertIn("transformer", message)


if __name__ == "__main__":
    unittest.main()
