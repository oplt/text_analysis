"""TASK-004: rerun adapters — exact reproduce or explicit unavailability."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.application import run_adapters as adapters
from backend.modules.text_research.application.run_adapters import (
    ADAPTER_VERSION,
    ClassifierTrainingAdapter,
    RobustnessAdapter,
    TopicModelAdapter,
    capability_for_run,
    describe_rerun_capability,
    execute_rerun,
)
from backend.modules.text_research.application.run_service import RunService
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import dumps


class ClassifierRoundTripTests(unittest.TestCase):
    def test_normalize_preserves_full_training_contract(self) -> None:
        params = {
            "snapshot_id": "snap-1",
            "algorithm": "multinomial_nb",
            "task_type": "multiclass",
            "preprocessing_profile_id": "prof-1",
            "vectorizer": "count",
            "use_word_ngrams": False,
            "ngram_min": 1,
            "ngram_max": 2,
            "use_char_ngrams": True,
            "char_ngram_min": 2,
            "char_ngram_max": 4,
            "min_df": 2,
            "max_df": 0.9,
            "max_features": 5000,
            "feature_selection_method": "chi2",
            "feature_selection_k": 100,
            "feature_selection_percentile": 10.0,
            "class_weight": "balanced",
            "regularization_c": 0.5,
            "nb_alpha": 0.25,
            "sgd_loss": "hinge",
            "test_size": 0.15,
            "val_size": 0.1,
            "random_seed": 7,
            "tune_hyperparameters": True,
            "hyperparameter_search_type": "random",
            "hyperparameter_param_grid": {"C": [0.1, 1.0]},
            "hyperparameter_n_iter": 12,
            "hyperparameter_scoring": "f1_weighted",
            "tune_thresholds": False,
            "n_bootstrap": 50,
            "ci_confidence_level": 0.9,
            "calibration_method": "isotonic",
            "validation_strategy": "nested_cv",
            "nested_cv_outer_splits": 4,
            "nested_cv_inner_splits": 2,
            "embedding_provider": "openai",
            "threshold_objective": "utility",
            "threshold_utility_tp": 2.0,
            "threshold_utility_tn": 0.5,
            "threshold_utility_fp": -2.0,
            "threshold_utility_fn": -3.0,
            "name": "exp-a",
            "split_strategy": "grouped_by_source_document",
            "analysis_spec_hash": "ignore-me",
        }
        normalized = ClassifierTrainingAdapter().normalize(params)
        expected_keys = {
            "snapshot_id",
            "algorithm",
            "task_type",
            "preprocessing_profile_id",
            "vectorizer",
            "use_word_ngrams",
            "ngram_min",
            "ngram_max",
            "use_char_ngrams",
            "char_ngram_min",
            "char_ngram_max",
            "min_df",
            "max_df",
            "max_features",
            "feature_selection_method",
            "feature_selection_k",
            "feature_selection_percentile",
            "class_weight",
            "regularization_c",
            "nb_alpha",
            "sgd_loss",
            "test_size",
            "val_size",
            "random_seed",
            "tune_hyperparameters",
            "hyperparameter_search_type",
            "hyperparameter_param_grid",
            "hyperparameter_n_iter",
            "hyperparameter_scoring",
            "tune_thresholds",
            "n_bootstrap",
            "ci_confidence_level",
            "calibration_method",
            "validation_strategy",
            "nested_cv_outer_splits",
            "nested_cv_inner_splits",
            "embedding_provider",
            "threshold_objective",
            "threshold_utility_tp",
            "threshold_utility_tn",
            "threshold_utility_fp",
            "threshold_utility_fn",
            "name",
        }
        self.assertEqual(set(normalized), expected_keys)
        for key in expected_keys:
            self.assertEqual(normalized[key], params[key], msg=key)
        self.assertTrue(describe_rerun_capability("classifier_training", params).rerunnable)

    def test_missing_snapshot_blocks_rerun(self) -> None:
        cap = describe_rerun_capability("classifier_training", {"algorithm": "logistic_regression"})
        self.assertFalse(cap.rerunnable)
        self.assertIn("snapshot_id", cap.block_reason or "")


class TopicRoundTripTests(unittest.TestCase):
    def test_normalize_includes_holdout_and_embedding(self) -> None:
        params = {
            "unit_type": "paragraph",
            "algorithm": "bertopic",
            "n_topics": 12,
            "preprocessing_profile_id": "p1",
            "max_iterations": 40,
            "random_seed": 99,
            "group_by": ["organization"],
            "holdout_fraction": 0.2,
            "holdout_unit_ids": ["u1", "u2"],
            "embedding_provider": "sentence_transformers",
            "embedding_model_name": "all-MiniLM-L6-v2",
            "persist_embedding_artifacts": False,
            "filters": {"language": "en", "organization": "OECD"},
        }
        normalized = TopicModelAdapter().normalize(params)
        self.assertEqual(normalized["holdout_fraction"], 0.2)
        self.assertEqual(normalized["holdout_unit_ids"], ["u1", "u2"])
        self.assertEqual(normalized["embedding_provider"], "sentence_transformers")
        self.assertEqual(normalized["embedding_model_name"], "all-MiniLM-L6-v2")
        self.assertFalse(normalized["persist_embedding_artifacts"])
        self.assertEqual(normalized["filters"], {"language": "en", "organization": "OECD"})
        self.assertTrue(describe_rerun_capability("topic_model", params).rerunnable)


class RobustnessRoundTripTests(unittest.TestCase):
    def test_normalize_includes_group_temporal_transfer(self) -> None:
        params = {
            "snapshot_id": "snap-9",
            "algorithm": "logistic_regression",
            "seeds": [1, 2, 3],
            "cv_folds": 4,
            "class_weights": [None, "balanced"],
            "test_size": 0.3,
            "group_field": "country",
            "max_groups": 10,
            "temporal_field": "year",
            "temporal_windows": True,
            "transfer_field": "region",
            "transfer_train_values": ["north"],
            "transfer_test_values": ["south"],
        }
        normalized = RobustnessAdapter().normalize(params)
        for key, value in params.items():
            self.assertEqual(normalized[key], value, msg=key)
        self.assertTrue(describe_rerun_capability("robustness", params).rerunnable)


class UnsupportedAndSimilarityTests(unittest.TestCase):
    def test_unsupported_types_have_clear_reasons(self) -> None:
        for run_type in (
            AnalysisRunType.CLUSTERING.value,
            AnalysisRunType.DIMENSIONALITY_REDUCTION.value,
            AnalysisRunType.READABILITY.value,
            AnalysisRunType.STATISTICAL_MODEL.value,
            AnalysisRunType.MEASUREMENT_VALIDATION.value,
            AnalysisRunType.DRIFT_MONITORING.value,
        ):
            cap = describe_rerun_capability(run_type, {})
            self.assertFalse(cap.rerunnable, msg=run_type)
            self.assertTrue(cap.block_reason, msg=run_type)
            self.assertEqual(cap.adapter_version, ADAPTER_VERSION)

    def test_embedding_cosine_similarity_not_rerunnable(self) -> None:
        cap = describe_rerun_capability(
            "similarity",
            {
                "unit_type": "document",
                "method": "embedding_cosine",
                "has_embeddings": True,
            },
        )
        self.assertFalse(cap.rerunnable)
        self.assertIn("embedding", (cap.block_reason or "").lower())

    def test_lexical_similarity_is_rerunnable(self) -> None:
        cap = describe_rerun_capability(
            "similarity",
            {"unit_type": "document", "method": "tfidf_cosine", "mode": "pairwise"},
        )
        self.assertTrue(cap.rerunnable)
        self.assertIsNone(cap.block_reason)


class ExecuteRerunTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_classifier_forwards_normalized_kwargs(self) -> None:
        run = SimpleNamespace(
            id="run-1",
            run_type=AnalysisRunType.CLASSIFIER_TRAINING.value,
            corpus_id="c1",
            parameters_json=dumps(
                {
                    "snapshot_id": "snap-1",
                    "feature_selection_method": "chi2",
                    "nb_alpha": 0.3,
                    "tune_hyperparameters": True,
                    "calibration_method": "isotonic",
                    "nested_cv_outer_splits": 4,
                    "embedding_provider": "hashing",
                }
            ),
        )
        mock_train = AsyncMock(return_value=SimpleNamespace(id="new-run"))
        with patch(
            "backend.modules.text_research.application.classification_service.ClassificationService"
        ) as cls_svc:
            cls_svc.return_value.train = mock_train
            result = await execute_rerun(MagicMock(), run, user_id="u1", run_async=True)
        self.assertEqual(result.id, "new-run")
        kwargs = mock_train.await_args.kwargs
        self.assertEqual(kwargs["snapshot_id"], "snap-1")
        self.assertEqual(kwargs["feature_selection_method"], "chi2")
        self.assertEqual(kwargs["nb_alpha"], 0.3)
        self.assertTrue(kwargs["tune_hyperparameters"])
        self.assertEqual(kwargs["calibration_method"], "isotonic")
        self.assertEqual(kwargs["nested_cv_outer_splits"], 4)
        self.assertEqual(kwargs["embedding_provider"], "hashing")
        self.assertTrue(kwargs["run_async"])

    async def test_execute_topic_forwards_holdout_and_filters(self) -> None:
        run = SimpleNamespace(
            id="run-2",
            run_type=AnalysisRunType.TOPIC_MODEL.value,
            corpus_id="c1",
            parameters_json=dumps(
                {
                    "unit_type": "sentence",
                    "holdout_fraction": 0.25,
                    "embedding_provider": "hashing",
                    "persist_embedding_artifacts": False,
                    "filters": {"language": "fr"},
                }
            ),
        )
        mock_train = AsyncMock(return_value=SimpleNamespace(id="topic-run"))
        with patch(
            "backend.modules.text_research.application.topic_model_service.TopicModelService"
        ) as topic_svc:
            topic_svc.return_value.train = mock_train
            await execute_rerun(MagicMock(), run, user_id="u1", run_async=False)
        kwargs = mock_train.await_args.kwargs
        self.assertEqual(kwargs["holdout_fraction"], 0.25)
        self.assertEqual(kwargs["embedding_provider"], "hashing")
        self.assertFalse(kwargs["persist_embedding_artifacts"])
        self.assertEqual(kwargs["language"], "fr")

    async def test_execute_robustness_forwards_transfer_fields(self) -> None:
        run = SimpleNamespace(
            id="run-3",
            run_type=AnalysisRunType.ROBUSTNESS.value,
            corpus_id="c1",
            parameters_json=dumps(
                {
                    "snapshot_id": "snap-2",
                    "group_field": "org",
                    "temporal_windows": True,
                    "transfer_field": "region",
                    "transfer_train_values": ["a"],
                    "transfer_test_values": ["b"],
                }
            ),
        )
        mock_sweep = AsyncMock(return_value=SimpleNamespace(id="rob-run"))
        with patch(
            "backend.modules.text_research.application.robustness_service.RobustnessService"
        ) as rob_svc:
            rob_svc.return_value.run_sweep = mock_sweep
            await execute_rerun(MagicMock(), run, user_id="u1", run_async=True)
        args, kwargs = mock_sweep.await_args
        self.assertEqual(args[0], "snap-2")
        self.assertEqual(kwargs["group_field"], "org")
        self.assertTrue(kwargs["temporal_windows"])
        self.assertEqual(kwargs["transfer_field"], "region")
        self.assertEqual(kwargs["transfer_train_values"], ["a"])
        self.assertEqual(kwargs["transfer_test_values"], ["b"])

    async def test_execute_unsupported_raises_http_400(self) -> None:
        run = SimpleNamespace(
            id="run-4",
            run_type=AnalysisRunType.CLUSTERING.value,
            corpus_id="c1",
            parameters_json=dumps({"unit_type": "document", "n_clusters": 5}),
        )
        with self.assertRaises(HTTPException) as ctx:
            await execute_rerun(MagicMock(), run, user_id="u1")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Clustering", str(ctx.exception.detail))

    async def test_run_service_rerun_delegates_to_registry(self) -> None:
        service = RunService(MagicMock())
        original = SimpleNamespace(id="orig", run_type="frequency_analysis")
        service.get_run_or_404 = AsyncMock(return_value=original)
        with patch(
            "backend.modules.text_research.application.run_service.execute_rerun",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(id="fresh"),
        ) as exec_mock:
            result = await service.rerun("orig", user_id="u1", run_async=True)
        self.assertEqual(result.id, "fresh")
        exec_mock.assert_awaited_once()
        self.assertEqual(exec_mock.await_args.args[1], original)

    def test_capability_for_run_uses_parameters_json(self) -> None:
        run = SimpleNamespace(
            run_type=AnalysisRunType.DRIFT_MONITORING.value,
            parameters_json=dumps({}),
        )
        cap = capability_for_run(run)
        self.assertFalse(cap.rerunnable)


class RegistryCoverageTests(unittest.TestCase):
    def test_every_enum_run_type_is_registered(self) -> None:
        for run_type in AnalysisRunType:
            self.assertIn(run_type.value, adapters.RUN_ADAPTERS)


if __name__ == "__main__":
    unittest.main()
