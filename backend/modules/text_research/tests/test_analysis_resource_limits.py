"""TASK-010 / LATEST-012: server-side analysis resource bounds reject unsafe payloads."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.modules.text_research.api.schemas import (
    MAX_CLASSIFIER_TRAINING_FITS,
    MAX_CLUSTERS,
    MAX_EMBEDDING_ITEMS,
    MAX_MEASUREMENT_VALUES,
    MAX_MODEL_FEATURES,
    MAX_RESULT_TOP_N,
    MAX_ROBUSTNESS_FITS,
    MAX_ROBUSTNESS_TRANSFER_VALUES,
    MAX_STATISTICAL_ROWS,
    MAX_TOPIC_HOLDOUT_UNITS,
    MAX_TOPIC_SEED_STABILITY_SEEDS,
    MAX_TOPIC_SWEEP_VALUES,
    ClassifierTrainRequest,
    ClusteringRequest,
    FrequencyRequest,
    MeasurementComparisonRequest,
    RobustnessRequest,
    SimilarityRequest,
    StatisticalModelRequest,
    TopicKSweepRequest,
    TopicSeedStabilityRequest,
    TopicTrainRequest,
)


class AnalysisResourceLimitTests(unittest.TestCase):
    def test_top_n_over_ceiling_is_422(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            FrequencyRequest(unit_type="paragraph", top_n=MAX_RESULT_TOP_N + 1)
        self.assertIn("top_n", str(ctx.exception))

    def test_n_clusters_over_ceiling_is_422(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            ClusteringRequest(unit_type="paragraph", n_clusters=MAX_CLUSTERS + 1)
        self.assertIn("n_clusters", str(ctx.exception))

    def test_svd_components_below_clusters_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            ClusteringRequest(
                unit_type="paragraph",
                n_clusters=10,
                use_svd=True,
                n_svd_components=5,
            )
        self.assertIn("n_svd_components", str(ctx.exception))

    def test_misaligned_measurement_arrays_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            MeasurementComparisonRequest(
                source_a="a",
                values_a=[1, 2, 3],
                source_b="b",
                values_b=[1, 2],
            )
        self.assertIn("same length", str(ctx.exception))

    def test_measurement_array_ceiling(self) -> None:
        with self.assertRaises(ValidationError):
            MeasurementComparisonRequest(
                source_a="a",
                values_a=[0] * (MAX_MEASUREMENT_VALUES + 1),
                source_b="b",
                values_b=[1] * (MAX_MEASUREMENT_VALUES + 1),
            )

    def test_statistical_row_ceiling(self) -> None:
        with self.assertRaises(ValidationError):
            StatisticalModelRequest(
                dependent_var="y",
                independent_vars=["x"],
                rows=[{"y": 1, "x": 2}] * (MAX_STATISTICAL_ROWS + 1),
            )

    def test_embedding_payload_item_ceiling(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimilarityRequest(
                unit_type="paragraph",
                method="embedding_cosine",
                embeddings={f"u{i}": [0.1, 0.2] for i in range(MAX_EMBEDDING_ITEMS + 1)},
            )
        self.assertIn("embeddings", str(ctx.exception))

    def test_embedding_dim_mismatch_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimilarityRequest(
                unit_type="paragraph",
                method="embedding_cosine",
                embeddings={"u1": [0.1, 0.2], "u2": [0.1, 0.2, 0.3]},
            )
        self.assertIn("dimensionality", str(ctx.exception))

    def test_topic_sweep_ceiling_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TopicKSweepRequest(
                unit_type="paragraph",
                k_values=[2] * (MAX_TOPIC_SWEEP_VALUES + 1),
            )

    def test_classifier_grid_ceiling_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(
                snapshot_id="snapshot",
                hyperparameter_param_grid={"regularization_c": list(range(300))},
            )

    def test_excessive_classifier_bootstrap_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="snapshot", n_bootstrap=20_001)

    def test_excessive_nested_cv_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(
                snapshot_id="snapshot",
                nested_cv_outer_splits=11,
            )

    def test_excessive_topic_seed_stability_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TopicSeedStabilityRequest(
                unit_type="paragraph",
                seeds=list(range(MAX_TOPIC_SEED_STABILITY_SEEDS + 1)),
            )

    def test_pairwise_embedding_without_top_k_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimilarityRequest(
                unit_type="paragraph",
                method="embedding_cosine",
                mode="pairwise",
                top_k=None,
                embeddings={"u1": [0.1, 0.2]},
            )
        self.assertIn("top_k", str(ctx.exception))

    def test_raw_embedding_async_rejected_on_route_schema(self) -> None:
        from backend.modules.text_research.api.schemas_quantitative import (
            SimilarityRequest as QuantSimilarityRequest,
        )

        with self.assertRaises(ValidationError) as ctx:
            QuantSimilarityRequest(
                unit_type="paragraph",
                method="embedding_cosine",
                embeddings={"u1": [0.1, 0.2]},
                run_async=True,
            )
        self.assertIn("managed embedding artifact", str(ctx.exception).lower())

    def test_classifier_ngram_order_and_relation(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="s", ngram_min=3, ngram_max=2)
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="s", ngram_max=99)

    def test_classifier_min_df_max_df_semantics(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="s", min_df=0.8, max_df=0.2)
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="s", min_df=0)

    def test_classifier_max_features_ceiling(self) -> None:
        with self.assertRaises(ValidationError):
            ClassifierTrainRequest(snapshot_id="s", max_features=MAX_MODEL_FEATURES + 1)

    def test_classifier_split_leaves_training_data(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            ClassifierTrainRequest(snapshot_id="s", test_size=0.9, val_size=0.9)
        self.assertIn("training", str(ctx.exception).lower())

    def test_classifier_nested_fit_budget(self) -> None:
        # 10 * 10 * 100 = 10_000 > MAX_CLASSIFIER_TRAINING_FITS
        with self.assertRaises(ValidationError) as ctx:
            ClassifierTrainRequest(
                snapshot_id="s",
                validation_strategy="nested_grouped_cv",
                nested_cv_outer_splits=10,
                nested_cv_inner_splits=10,
                tune_hyperparameters=True,
                hyperparameter_param_grid={"C": list(range(100))},
            )
        self.assertIn(str(MAX_CLASSIFIER_TRAINING_FITS), str(ctx.exception))

    def test_topic_holdout_unit_ceiling(self) -> None:
        with self.assertRaises(ValidationError):
            TopicTrainRequest(
                unit_type="paragraph",
                holdout_unit_ids=[f"u{i}" for i in range(MAX_TOPIC_HOLDOUT_UNITS + 1)],
            )

    def test_robustness_transfer_value_ceiling(self) -> None:
        with self.assertRaises(ValidationError):
            RobustnessRequest(
                snapshot_id="s",
                transfer_field="region",
                transfer_train_values=[f"a{i}" for i in range(MAX_ROBUSTNESS_TRANSFER_VALUES + 1)],
                transfer_test_values=["b"],
            )

    def test_robustness_fit_budget(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RobustnessRequest(
                snapshot_id="s",
                seeds=list(range(100)),
                class_weights=[None] * 32,
                cv_folds=10,
                max_groups=100,
                temporal_windows=True,
                transfer_field="region",
                transfer_train_values=["a"],
                transfer_test_values=["b"],
            )
        detail = str(ctx.exception).lower()
        self.assertTrue("robustness" in detail or str(MAX_ROBUSTNESS_FITS) in detail)

    def test_robustness_transfer_requires_filters(self) -> None:
        with self.assertRaises(ValidationError):
            RobustnessRequest(snapshot_id="s", transfer_field="region")


if __name__ == "__main__":
    unittest.main()
