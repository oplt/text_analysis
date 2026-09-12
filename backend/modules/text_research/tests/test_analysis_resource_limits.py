"""TASK-010: server-side analysis resource bounds reject unsafe payloads with 422."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.modules.text_research.api.schemas import (
    MAX_CLUSTERS,
    MAX_EMBEDDING_ITEMS,
    MAX_MEASUREMENT_VALUES,
    MAX_RESULT_TOP_N,
    MAX_STATISTICAL_ROWS,
    ClusteringRequest,
    FrequencyRequest,
    MeasurementComparisonRequest,
    SimilarityRequest,
    StatisticalModelRequest,
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


if __name__ == "__main__":
    unittest.main()
