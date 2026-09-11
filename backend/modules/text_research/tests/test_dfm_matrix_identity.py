"""Full-matrix DFM identity and comparison contracts."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.engine_comparison import compare_engine_results
from backend.modules.text_research.infrastructure.dfm_matrix_identity import (
    compare_dfm_identities,
    inline_sparse_is_complete,
    matrix_checksum_from_coo,
)
from backend.modules.text_research.infrastructure.quantitative import build_dfm_matrix


class DfmMatrixIdentityTests(unittest.TestCase):
    def test_checksum_is_order_independent_for_coo(self) -> None:
        unit_ids = ["u1", "u2"]
        features = ["alpha", "beta"]
        left, n_left = matrix_checksum_from_coo(
            unit_ids=unit_ids,
            feature_names=features,
            triples=[(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)],
        )
        right, n_right = matrix_checksum_from_coo(
            unit_ids=unit_ids,
            feature_names=features,
            triples=[(1, 1, 3.0), (0, 1, 2.0), (0, 0, 1.0)],
        )
        self.assertEqual(left, right)
        self.assertEqual(n_left, 3)
        self.assertEqual(n_right, 3)

    def test_python_dfm_attaches_complete_checksum(self) -> None:
        result = build_dfm_matrix(
            [["alpha", "beta"], ["beta"], []],
            mode="count",
            unit_ids=["u1", "u2", "u3"],
            force_sparse_only=True,
        )
        self.assertTrue(result["matrix_checksum_complete"])
        self.assertEqual(len(result["matrix_checksum"]), 64)
        self.assertEqual(result["matrix_checksum_cells"], result["nnz"])

    def test_preview_only_never_claims_cells_equal(self) -> None:
        comparison = compare_engine_results(
            "dfm",
            {
                "dfm": {
                    "feature_names": ["a"],
                    "unit_ids": ["u1"],
                    "nnz": 250,
                    "sparse": {
                        "row": [0] * 200,
                        "col": [0] * 200,
                        "data": [1.0] * 200,
                        "nnz": 250,
                        "truncated": True,
                        "nnz_exported": 200,
                    },
                }
            },
            {
                "feature_names": ["a"],
                "unit_ids": ["u1"],
                "nnz": 250,
                "sparse_coo": {
                    "rows": [0] * 200,
                    "cols": [0] * 200,
                    "values": [1.0] * 200,
                    "preview_only": True,
                    "preview_limit": 200,
                },
            },
        )
        self.assertFalse(comparison["comparison_complete"])
        self.assertIsNone(comparison["cells_equal"])
        self.assertEqual(comparison["comparison_status"], "inconclusive")
        self.assertFalse(
            inline_sparse_is_complete(
                {
                    "rows": [0] * 200,
                    "values": [1.0] * 200,
                    "preview_only": True,
                },
                declared_nnz=250,
            )
        )

    def test_matching_checksums_prove_full_matrix_equality(self) -> None:
        checksum = "a" * 64
        comparison = compare_dfm_identities(
            {
                "feature_names": ["a", "b"],
                "unit_ids": ["u1"],
                "nnz": 500,
                "matrix_checksum": checksum,
                "matrix_checksum_complete": True,
                "sparse": {"row": [0], "col": [0], "data": [1], "truncated": True},
            },
            {
                "feature_names": ["a", "b"],
                "unit_ids": ["u1"],
                "nnz": 500,
                "matrix_checksum": checksum,
                "matrix_checksum_complete": True,
                "sparse_coo": {"rows": [0], "cols": [0], "values": [1], "preview_only": True},
            },
        )
        self.assertTrue(comparison["comparison_complete"])
        self.assertTrue(comparison["matrix_checksum_equal"])
        self.assertTrue(comparison["cells_equal"])
        self.assertEqual(comparison["comparison_status"], "equal")
        self.assertEqual(comparison["cells_compared"], 500)

    def test_complete_inline_coo_still_compares_cells(self) -> None:
        comparison = compare_engine_results(
            "dfm",
            {
                "dfm": {
                    "dimensions": {"units": 2, "features": 2},
                    "feature_names": ["a", "b"],
                    "unit_ids": ["u1", "u2"],
                    "nnz": 3,
                    "sparse": {"row": [0, 0, 1], "col": [0, 1, 1], "data": [1, 2, 3]},
                }
            },
            {
                "analysis_result": {
                    "results": {
                        "dimensions": {"documents": 2, "features": 2},
                        "feature_names": ["a", "b"],
                        "unit_ids": ["u1", "u2"],
                        "summary": {"nnz": 3},
                        "nnz": 3,
                        "sparse_coo": {
                            "rows": [0, 0, 1],
                            "cols": [0, 1, 1],
                            "values": [1, 2, 3],
                        },
                    }
                }
            },
        )
        self.assertTrue(comparison["comparison_complete"])
        self.assertTrue(comparison["cells_equal"])
        self.assertTrue(comparison["matrix_checksum_equal"])
        self.assertTrue(comparison["units_equal"])
        self.assertTrue(comparison["vocabulary_equal"])


if __name__ == "__main__":
    unittest.main()
