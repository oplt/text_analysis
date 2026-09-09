"""Phase 15 scientific warning builders."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.quantitative import dfm_summary
from backend.modules.text_research.infrastructure.scientific_warnings import (
    assert_no_leakage_soft_warning,
    classifier_scientific_warnings,
    dfm_scientific_warnings,
)


class ScientificWarningTests(unittest.TestCase):
    def test_dfm_summary_includes_sparsity_memory_and_warnings(self) -> None:
        summary = dfm_summary(
            {
                "dimensions": {"units": 8, "features": 4},
                "density": 0.001,
                "nnz": 3,
                "mode": "count",
                "weighting": "count",
                "storage": "sparse",
                "sparse": {"nnz": 3},
            }
        )
        self.assertAlmostEqual(summary["sparsity"], 0.999)
        self.assertIn("estimated_memory_bytes", summary)
        self.assertTrue(summary["scientific_warnings"])
        self.assertTrue(any("Warning:" in w for w in summary["scientific_warnings"]))

    def test_dfm_scientific_warnings_empty_matrix(self) -> None:
        warnings = dfm_scientific_warnings(
            unit_count=0, feature_count=0, nnz=0, density=0.0
        )
        self.assertTrue(any("empty" in w.lower() for w in warnings))

    def test_classifier_warnings_never_claim_leakage(self) -> None:
        warnings = classifier_scientific_warnings(
            feature_space={
                "raw_vocabulary": 100,
                "after_df_pruning": 80,
                "after_supervised_selection": 20,
                "selection_step": "SelectKBest",
            },
            n_train=12,
            n_test=8,
            class_prevalence={"a": 0.9, "b": 0.1},
        )
        assert_no_leakage_soft_warning(warnings)
        joined = " | ".join(warnings)
        self.assertIn("raw=100", joined)
        self.assertIn("training units", joined.lower())

    def test_assert_no_leakage_soft_warning_raises(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_leakage_soft_warning(
                ["Warning: Feature selection was fitted outside the training fold."]
            )

    def test_fit_text_classifier_emits_warnings_without_leakage_claim(self) -> None:
        result = classifiers.fit_text_classifier(
            ["good one", "good two", "bad one", "bad two"] * 3,
            ["pos", "pos", "neg", "neg"] * 3,
            ["good hold", "bad hold"],
            ["pos", "neg"],
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="chi2", k=5),
            tune_thresholds=False,
        )
        warnings = result.get("scientific_warnings") or []
        assert_no_leakage_soft_warning(warnings)
        self.assertIn("feature_space", result)
        self.assertIn("raw_vocabulary", result["feature_space"])


if __name__ == "__main__":
    unittest.main()
