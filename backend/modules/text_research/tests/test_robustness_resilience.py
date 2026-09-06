"""Unit checks for robustness per-check resilience."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.robustness_service import _safe_fit, _safe_train_eval


class RobustnessResilienceTests(unittest.TestCase):
    def test_safe_fit_empty_split_is_not_evaluable(self):
        result = _safe_fit(
            [],
            [],
            ["unit"],
            [["label"]],
            label_names=["label"],
            config={},
            algorithm="logistic_regression",
            class_weight=None,
            regularization_c=1.0,
            random_seed=1,
        )
        self.assertEqual(result["status"], "not_evaluable")
        self.assertIsNone(result["macro_f1"])
        self.assertIn("Empty", result["reason"])

    def test_safe_train_eval_insufficient_groups_is_not_evaluable(self):
        # Single group cannot form a grouped train/test split.
        result = _safe_train_eval(
            ["one document text", "same group text"],
            [["a"], ["a"]],
            ["doc-1", "doc-1"],
            label_names=["a"],
            config={"lowercase": True, "remove_punctuation": True, "lemmatization": False},
            algorithm="logistic_regression",
            class_weight=None,
            regularization_c=1.0,
            random_seed=1,
            test_size=0.25,
        )
        self.assertEqual(result["status"], "not_evaluable")
        self.assertIsNone(result["macro_f1"])


if __name__ == "__main__":
    unittest.main()
