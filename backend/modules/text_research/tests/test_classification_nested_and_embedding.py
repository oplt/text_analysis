"""Tests for nested grouped CV, embedding classifiers, and custom utility thresholds."""

from __future__ import annotations

import unittest

import numpy as np

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
