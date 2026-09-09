"""Tests for threshold objectives, abstention, and classifier error analysis."""

from __future__ import annotations

import unittest

import numpy as np

from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.error_analysis import classifier_error_report


class OptimizeThresholdObjectivesTests(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        self.y_proba = np.column_stack([1 - np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.25, 0.85, 0.15]),
                                        np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.25, 0.85, 0.15])])
        self.classes = ["neg", "pos"]

    def test_f1_objective_returns_threshold_and_score(self):
        result = classifiers.optimize_thresholds(
            self.y_true, self.y_proba, "binary", self.classes, objective="f1"
        )
        self.assertEqual(result["objective"], "f1")
        self.assertIn("threshold", result)
        self.assertIn("val_f1", result)
        self.assertIn("val_score", result)

    def test_precision_and_recall_objectives(self):
        for objective in ("precision", "recall"):
            result = classifiers.optimize_thresholds(
                self.y_true, self.y_proba, "binary", self.classes, objective=objective
            )
            self.assertEqual(result["objective"], objective)
            self.assertIn("val_score", result)

    def test_balanced_accuracy_and_youden_j_binary_only(self):
        for objective in ("balanced_accuracy", "youden_j"):
            result = classifiers.optimize_thresholds(
                self.y_true, self.y_proba, "binary", self.classes, objective=objective
            )
            self.assertEqual(result["objective"], objective)
            self.assertIn("threshold", result)

        with self.assertRaises(ValueError):
            classifiers.optimize_thresholds(
                np.array([[1, 0], [0, 1]]),
                np.array([[0.8, 0.2], [0.3, 0.7]]),
                "multilabel",
                ["a", "b"],
                objective="youden_j",
            )

    def test_expected_cost_uses_custom_weights(self):
        result = classifiers.optimize_thresholds(
            self.y_true,
            self.y_proba,
            "binary",
            self.classes,
            objective="expected_cost",
            cost_fp=2.0,
            cost_fn=5.0,
        )
        self.assertEqual(result["cost_fp"], 2.0)
        self.assertEqual(result["cost_fn"], 5.0)
        self.assertIn("threshold", result)

    def test_unknown_objective_raises(self):
        with self.assertRaises(ValueError):
            classifiers.optimize_thresholds(
                self.y_true, self.y_proba, "binary", self.classes, objective="not_real"
            )

    def test_multilabel_per_label_precision_recall(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.3, 0.4]])
        classes = ["a", "b"]
        result = classifiers.optimize_thresholds(
            y_true, y_proba, "multilabel", classes, objective="recall"
        )
        self.assertEqual(set(result["thresholds"]), {"a", "b"})
        self.assertIn("a", result["val_score"])


class AbstentionTests(unittest.TestCase):
    def test_binary_abstention_marks_low_confidence(self):
        y_proba = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
        result = classifiers.apply_abstention(
            y_proba, confidence_threshold=0.6, task_type="binary"
        )
        self.assertEqual(result["abstained"], [False, True, False])
        self.assertEqual(result["predictions"][1], classifiers.ABSTENTION_MARKER)
        self.assertEqual(result["predictions"][2], 1)

    def test_multilabel_abstention_uses_none_marker(self):
        y_proba = np.array([[0.9, 0.1], [0.55, 0.52]])
        result = classifiers.apply_abstention(
            y_proba, confidence_threshold=0.7, task_type="multilabel"
        )
        self.assertIsNone(result["predictions"][1])
        self.assertEqual(result["predictions"][0], [1, 0])

    def test_abstention_summary_reports_scored_metrics(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, classifiers.ABSTENTION_MARKER, 1, 1])
        abstained = np.array([False, True, False, False])
        summary = classifiers.abstention_summary(y_true, y_pred, abstained)
        self.assertEqual(summary["n_abstained"], 1)
        self.assertEqual(summary["n_scored"], 3)
        self.assertIn("scored_accuracy", summary)


class ClassifierErrorReportTests(unittest.TestCase):
    def test_binary_error_report_includes_fp_fn_and_confusion_matrix(self):
        unit_ids = ["u1", "u2", "u3", "u4"]
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]])
        groups = ["doc-a", "doc-a", "doc-b", "doc-b"]
        report = classifier_error_report(
            unit_ids=unit_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            groups=groups,
            label_names=["neg", "pos"],
        )
        self.assertEqual(report["errors"]["false_negatives"], ["u2"])
        self.assertEqual(report["errors"]["false_positives"], ["u4"])
        self.assertIn("confusion_matrix", report)
        self.assertIn("performance_by_document", report)
        self.assertEqual(set(report["performance_by_document"]), {"doc-a", "doc-b"})
        self.assertTrue(report["most_uncertain_cases"])
        self.assertTrue(any(item["unit_id"] == "u4" for item in report["high_confidence_errors"]))

    def test_multilabel_error_report_per_label(self):
        unit_ids = ["u1", "u2"]
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0, 0], [0, 1]])
        report = classifier_error_report(
            unit_ids=unit_ids,
            y_true=y_true,
            y_pred=y_pred,
            label_names=["a", "b"],
        )
        self.assertEqual(report["errors"]["a"]["false_negatives"], ["u1"])
        self.assertEqual(report["errors"]["b"]["false_positives"], [])
        self.assertIn("a", report["performance_by_label"])

    def test_metadata_slices_when_provided(self):
        unit_ids = ["u1", "u2", "u3", "u4"]
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        metadata = {
            "u1": {"organization": "org-a"},
            "u2": {"organization": "org-a"},
            "u3": {"organization": "org-b"},
            "u4": {"organization": "org-b"},
        }
        report = classifier_error_report(
            unit_ids=unit_ids,
            y_true=y_true,
            y_pred=y_pred,
            metadata_by_unit=metadata,
            slice_fields=["organization"],
            label_names=["neg", "pos"],
        )
        self.assertIn("performance_by_metadata_slice", report)
        self.assertIn("org-a", report["performance_by_metadata_slice"]["organization"])


if __name__ == "__main__":
    unittest.main()
