"""Tests for grouped split planning (stratified group holdout + nested CV)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.split_planner import (
    SplitPlanner,
    evaluate_stratified_group_feasibility,
    nested_grouped_cv_indices,
    plan_grouped_splits,
)


class SplitFeasibilityTests(unittest.TestCase):
    def test_infeasible_when_class_in_single_group(self):
        labels = ["a", "a", "b", "b"]
        groups = ["g1", "g1", "g2", "g2"]
        result = evaluate_stratified_group_feasibility(labels, groups)
        self.assertFalse(result.feasible_stratified_group)
        self.assertEqual(result.recommended_strategy, "group_shuffle")
        self.assertIn("appears in only 1 group", result.reason)

    def test_feasible_with_two_groups_per_class(self):
        labels = ["a", "a", "a", "b", "b", "b"]
        groups = ["g1", "g1", "g2", "g3", "g3", "g4"]
        result = evaluate_stratified_group_feasibility(labels, groups)
        self.assertTrue(result.feasible_stratified_group)
        self.assertEqual(result.recommended_strategy, "stratified_group")


class PlanGroupedSplitsTests(unittest.TestCase):
    def setUp(self):
        self.texts = [f"text sample number {i}" for i in range(24)]
        self.labels = ["pos" if i % 2 == 0 else "neg" for i in range(24)]
        self.groups = [f"doc-{i // 3}" for i in range(24)]

    def test_no_group_leakage(self):
        planned = plan_grouped_splits(
            self.labels, self.groups, test_size=0.25, val_size=0.25, random_seed=11
        )
        train_groups = {self.groups[i] for i in planned["train_index"]}
        val_groups = {self.groups[i] for i in planned["val_index"]}
        test_groups = {self.groups[i] for i in planned["test_index"]}
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        total = len(planned["train_index"]) + len(planned["val_index"]) + len(planned["test_index"])
        self.assertEqual(total, 24)

    def test_fallback_when_stratified_infeasible(self):
        labels = ["a", "a", "b", "b"]
        groups = ["g1", "g1", "g2", "g2"]
        planned = plan_grouped_splits(labels, groups, test_size=0.5, val_size=0.0, random_seed=3)
        self.assertEqual(planned["strategy"], "group_shuffle")
        self.assertTrue(any("falling back" in note.lower() for note in planned["notes"]))

    def test_classifiers_integration(self):
        split = classifiers.grouped_train_val_test_split(
            self.texts,
            self.labels,
            self.groups,
            test_size=0.25,
            val_size=0.25,
            random_seed=5,
            prefer_stratified_groups=True,
        )
        self.assertIn("split_strategy", split)
        train_groups = set(split["groups_train"])
        test_groups = set(split["groups_test"])
        self.assertTrue(train_groups.isdisjoint(test_groups))


class NestedGroupedCvTests(unittest.TestCase):
    def test_nested_structure(self):
        labels = ["a", "a", "a", "b", "b", "b", "a", "b"]
        groups = ["g1", "g1", "g2", "g3", "g3", "g4", "g5", "g6"]
        folds = nested_grouped_cv_indices(
            labels, groups, outer_splits=4, inner_splits=2, random_seed=42
        )
        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertIn("outer_train", fold)
            self.assertIn("outer_test", fold)
            self.assertIn("inner_folds", fold)
            outer_train = set(fold["outer_train"])
            outer_test = set(fold["outer_test"])
            self.assertTrue(outer_train.isdisjoint(outer_test))
            for inner in fold["inner_folds"]:
                inner_train = set(inner["train"])
                inner_val = set(inner["val"])
                self.assertTrue(inner_train.isdisjoint(inner_val))
                self.assertTrue(inner_train.issubset(outer_train))
                self.assertTrue(inner_val.issubset(outer_train))

    def test_split_planner_class_aliases(self):
        labels = ["a", "b", "a", "b"]
        groups = ["g1", "g2", "g3", "g4"]
        feasibility = SplitPlanner.evaluate_feasibility(labels, groups)
        self.assertTrue(feasibility.feasible_stratified_group)
        planned = SplitPlanner.plan_grouped_splits(labels, groups, test_size=0.5, val_size=0.0)
        self.assertEqual(
            len(planned["train_index"]) + len(planned["test_index"]),
            len(labels),
        )


class MultilabelGroupedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = [f"unit {i} about topic" for i in range(30)]
        self.labels = [
            ["a"] if i % 3 == 0 else ["b"] if i % 3 == 1 else ["a", "b"] for i in range(30)
        ]
        self.groups = [f"doc-{i // 3}" for i in range(30)]  # 10 docs × 3 units

    def test_multilabel_does_not_crash_and_uses_group_shuffle(self) -> None:
        planned = plan_grouped_splits(
            self.labels,
            self.groups,
            test_size=0.2,
            val_size=0.2,
            random_seed=42,
            task_type="multilabel",
        )
        self.assertEqual(planned["split_strategy"], "group_shuffle_multilabel")
        self.assertTrue(planned["stratification_requested"])
        self.assertFalse(planned["stratification_applied"])
        self.assertIn("not valid for multilabel", planned["reason"])
        self.assertEqual(planned["grouping_variable"], "corpus_document_id")

    def test_multilabel_no_group_leakage(self) -> None:
        planned = plan_grouped_splits(
            self.labels, self.groups, task_type="multilabel", random_seed=7
        )
        train_g = {self.groups[i] for i in planned["train_index"]}
        val_g = {self.groups[i] for i in planned["val_index"]}
        test_g = {self.groups[i] for i in planned["test_index"]}
        self.assertTrue(train_g.isdisjoint(val_g))
        self.assertTrue(train_g.isdisjoint(test_g))
        self.assertTrue(val_g.isdisjoint(test_g))

    def test_multilabel_reproducible_seed(self) -> None:
        a = plan_grouped_splits(self.labels, self.groups, task_type="multilabel", random_seed=99)
        b = plan_grouped_splits(self.labels, self.groups, task_type="multilabel", random_seed=99)
        self.assertEqual(a["train_index"], b["train_index"])
        self.assertEqual(a["test_index"], b["test_index"])

    def test_multilabel_via_classifier_helper(self) -> None:
        split = classifiers.grouped_train_val_test_split(
            self.texts,
            self.labels,
            self.groups,
            test_size=0.2,
            val_size=0.2,
            random_seed=3,
            task_type="multilabel",
        )
        self.assertEqual(split["split_strategy"], "group_shuffle_multilabel")
        self.assertFalse(split["stratification_applied"])
        self.assertTrue(set(split["groups_train"]).isdisjoint(set(split["groups_test"])))
        self.assertTrue(all(isinstance(row, list) for row in split["y_train"]))

    def test_multilabel_few_groups_still_runs(self) -> None:
        labels = [["a"], ["b"], ["a", "b"], ["b"]]
        groups = ["g1", "g2", "g1", "g2"]
        planned = plan_grouped_splits(
            labels, groups, test_size=0.5, val_size=0.0, task_type="multilabel", random_seed=1
        )
        self.assertEqual(planned["split_strategy"], "group_shuffle_multilabel")
        self.assertEqual(
            len(planned["train_index"]) + len(planned["test_index"]),
            len(labels),
        )

    def test_infer_multilabel_shape_without_explicit_task_type(self) -> None:
        feasibility = evaluate_stratified_group_feasibility(self.labels, self.groups)
        self.assertFalse(feasibility.feasible_stratified_group)
        self.assertEqual(feasibility.recommended_strategy, "group_shuffle_multilabel")


if __name__ == "__main__":
    unittest.main()
