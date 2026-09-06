"""Tests for annotation assignment planning."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.assignment_planning import (
    assignment_pairs,
    plan_annotation_assignment,
)


class AssignmentPlanningTests(unittest.TestCase):
    def test_overlap_strategy_matches_prompt_example(self):
        units = [f"u{i}" for i in range(300)]
        plan = plan_annotation_assignment(
            units,
            ["A", "B"],
            sample_size=300,
            strategy="overlap",
            overlap_count=100,
        )
        self.assertEqual(len(plan["A"]), 200)
        self.assertEqual(len(plan["B"]), 200)
        shared = set(plan["A"]) & set(plan["B"])
        self.assertEqual(len(shared), 100)
        self.assertEqual(len(set(plan["A"]) | set(plan["B"])), 300)

    def test_shared_strategy_gives_identical_queues(self):
        plan = plan_annotation_assignment(
            ["u1", "u2", "u3"],
            ["A", "B"],
            sample_size=2,
            strategy="shared",
        )
        self.assertEqual(plan["A"], ["u1", "u2"])
        self.assertEqual(plan["B"], ["u1", "u2"])

    def test_disjoint_strategy_has_no_overlap(self):
        plan = plan_annotation_assignment(
            ["u1", "u2", "u3", "u4"],
            ["A", "B"],
            sample_size=4,
            strategy="disjoint",
        )
        self.assertEqual(set(plan["A"]) & set(plan["B"]), set())
        self.assertEqual(len(assignment_pairs(plan)), 4)


if __name__ == "__main__":
    unittest.main()
