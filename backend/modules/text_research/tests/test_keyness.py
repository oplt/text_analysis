"""Tests for keyness / group comparison statistics."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import keyness, quantitative
from backend.modules.text_research.infrastructure.preprocessing import tokenize


class BenjaminiHochbergTests(unittest.TestCase):
    def test_bh_monotonic_and_bounded(self):
        raw = [0.001, 0.04, 0.03, 0.5]
        adj = keyness.benjamini_hochberg(raw)
        self.assertEqual(len(adj), 4)
        self.assertTrue(all(v is not None and 0 <= v <= 1 for v in adj))
        # Adjusted p for the smallest raw p should be <= that for larger raw p
        # when comparing values at their original indices after BH.
        self.assertLessEqual(adj[0], adj[3])


class KeynessReportTests(unittest.TestCase):
    def setUp(self):
        text_a = "liberty market freedom liberty market liberty"
        text_b = "equality solidarity cohesion equality solidarity equality"
        self.tokenized_a = [tokenize(text_a) for _ in range(8)]
        self.tokenized_b = [tokenize(text_b) for _ in range(8)]

    def test_log_likelihood_direction_and_effect_sizes(self):
        report = keyness.keyness_report(
            self.tokenized_a,
            self.tokenized_b,
            method="log_likelihood",
            top_n=20,
            correction="bh",
            group_field="organization",
            group_a_label="org=A",
            group_b_label="org=B",
        )
        by_feature = {row["feature"]: row for row in report["features"]}
        self.assertIn("liberty", by_feature)
        self.assertEqual(by_feature["liberty"]["effect_direction"], "a")
        self.assertGreater(by_feature["liberty"]["keyness_statistic"], 0)
        self.assertIn("p_value", by_feature["liberty"])
        self.assertIn("p_adjusted", by_feature["liberty"])
        self.assertIn("log_ratio", by_feature["liberty"])
        self.assertIn("odds_ratio", by_feature["liberty"])
        self.assertEqual(report["method"], "log_likelihood")
        self.assertEqual(report["correction"], "bh")
        self.assertEqual(report["group_field"], "organization")

    def test_chi_square_and_fisher_methods(self):
        chi = keyness.keyness_report(
            self.tokenized_a, self.tokenized_b, method="chi_square", top_n=10
        )
        fish = keyness.keyness_report(self.tokenized_a, self.tokenized_b, method="fisher", top_n=10)
        self.assertEqual(chi["method"], "chi_square")
        self.assertEqual(fish["method"], "fisher")
        self.assertTrue(chi["features"])
        self.assertTrue(all("chi_square" in row for row in chi["features"]))
        self.assertTrue(all("fisher_p_value" in row for row in fish["features"]))

    def test_no_hardcoded_categories_in_capabilities(self):
        caps = keyness.describe_keyness_capabilities()
        blob = str(caps).lower()
        self.assertNotIn("cultural_sphere", blob)
        self.assertIn("metadata", caps["group_selection"])

    def test_quantitative_wrapper(self):
        rows = quantitative.keyness(self.tokenized_a, self.tokenized_b, top_n=10)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("effect_direction", rows[0])
        self.assertIn("log_ratio", rows[0])


if __name__ == "__main__":
    unittest.main()
