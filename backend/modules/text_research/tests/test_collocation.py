"""Tests for collocation / co-occurrence association measures."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import collocation, quantitative
from backend.modules.text_research.infrastructure.preprocessing import tokenize


class CollocationTests(unittest.TestCase):
    def setUp(self):
        text = "universal education policy universal education rights"
        self.tokenized = [tokenize(text) for _ in range(6)]

    def test_methods_exposed_and_attached(self):
        report = collocation.collocation_report(
            self.tokenized, window=3, top_n=20, association_method="pmi"
        )
        self.assertEqual(report["association_method"], "pmi")
        self.assertIn("pmi", report["available_methods"])
        self.assertTrue(report["pairs"])
        row = report["pairs"][0]
        for key in ("count", "pmi", "npmi", "dice", "log_dice", "t_score", "association_score"):
            self.assertIn(key, row)

    def test_rank_by_selected_method(self):
        by_count = collocation.collocation_report(
            self.tokenized, window=3, top_n=5, association_method="count"
        )
        by_npmi = collocation.collocation_report(
            self.tokenized, window=3, top_n=5, association_method="npmi"
        )
        self.assertEqual(by_count["association_method"], "count")
        self.assertEqual(by_npmi["association_method"], "npmi")
        self.assertEqual(by_count["pairs"][0]["association_method"], "count")
        self.assertEqual(by_npmi["pairs"][0]["association_score"], by_npmi["pairs"][0]["npmi"])

    def test_directional_vs_undirected(self):
        undirected = collocation.collocation_report(
            self.tokenized, window=2, directional=False, association_method="count"
        )
        directional = collocation.collocation_report(
            self.tokenized, window=2, directional=True, association_method="count"
        )
        self.assertEqual(undirected["direction"], "undirected")
        self.assertEqual(directional["direction"], "directional")
        # Undirected pairs are sorted alphabetically; directional keeps order.
        undirected_pairs = {(p["term_a"], p["term_b"]) for p in undirected["pairs"]}
        for a, b in undirected_pairs:
            self.assertLessEqual(a, b)

    def test_min_thresholds(self):
        report = collocation.collocation_report(
            self.tokenized,
            window=3,
            association_method="count",
            min_count=1000,
            top_n=10,
        )
        self.assertEqual(report["pairs_returned"], 0)

    def test_quantitative_wrapper_finds_pair(self):
        results = quantitative.cooccurrence(self.tokenized, window=3, top_n=20)
        pairs = {(row["term_a"], row["term_b"]) for row in results}
        self.assertIn(("education", "universal"), pairs)


if __name__ == "__main__":
    unittest.main()
