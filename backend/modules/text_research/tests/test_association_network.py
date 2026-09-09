"""Tests for graph-ready association network payloads."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import association_network, collocation
from backend.modules.text_research.infrastructure.preprocessing import tokenize


class AssociationNetworkTests(unittest.TestCase):
    def setUp(self):
        text = "universal education policy universal education rights"
        self.tokenized = [tokenize(text) for _ in range(5)]

    def test_build_network_from_pairs(self):
        report = collocation.collocation_report(
            self.tokenized,
            window=3,
            top_n=20,
            association_method="pmi",
            include_network=True,
        )
        network = report["network"]
        self.assertIn("nodes", network)
        self.assertIn("edges", network)
        self.assertGreater(network["node_count"], 0)
        self.assertGreater(network["edge_count"], 0)
        self.assertIsNone(network["interpretation"])
        edge = network["edges"][0]
        self.assertIn("weight", edge)
        self.assertIn("association_statistics", edge)
        self.assertIn("pmi", edge["association_statistics"])
        node = network["nodes"][0]
        self.assertIn("id", node)
        self.assertIn("degree", node)

    def test_include_network_false(self):
        report = collocation.collocation_report(self.tokenized, window=2, include_network=False)
        self.assertNotIn("network", report)

    def test_no_semantic_claims(self):
        caps = association_network.describe_association_network_capabilities()
        self.assertFalse(caps["semantic_interpretation"])

    def test_build_from_raw_pairs(self):
        pairs = [
            {
                "term_a": "alpha",
                "term_b": "beta",
                "count": 3,
                "freq_a": 5,
                "freq_b": 4,
                "association_score": 1.2,
                "pmi": 1.2,
                "npmi": 0.5,
                "dice": 0.4,
                "log_dice": 12.0,
                "t_score": 1.1,
            }
        ]
        network = association_network.build_association_network(pairs, weight_field="count")
        self.assertEqual(network["edge_count"], 1)
        self.assertEqual(network["edges"][0]["weight"], 3.0)
        self.assertEqual({n["id"] for n in network["nodes"]}, {"alpha", "beta"})


if __name__ == "__main__":
    unittest.main()
