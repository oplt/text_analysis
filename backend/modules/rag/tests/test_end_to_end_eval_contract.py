from __future__ import annotations

import unittest

from backend.modules.rag.eval.end_to_end_eval import (
    load_end_to_end_qrels,
    structural_citation_validity,
)


class EndToEndEvalContractTests(unittest.TestCase):
    def test_qrels_are_deterministic_and_cover_required_categories(self):
        first = load_end_to_end_qrels()
        second = load_end_to_end_qrels()
        self.assertEqual(first, second)
        self.assertEqual(len({case.category for case in first}), 7)
        self.assertTrue(any(case.no_answer for case in first))

    def test_qrels_include_expected_chunk_and_source_ids(self):
        cases = {case.id: case for case in load_end_to_end_qrels()}
        self.assertIn("fixture-governance#0", cases["keyword"].expected_chunk_ids)
        self.assertIn("fixture-governance", cases["keyword"].expected_source_ids)
        self.assertIn("fixture-participation#0", cases["contradiction"].expected_chunk_ids)
        self.assertEqual(cases["no-answer"].expected_chunk_ids, ())

    def test_structural_citation_validity_placeholder(self):
        metrics = structural_citation_validity(
            claim_chunk_ids=["a", "escape"],
            retrieved_chunk_ids={"a"},
        )
        self.assertEqual(metrics["citation_structural_validity"], 0.5)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.5)
