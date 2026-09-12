from __future__ import annotations

import unittest

from backend.modules.rag.eval.pgvector_benchmark import (
    exact_filtered_topk,
    explain_pgvector_query_plan,
    filtered_ann_vs_exact,
)


class PgvectorQueryPlanTests(unittest.TestCase):
    def test_empty_allow_list_contract_in_memory(self):
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.7, 0.7],
        }
        report = filtered_ann_vs_exact(
            query=[1.0, 0.0],
            vectors=vectors,
            allow_list=set(),
            k=3,
        )
        self.assertTrue(report["empty_allow_list_contract"])
        self.assertEqual(report["exact_ids"], [])
        self.assertEqual(report["ann_ids"], [])
        self.assertEqual(report["recall_at_k"], 1.0)

    def test_filtered_ann_vs_exact_with_allow_list(self):
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.9, 0.1],
        }
        exact = exact_filtered_topk([1.0, 0.0], vectors, allow_list={"a", "c"}, k=2)
        self.assertEqual(exact[0], "a")
        report = filtered_ann_vs_exact(
            query=[1.0, 0.0],
            vectors=vectors,
            allow_list={"a", "c"},
            k=2,
        )
        self.assertEqual(report["recall_at_k"], 1.0)

    def test_explain_placeholder_skips_without_db(self):
        plan = explain_pgvector_query_plan(db_available=False)
        self.assertTrue(plan["skipped"])
        self.assertEqual(plan["status"], "NEEDS_LIVE_MEASUREMENT")
