from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from backend.modules.rag.domain.models import RetrievalOutcome
from backend.modules.rag.eval.end_to_end_eval import (
    load_end_to_end_qrels,
    run_end_to_end_benchmark,
    structural_citation_validity,
)
from backend.modules.rag.eval.fixture_ingestion import FixtureIndex


class EndToEndEvalContractTests(unittest.TestCase):
    def test_fixture_binding_changes_runtime_ids_without_changing_gold_evidence(self):
        case = load_end_to_end_qrels()[0]
        index = FixtureIndex(
            {"fixture-governance": "runtime-document"},
            {"fixture-governance#0": "runtime-chunk"},
            ("revision",),
            {},
        )
        bound = index.bind([case])[0]
        self.assertEqual(bound.expected_chunk_ids, ("runtime-chunk",))
        self.assertEqual(bound.expected_document_ids, ("runtime-document",))
        self.assertEqual(bound.expected_source_ids, case.expected_source_ids)
        self.assertEqual(case.expected_chunk_ids, ("fixture-governance#0",))
        with self.assertRaises(KeyError):
            FixtureIndex({}, {}, (), {}).bind([case])

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


class CitationBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_gold_chunk_not_retrieved_is_unsupported(self):
        case = load_end_to_end_qrels()[0]
        retrieval = AsyncMock()
        retrieval.retrieve.return_value = RetrievalOutcome(chunks=[], no_matches=True)
        report = await run_end_to_end_benchmark(
            retrieval,
            user_id="user",
            project_id=None,
            document_ids=list(case.expected_document_ids),
            cases=[case],
            claim_chunk_ids_by_case={case.id: list(case.expected_chunk_ids)},
        )
        self.assertEqual(report["cases"][0]["citation_structural_validity"], 0.0)
        self.assertEqual(report["cases"][0]["unsupported_claim_rate"], 1.0)

    async def test_absent_generation_is_not_scored_as_perfect_citations(self):
        retrieval = AsyncMock()
        retrieval.retrieve.return_value = RetrievalOutcome(chunks=[], no_matches=True)
        report = await run_end_to_end_benchmark(
            retrieval,
            user_id="user",
            project_id=None,
            document_ids=[],
        )
        self.assertTrue(
            all(case["citation_structural_validity"] is None for case in report["cases"])
        )
