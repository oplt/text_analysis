"""Adversarial tests for claim-level citation validation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.application.citation_validation_service import (
    CitationValidationService,
)
from backend.modules.rag.application.rag_answer_service import (
    CITATION_FAILURE_ANSWER,
    RagAnswerService,
)
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk


def _chunk(chunk_id: str = "c1", document_id: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"Evidence for {chunk_id}",
        score=0.9,
        filename="evidence.txt",
        chunk_index=0,
    )


class CitationIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CitationValidationService()
        self.chunks = [_chunk("c1"), _chunk("c2")]

    def validate(self, raw: str):
        return self.validator.validate(
            raw_output=raw,
            retrieved_chunks=self.chunks,
            allowed_document_ids=["d1"],
        )

    def test_substantive_answer_with_empty_claims_fails(self):
        result = self.validate('{"answer":"The project uses PostgreSQL.","claims":[]}')

        self.assertEqual(result.citation_validation_status, "missing_claims")
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.claims, [])

    def test_fabricated_chunk_ids_are_rejected(self):
        result = self.validate(
            '{"answer":"The project uses PostgreSQL.","claims":['
            '{"text":"The project uses PostgreSQL.","chunk_ids":["fabricated"]}]}'
        )

        self.assertEqual(result.citation_validation_status, "invalid")
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.claims, [])
        self.assertTrue(all(not citation.used_in_answer for citation in result.citations))

    def test_mixed_valid_and_invalid_chunk_ids_keep_only_retrieved_evidence(self):
        result = self.validate(
            '{"answer":"The project uses PostgreSQL.","claims":['
            '{"text":"The project uses PostgreSQL.","chunk_ids":["c1","fabricated"]}]}'
        )

        self.assertEqual(result.citation_validation_status, "partial")
        self.assertEqual(result.claims[0].chunk_ids, ["c1"])
        self.assertEqual([c.chunk_id for c in result.citations if c.used_in_answer], ["c1"])

    def test_rendered_answer_is_reconstructed_from_claim_text(self):
        result = self.validate(
            '{"answer":"The project uses PostgreSQL.","claims":['
            '{"text":"The project uses SQLite.","chunk_ids":["c1"]}]}'
        )

        self.assertEqual(result.citation_validation_status, "valid")
        self.assertEqual(result.answer, "The project uses SQLite.")

    def test_malformed_json_fails(self):
        result = self.validate("not json")

        self.assertEqual(result.citation_validation_status, "unstructured")
        self.assertTrue(result.citation_validation_failed)

    def test_empty_claims_without_no_evidence_fails(self):
        result = self.validate('{"answer":"","claims":[]}')

        self.assertEqual(result.citation_validation_status, "missing_claims")
        self.assertTrue(result.citation_validation_failed)

    def test_explicit_no_context_is_valid_without_claims(self):
        result = self.validate(
            '{"answer":"No relevant evidence was found.","claims":[],"no_evidence":true}'
        )

        self.assertEqual(result.citation_validation_status, "no_evidence")
        self.assertFalse(result.citation_validation_failed)
        self.assertTrue(result.no_evidence)

    def test_structurally_incomplete_claim_fails(self):
        result = self.validate(
            '{"answer":"The project uses PostgreSQL.","claims":['
            '{"text":"The project uses PostgreSQL."}]}'
        )

        self.assertEqual(result.citation_validation_status, "incomplete")
        self.assertTrue(result.citation_validation_failed)


class CitationRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_repair_is_bounded_and_returns_safe_answer_with_evidence(self):
        db = AsyncMock()
        service = RagAnswerService(db)
        service.config = SimpleNamespace(enabled=True, max_context_tokens=6000)
        service.memory_config = SimpleNamespace(enabled=False)
        service.repo = MagicMock()
        service.repo.create_query_record = AsyncMock()
        service.generation = MagicMock()
        service.generation.run_rag_answer = AsyncMock(
            side_effect=[
                SimpleNamespace(
                    id="run-1",
                    output_text='{"answer":"Unsupported prose.","claims":[]}',
                    model_name="test-model",
                ),
                SimpleNamespace(
                    id="run-2",
                    output_text='{"answer":"Still unsupported.","claims":[]}',
                    model_name="test-model",
                ),
            ]
        )

        result = await service.answer_from_retrieval(
            "What does the evidence say?",
            outcome=RetrievalOutcome(chunks=[_chunk()]),
            user=SimpleNamespace(id="user-1"),
            project_id=None,
            document_ids=["d1"],
            include_memory=False,
        )

        self.assertEqual(service.generation.run_rag_answer.await_count, 2)
        self.assertEqual(result.answer, CITATION_FAILURE_ANSWER)
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.claims, [])
        self.assertTrue(all(not citation.used_in_answer for citation in result.citations))


if __name__ == "__main__":
    unittest.main()
