"""Tests for hierarchical corpus synthesis map/reduce."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import (
    Citation,
    ClaimCitation,
    RagAnswer,
    RetrievalCoverage,
    RetrievalOutcome,
    RetrievedChunk,
)
from backend.modules.text_research.application.corpus_synthesis_service import (
    CorpusSynthesisService,
)


class HierarchicalSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_hierarchical_synthesis_uses_map_findings_not_union(self):
        service = CorpusSynthesisService(AsyncMock())
        service.rag_config = SimpleNamespace(
            enabled=True,
            synthesis_max_documents=2,
            synthesis_batch_size=2,
            synthesis_passages_per_document=2,
        )

        chunk_a = RetrievedChunk(
            chunk_id="ca",
            document_id="d1",
            content="Doc1 says X",
            score=0.9,
            filename="a.pdf",
            chunk_index=0,
        )
        chunk_b = RetrievedChunk(
            chunk_id="cb",
            document_id="d2",
            content="Doc2 says Y",
            score=0.8,
            filename="b.pdf",
            chunk_index=0,
        )
        chunk_c = RetrievedChunk(
            chunk_id="cc",
            document_id="d3",
            content="Doc3 says Z",
            score=0.7,
            filename="c.pdf",
            chunk_index=0,
        )

        retrieve_calls: list[dict] = []

        async def fake_retrieve(query, **kwargs):
            retrieve_calls.append({"query": query, **kwargs})
            filters = kwargs.get("filters") or {}
            doc_ids = filters.get("document_ids") or []
            if doc_ids == ["d1"]:
                return RetrievalOutcome(
                    chunks=[chunk_a],
                    intent=RetrievalIntent.SYNTHESIS,
                    retrieval_trace_id="t1",
                )
            if doc_ids == ["d2"]:
                return RetrievalOutcome(
                    chunks=[chunk_b],
                    intent=RetrievalIntent.SYNTHESIS,
                    retrieval_trace_id="t2",
                )
            if doc_ids == ["d3"]:
                return RetrievalOutcome(
                    chunks=[chunk_c],
                    intent=RetrievalIntent.SYNTHESIS,
                    retrieval_trace_id="t3",
                )
            raise AssertionError(f"unexpected retrieve scope: {doc_ids}")

        service.retrieval = SimpleNamespace(retrieve=fake_retrieve)

        async def fake_answer(query, *, outcome, **kwargs):
            assert {c.chunk_id for c in outcome.chunks} == {"ca", "cb", "cc"}
            assert outcome.fusion_method == "deterministic_map_reduce"
            return RagAnswer(
                query=query,
                answer="Synthesized",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="ca",
                        filename="a.pdf",
                        score=0.9,
                        snippet="Doc1",
                        used_in_answer=True,
                        citation_number=1,
                    )
                ],
                claims=[
                    ClaimCitation(text="Synthesized", chunk_ids=["ca"], citation_numbers=[1])
                ],
                retrieved_chunk_ids=["ca", "cb", "cc"],
                model_name="test",
                latency_ms=1,
                coverage=RetrievalCoverage(
                    documents_in_scope=3,
                    documents_with_retrieved_evidence=3,
                    retrieved_passage_count=3,
                    coverage_ratio=1.0,
                ),
                citation_validation_status="valid",
                retrieval_trace_id="t-answer",
            )

        service.answers = SimpleNamespace(answer_from_retrieval=fake_answer)

        scope = SimpleNamespace(
            corpus_id="c1",
            project_id="p1",
            indexed_count=3,
            scope_hash="abc",
            to_dict=lambda: {"corpus_id": "c1", "indexed_count": 3},
        )
        result = await service._synthesize_sync(
            user=SimpleNamespace(id="u1"),
            query="Compare positions",
            scope=scope,
            allow_list=["d1", "d2", "d3"],
        )
        self.assertFalse(result["truncated"])
        self.assertEqual(result["documents_total"], 3)
        self.assertEqual(result["documents_considered"], 3)
        self.assertEqual(result["documents_with_evidence"], 3)
        self.assertEqual(result["answer"], "Synthesized")
        self.assertEqual(len(retrieve_calls), 3)
        self.assertIsNone(result["retrieval_trace_id"])
        self.assertEqual(result["retrieval_trace_ids"], ["t1", "t2", "t3"])
        self.assertEqual(result["omitted_document_count"], 0)
        self.assertEqual(result["omitted_document_ids"], [])
        self.assertEqual(
            result["synthesis_provenance"]["per_document_retrieval_trace_ids"],
            [
                {"rag_document_id": "d1", "retrieval_trace_id": "t1"},
                {"rag_document_id": "d2", "retrieval_trace_id": "t2"},
                {"rag_document_id": "d3", "retrieval_trace_id": "t3"},
            ],
        )
        self.assertEqual(
            result["synthesis_provenance"]["documents_considered"],
            ["d1", "d2", "d3"],
        )
        self.assertEqual(result["synthesis_provenance"]["documents_omitted"], [])
        self.assertEqual(
            result["synthesis_provenance"]["map_stage"]["operation"],
            "deterministic_evidence_collection",
        )
