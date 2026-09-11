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
            raise AssertionError(f"unexpected retrieve scope: {doc_ids}")

        service.retrieval = SimpleNamespace(retrieve=fake_retrieve)

        async def fake_answer(query, *, outcome, **kwargs):
            assert {c.chunk_id for c in outcome.chunks} == {"ca", "cb"}
            assert outcome.fusion_method == "hierarchical_map_reduce"
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
                retrieved_chunk_ids=["ca", "cb"],
                model_name="test",
                latency_ms=1,
                coverage=RetrievalCoverage(
                    documents_in_scope=3,
                    documents_with_retrieved_evidence=2,
                    retrieved_passage_count=2,
                    coverage_ratio=2 / 3,
                ),
                citation_validation_status="valid",
                retrieval_trace_id="t-answer",
            )

        service.answers = SimpleNamespace(answer_from_retrieval=fake_answer)

        scope = SimpleNamespace(
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
        self.assertTrue(result["truncated"])
        self.assertEqual(result["documents_total"], 3)
        self.assertEqual(result["documents_considered"], 2)
        self.assertEqual(result["documents_with_evidence"], 2)
        self.assertEqual(result["answer"], "Synthesized")
        self.assertEqual(len(retrieve_calls), 2)
