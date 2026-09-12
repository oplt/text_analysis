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
from backend.modules.text_research.application.corpus_scope_service import (
    CorpusScopeDocumentBinding,
    CorpusScopeSnapshot,
)
from backend.modules.text_research.application.corpus_synthesis_service import (
    CorpusSynthesisService,
)
from backend.modules.text_research.application.hierarchical_reduce import (
    filter_claims_to_supporting_chunks,
    pack_reduction_nodes,
    write_reduce_checkpoint,
)


class HierarchicalSynthesisTests(unittest.IsolatedAsyncioTestCase):
    def test_reduction_nodes_are_token_bounded_without_reordering(self):
        nodes = [
            {"rag_document_id": "one", "finding": "alpha " * 20},
            {"rag_document_id": "two", "finding": "beta " * 20},
        ]
        batches = pack_reduction_nodes(nodes, max_tokens=300, max_items=10)
        self.assertEqual(batches, [[nodes[0]], [nodes[1]]])

    def test_oversized_reduction_node_fails_before_prompt_construction(self):
        with self.assertRaisesRegex(ValueError, "exceeds the configured token budget"):
            pack_reduction_nodes(
                [{"rag_document_id": "one", "finding": "alpha " * 500}],
                max_tokens=300,
                max_items=10,
            )

    def test_async_synthesis_freezes_and_reconstructs_scope(self):
        service = CorpusSynthesisService(AsyncMock())
        service.rag_config = SimpleNamespace(synthesis_passages_per_document=2)
        scope = CorpusScopeSnapshot(
            corpus_id="corpus-1",
            project_id="project-1",
            corpus_name="Corpus",
            rag_document_ids=["rag-1"],
            corpus_document_ids=["corpus-document-1"],
            indexed_rag_document_ids=["rag-1"],
            unavailable_rag_document_ids=[],
            scope_hash="scope-hash",
            index_version="index-v1",
            retrieval_version="retrieval-v1",
            evidence_revision_hash="evidence-hash",
            total_documents=1,
            indexed_count=1,
            unavailable_count=0,
            document_bindings=[
                CorpusScopeDocumentBinding(
                    "corpus-document-1",
                    "rag-1",
                    "indexed",
                    index_revision_id="revision-1",
                )
            ],
        )

        frozen = service._freeze_scope(scope)
        reconstructed = service._frozen_scope_from_run(
            {
                "scope_hash": "scope-hash",
                "evidence_revision_hash": "evidence-hash",
                "frozen_synthesis_scope": frozen,
            },
            SimpleNamespace(corpus_id="corpus-1", project_id="project-1"),
        )

        self.assertEqual(frozen["retrieval_config"]["synthesis_passages_per_document"], 2)
        self.assertEqual(reconstructed.document_bindings[0].index_revision_id, "revision-1")

    async def test_citations_use_explicit_binding_not_sorted_id_positions(self):
        service = CorpusSynthesisService(AsyncMock())
        service.rag_config = SimpleNamespace(
            synthesis_batch_size=8,
            synthesis_passages_per_document=1,
        )
        cited_chunk = RetrievedChunk(
            chunk_id="chunk-z",
            document_id="rag-z",
            content="Evidence from Z",
            score=0.9,
            filename="z.pdf",
            chunk_index=0,
        )
        service.retrieval = SimpleNamespace(
            retrieve=AsyncMock(
                return_value=RetrievalOutcome(
                    chunks=[cited_chunk],
                    intent=RetrievalIntent.SYNTHESIS,
                )
            )
        )
        service.answers = SimpleNamespace(
            answer_from_retrieval=AsyncMock(
                return_value=RagAnswer(
                    query="question",
                    answer="answer",
                    citations=[
                        Citation(
                            document_id="rag-z",
                            chunk_id="chunk-z",
                            filename="z.pdf",
                            score=0.9,
                            snippet="Evidence from Z",
                        )
                    ],
                    retrieved_chunk_ids=["chunk-z"],
                    model_name="test",
                    latency_ms=1,
                )
            )
        )
        scope = SimpleNamespace(
            corpus_id="corpus-1",
            project_id="project-1",
            scope_hash="scope-hash",
            evidence_revision_hash=None,
            corpus_document_ids=["corpus-a", "corpus-b", "corpus-unavailable"],
            document_bindings=[
                CorpusScopeDocumentBinding(
                    "corpus-b", "rag-a", "indexed", index_revision_id="revision-a"
                ),
                CorpusScopeDocumentBinding("corpus-unavailable", None, "unavailable"),
                CorpusScopeDocumentBinding(
                    "corpus-a", "rag-z", "indexed", index_revision_id="revision-z"
                ),
            ],
            to_dict=lambda: {},
        )

        result = await service._synthesize_sync(
            user=SimpleNamespace(id="user-1"),
            query="question",
            scope=scope,
            allow_list=["rag-a", "rag-z"],
        )

        self.assertEqual(result["citations"][0]["corpus_document_id"], "corpus-a")
        self.assertEqual(
            [
                call.kwargs["filters"]["index_revision_ids"]
                for call in service.retrieval.retrieve.await_args_list
            ],
            [["revision-a"], ["revision-z"]],
        )

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

        reduction_chunk_sets: list[set[str]] = []

        async def fake_answer(query, *, outcome, **kwargs):
            reduction_chunk_sets.append({c.chunk_id for c in outcome.chunks})
            assert outcome.fusion_method == "deterministic_map_reduce"
            chunk = outcome.chunks[0] if outcome.chunks else chunk_a
            return RagAnswer(
                query=query,
                answer="Synthesized",
                citations=[
                    Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        filename=chunk.filename,
                        score=chunk.score,
                        snippet=chunk.content,
                        used_in_answer=True,
                        citation_number=1,
                    )
                ],
                claims=[
                    ClaimCitation(
                        text="Synthesized", chunk_ids=[chunk.chunk_id], citation_numbers=[1]
                    )
                ],
                retrieved_chunk_ids=[chunk.chunk_id],
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
            rag_document_ids=["d1", "d2", "d3"],
            mapping_status="ok",
            document_bindings=[
                SimpleNamespace(
                    rag_document_id="d1",
                    corpus_document_id="cd1",
                    availability="indexed",
                    index_revision_id="rev-1",
                ),
                SimpleNamespace(
                    rag_document_id="d2",
                    corpus_document_id="cd2",
                    availability="indexed",
                    index_revision_id="rev-2",
                ),
                SimpleNamespace(
                    rag_document_id="d3",
                    corpus_document_id="cd3",
                    availability="indexed",
                    index_revision_id="rev-3",
                ),
            ],
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
        self.assertEqual(reduction_chunk_sets, [{"ca", "cb"}, {"cc"}, {"ca", "cc"}])
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
        self.assertTrue(result["reduction_checkpoints"])
        self.assertIn("reduction_checkpoints", result["synthesis_provenance"])

    async def test_claim_escape_is_rejected_and_checkpoints_recorded(self):
        service = CorpusSynthesisService(AsyncMock())
        service.rag_config = SimpleNamespace(
            synthesis_max_documents=8,
            synthesis_batch_size=8,
            synthesis_passages_per_document=1,
        )
        chunk = RetrievedChunk(
            chunk_id="allowed",
            document_id="d1",
            content="Evidence",
            score=0.9,
            filename="a.pdf",
            chunk_index=0,
        )

        async def fake_retrieve(query, **kwargs):
            return RetrievalOutcome(
                chunks=[chunk],
                intent=RetrievalIntent.SYNTHESIS,
                retrieval_trace_id="t1",
            )

        async def fake_answer(query, *, outcome, **kwargs):
            return RagAnswer(
                query=query,
                answer="bad",
                citations=[],
                claims=[
                    ClaimCitation(
                        text="escaped",
                        chunk_ids=["allowed", "not-in-map"],
                        citation_numbers=[1],
                    ),
                    ClaimCitation(
                        text="fully escaped",
                        chunk_ids=["ghost"],
                        citation_numbers=[2],
                    ),
                ],
                retrieved_chunk_ids=["allowed"],
                model_name="test",
                latency_ms=1,
            )

        service.retrieval = SimpleNamespace(retrieve=fake_retrieve)
        service.answers = SimpleNamespace(answer_from_retrieval=fake_answer)
        scope = SimpleNamespace(
            corpus_id="c1",
            project_id="p1",
            scope_hash="scope",
            evidence_revision_hash=None,
            rag_document_ids=["d1"],
            mapping_status="ok",
            document_bindings=[
                SimpleNamespace(
                    rag_document_id="d1",
                    corpus_document_id="cd1",
                    availability="indexed",
                    index_revision_id="rev-1",
                )
            ],
            to_dict=lambda: {},
        )
        result = await service._synthesize_sync(
            user=SimpleNamespace(id="u1"), query="q", scope=scope, allow_list=["d1"]
        )
        self.assertEqual(result["claims"], [{"text": "escaped", "chunk_ids": ["allowed"], "citation_numbers": [1]}])
        self.assertEqual(
            result["synthesis_provenance"]["rejected_claim_escapes"][0]["escaped_chunk_ids"],
            ["not-in-map"],
        )
        self.assertTrue(result["reduction_checkpoints"])
        kept, rejected = filter_claims_to_supporting_chunks(
            [{"text": "x", "chunk_ids": ["ghost"]}], {"allowed"}
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        store = write_reduce_checkpoint({}, level=1, batch=0, payload={"ok": True})
        self.assertIn("level-1/batch-0", store)

    async def test_failed_document_map_is_recorded_as_omitted_evidence(self):
        service = CorpusSynthesisService(AsyncMock())
        service.rag_config = SimpleNamespace(
            synthesis_max_documents=2,
            synthesis_batch_size=2,
            synthesis_passages_per_document=1,
        )

        async def failed_retrieve(query, **kwargs):
            raise RuntimeError("database unavailable")

        service.retrieval = SimpleNamespace(retrieve=failed_retrieve)
        service.answers = SimpleNamespace(
            answer_from_retrieval=AsyncMock(
                return_value=RagAnswer(
                    query="q",
                    answer="none",
                    citations=[],
                    retrieved_chunk_ids=[],
                    model_name="test",
                    latency_ms=1,
                    no_context_found=True,
                )
            )
        )
        scope = SimpleNamespace(
            corpus_id="c1",
            project_id="p1",
            scope_hash="scope",
            evidence_revision_hash=None,
            rag_document_ids=["d1"],
            mapping_status="ok",
            document_bindings=[
                SimpleNamespace(
                    rag_document_id="d1",
                    corpus_document_id="cd1",
                    availability="indexed",
                    index_revision_id="rev-1",
                )
            ],
            to_dict=lambda: {},
        )
        result = await service._synthesize_sync(
            user=SimpleNamespace(id="u1"), query="q", scope=scope, allow_list=["d1"]
        )
        finding = result["document_findings"][0]
        self.assertTrue(finding["map_failure"])
        self.assertEqual(finding["missing_evidence_reason"], "map_retrieval_failed:RuntimeError")
        self.assertEqual(result["omitted_document_ids"], ["d1"])
        self.assertEqual(
            result["synthesis_provenance"]["documents_omitted"],
            [{"rag_document_id": "d1", "reason": "map_retrieval_failed:RuntimeError"}],
        )
