"""Invariant tests for document allow-list semantics, fusion, and citations."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.application.document_scope import (
    document_ids_is_empty_allow_list,
    should_apply_document_id_filter,
)
from backend.modules.rag.application.retrieval_fusion import reciprocal_rank_fusion
from backend.modules.rag.application.source_diversifier import (
    diversify_by_document,
    filter_to_allow_list,
)
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.pgvector_adapter import _parse_scope


def _chunk(chunk_id: str, document_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content-{chunk_id}",
        score=score,
        filename=f"{document_id}.txt",
        chunk_index=0,
    )


class DocumentScopeSemanticsTests(unittest.TestCase):
    def test_none_vs_empty_allow_list(self):
        self.assertFalse(document_ids_is_empty_allow_list(None))
        self.assertTrue(document_ids_is_empty_allow_list([]))
        self.assertFalse(document_ids_is_empty_allow_list(["a"]))
        self.assertFalse(should_apply_document_id_filter(None))
        self.assertFalse(should_apply_document_id_filter([]))
        self.assertTrue(should_apply_document_id_filter(["a"]))

    def test_parse_scope_preserves_empty_list(self):
        self.assertEqual(_parse_scope(None), (None, True))
        self.assertEqual(_parse_scope({}), (None, True))
        self.assertEqual(_parse_scope({"document_ids": []}), ([], True))
        self.assertEqual(
            _parse_scope({"document_ids": ["d1"], "owner_scoped": False}),
            (["d1"], False),
        )


class FusionAndDiversifyTests(unittest.TestCase):
    def test_rrf_merges_lists(self):
        dense = [_chunk("c1", "d1", 0.9), _chunk("c2", "d2", 0.8)]
        lexical = [_chunk("c2", "d2", 0.7), _chunk("c3", "d3", 0.6)]
        fused = reciprocal_rank_fusion([dense, lexical], k=60, limit=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual({c.chunk_id for c in fused}, {"c1", "c2", "c3"})

    def test_allow_list_filter(self):
        chunks = [_chunk("c1", "d1"), _chunk("c2", "d2")]
        self.assertEqual(len(filter_to_allow_list(chunks, None)), 2)
        self.assertEqual(filter_to_allow_list(chunks, []), [])
        filtered = filter_to_allow_list(chunks, ["d1"])
        self.assertEqual([c.document_id for c in filtered], ["d1"])

    def test_diversify_caps_per_document(self):
        chunks = [_chunk(f"c{i}", "d1", 1.0 - i * 0.01) for i in range(5)]
        chunks += [_chunk("cx", "d2", 0.5)]
        selected, coverage = diversify_by_document(
            chunks, limit=5, max_per_document=2, documents_in_scope=10
        )
        self.assertLessEqual(sum(1 for c in selected if c.document_id == "d1"), 2)
        self.assertEqual(coverage.documents_in_scope, 10)


class CitationValidationTests(unittest.TestCase):
    def test_rejects_unretrieved_chunk_ids(self):
        chunks = [_chunk("c1", "d1")]
        raw = '{"answer":"A","claims":[{"text":"claim","chunk_ids":["c1","evil"]}]}'
        result = CitationValidationService().validate(
            raw_output=raw,
            retrieved_chunks=chunks,
            allowed_document_ids=["d1"],
        )
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.claims[0].chunk_ids, ["c1"])
        self.assertEqual(result.citations[0].citation_number, 1)
        self.assertTrue(result.citations[0].used_in_answer)

    def test_rejects_cross_corpus_document(self):
        chunks = [_chunk("c1", "other-doc")]
        raw = '{"answer":"A","claims":[{"text":"claim","chunk_ids":["c1"]}]}'
        result = CitationValidationService().validate(
            raw_output=raw,
            retrieved_chunks=chunks,
            allowed_document_ids=["allowed-doc"],
        )
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.claims, [])


class EmptyAllowListRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_empty_document_ids_returns_no_chunks(self):
        from backend.modules.rag.application.retrieval_service import RetrievalService
        from backend.modules.rag.infrastructure.rag_config import RagConfig

        service = RetrievalService(MagicMock(), RagConfig.from_settings())
        service.embeddings = MagicMock()
        service.embeddings.embed_texts = AsyncMock(return_value=[[0.1] * 8])
        service.vector_store = MagicMock()
        service.vector_store.similarity_search = AsyncMock(
            return_value=[_chunk("should-not-appear", "doc")]
        )
        service.repo = MagicMock()
        service.repo.lexical_search = AsyncMock(return_value=[_chunk("also-no", "doc")])

        outcome = await service.retrieve(
            "query",
            user_id="u1",
            project_id="p1",
            filters={"document_ids": [], "owner_scoped": False},
            persist_trace=False,
        )
        self.assertEqual(outcome.chunks, [])
        self.assertTrue(outcome.no_matches)
        service.vector_store.similarity_search.assert_not_awaited()
        service.repo.lexical_search.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
