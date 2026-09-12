from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.modules.rag.application.rerank_policy import apply_rerank_policy
from backend.modules.rag.application.retrieval_planner import plan_retrieval
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievedChunk


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "doc", chunk_id, score, "doc.txt", 0)


class _FailingReranker:
    name = "failing"
    version = "v1"

    def rerank(self, query, chunks, *, limit):
        raise RuntimeError("unavailable")


class RerankPolicyTests(unittest.TestCase):
    def test_fact_profile_skips_reranking(self):
        profile = plan_retrieval(
            RetrievalIntent.FACT,
            SimpleNamespace(top_k=5, rerank_enabled=True, rerank_max_depth=40),
        )
        self.assertEqual(profile.rerank_depth, 0)

    def test_evidence_profile_caps_reranking_depth(self):
        profile = plan_retrieval(
            RetrievalIntent.EVIDENCE,
            SimpleNamespace(top_k=5, rerank_enabled=True, rerank_max_depth=20),
        )
        self.assertEqual(profile.rerank_depth, 20)

    def test_reranker_failure_keeps_original_ranking(self):
        chunks = [_chunk("first", 0.9), _chunk("second", 0.8)]
        result = apply_rerank_policy(_FailingReranker(), "question", chunks, depth=20)
        self.assertTrue(result.failed)
        self.assertEqual(result.chunks, chunks)
