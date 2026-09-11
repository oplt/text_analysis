"""Unit tests for research-assistant gap closure (P0/P1)."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.application.parent_context_service import expand_parent_chunks
from backend.modules.rag.application.query_expansion_service import QueryExpansionService
from backend.modules.rag.application.reranker_port import (
    NoOpReranker,
    UnicodeHeuristicReranker,
    build_reranker,
)
from backend.modules.rag.application.retrieval_service import RetrievalService, _actual_fusion_method
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.eval.retrieval_eval import run_ci_regression
from backend.modules.rag.infrastructure.rag_config import RagConfig, validate_rag_config
from backend.modules.text_research.application.conversation_context_service import (
    ConversationContextService,
)


def _chunk(
    cid: str,
    doc: str = "d1",
    *,
    sources: tuple[str, ...] = ("dense",),
    content: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=doc,
        content=content or f"content {cid} Türkiye Allemagne français",
        score=0.5,
        filename="f.txt",
        chunk_index=0,
        retrieval_sources=sources,
    )


def test_query_expansion_contradiction_is_bounded():
    variants = QueryExpansionService().expand_contradiction("Policy X is true")
    assert len(variants) == 3
    labels = {v.label for v in variants}
    assert labels == {"neutral", "support", "contradict"}
    assert all("Policy X is true" in v.text or v.text == "Policy X is true" for v in variants)


def test_conversation_context_follow_up_rewrites_and_isolates_thread():
    svc = ConversationContextService(max_context_tokens=500, max_turns=4)
    prior = [
        SimpleNamespace(id="m1", role="user", content="How do papers discuss Germany?"),
        SimpleNamespace(
            id="m2",
            role="assistant",
            content="Paper A emphasizes federalism.",
        ),
    ]
    ctx = svc.build(original_query="What evidence supports that?", prior_messages=prior)
    assert ctx.needs_rewrite
    assert "Germany" in ctx.resolved_retrieval_query or "federalism" in ctx.resolved_retrieval_query
    assert ctx.prior_message_ids == ["m1", "m2"]

    empty = svc.build(original_query="Brand new topic about France", prior_messages=[])
    assert empty.prior_message_ids == []
    assert empty.resolved_retrieval_query == "Brand new topic about France"


def test_citation_unstructured_is_not_validated_success():
    chunks = [_chunk("c1"), _chunk("c2")]
    result = CitationValidationService().validate(
        raw_output="This is prose without JSON claims.",
        retrieved_chunks=chunks,
        allowed_document_ids=["d1"],
    )
    assert result.citation_validation_status == "unstructured"
    assert result.citation_validation_failed is True
    assert result.claims == []
    assert all(not c.used_in_answer for c in result.citations)


def test_citation_rejects_cross_corpus_chunk_ids():
    chunks = [_chunk("c1", "d1"), _chunk("c2", "d2")]
    raw = json.dumps(
        {
            "answer": "Claim text",
            "claims": [{"text": "Claim text", "chunk_ids": ["c1", "c-evil"]}],
        }
    )
    result = CitationValidationService().validate(
        raw_output=raw,
        retrieved_chunks=chunks,
        allowed_document_ids=["d1"],
    )
    assert result.citation_validation_status in {"partial", "invalid"}
    used = [c.chunk_id for c in result.citations if c.used_in_answer]
    assert used == ["c1"]


def test_fusion_method_reflects_what_ran():
    dense = [_chunk("a", sources=("dense",))]
    lexical = [_chunk("b", sources=("lexical",))]
    assert (
        _actual_fusion_method(
            dense_ok=True,
            lexical_ok=True,
            dense=dense,
            lexical=lexical,
            planned_dense=10,
            planned_lexical=10,
        )
        == "rrf"
    )
    assert (
        _actual_fusion_method(
            dense_ok=True,
            lexical_ok=False,
            dense=dense,
            lexical=[],
            planned_dense=10,
            planned_lexical=10,
        )
        == "dense_only"
    )
    assert (
        _actual_fusion_method(
            dense_ok=False,
            lexical_ok=True,
            dense=[],
            lexical=lexical,
            planned_dense=10,
            planned_lexical=10,
        )
        == "lexical_only"
    )


class HybridDegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_dense_failure_still_returns_lexical(self):
        config = RagConfig.from_settings()
        service = RetrievalService(AsyncMock(), config)
        service._dense_branch = AsyncMock(return_value=([], False, "boom"))  # type: ignore[method-assign]
        service._lexical_branch = AsyncMock(  # type: ignore[method-assign]
            return_value=([_chunk("lex1", sources=("lexical",))], True, None)
        )
        service.ranker = NoOpReranker()
        with (
            patch(
                "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                new=AsyncMock(),
            ),
        ):
            outcome = await service.retrieve(
                "query",
                user_id="u1",
                project_id="p1",
                filters={
                    "document_ids": ["d1"],
                    "owner_scoped": False,
                    "retrieval_mode": "hybrid",
                },
                persist_trace=False,
            )
        self.assertTrue(outcome.chunks)
        self.assertEqual(outcome.chunks[0].chunk_id, "lex1")
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.fusion_method, "lexical_only")


class ParentExpandTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_expand_dedupes_shared_parent(self):
        child_a = _chunk("child-a")
        child_b = _chunk("child-b")
        child_a.metadata = {}
        child_b.metadata = {}

        repo = MagicMock()
        repo.get_chunks_by_ids = AsyncMock(
            side_effect=[
                [
                    SimpleNamespace(id="child-a", parent_chunk_id="parent-1", document_id="d1"),
                    SimpleNamespace(id="child-b", parent_chunk_id="parent-1", document_id="d1"),
                ],
                [
                    SimpleNamespace(
                        id="parent-1",
                        document_id="d1",
                        content="PARENT CONTEXT WINDOW",
                        chunk_index=-1,
                    )
                ],
            ]
        )
        expanded = await expand_parent_chunks(
            [child_a, child_b],
            repo=repo,
            document_ids=["d1"],
        )
        parent_contents = [c.content for c in expanded if c.content == "PARENT CONTEXT WINDOW"]
        self.assertEqual(len(parent_contents), 1)
        self.assertTrue(any(c.metadata.get("parent_deduped") for c in expanded))


def test_unicode_reranker_handles_multilingual():
    chunks = [
        _chunk("de", content="Die Verantwortung in Deutschland"),
        _chunk("tr", content="Türkiye’de sorumluluk tartışması"),
        _chunk("fr", content="responsabilité en France"),
    ]
    ranked = UnicodeHeuristicReranker().rerank("responsabilité France", chunks, limit=3)
    assert ranked[0].chunk_id == "fr"
    assert build_reranker(enabled=False).name == "noop"


def test_vector_dimension_mismatch_detected():
    config = RagConfig.from_settings()
    validate_rag_config(config)
    bad = RagConfig(
        enabled=True,
        vector_backend=config.vector_backend,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_dimensions=768,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        max_context_tokens=config.max_context_tokens,
        allowed_file_types=config.allowed_file_types,
        max_file_bytes=config.max_file_bytes,
        expected_vector_dimensions=1536,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        validate_rag_config(bad)


def test_ci_retrieval_regression_fixture():
    snap = run_ci_regression()
    assert snap["activation-framing"] > 0.0
    assert snap["activation-framing_cite_validity"] >= 0.5
