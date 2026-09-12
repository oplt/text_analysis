from __future__ import annotations

import asyncio
import unittest
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.rag.application.query_expansion_service import QueryVariant
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.models import RetrievedChunk


def _chunk(**overrides) -> RetrievedChunk:
    base = {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "content": "safe content",
        "score": 0.9,
        "filename": "notes.txt",
        "chunk_index": 0,
        "retrieval_sources": ("dense",),
    }
    base.update(overrides)
    return RetrievedChunk(**base)


def _service_config(**overrides) -> SimpleNamespace:
    base = {
        "enabled": True,
        "top_k": 5,
        "embedding_dimensions": 2,
        "dense_candidates": 5,
        "lexical_candidates": 5,
        "query_variant_concurrency": 3,
        "retrieval_branch_concurrency": 2,
        "rrf_k": 60,
        "parent_context_enabled": False,
        "rerank_enabled": False,
        "rerank_heuristic_enabled": False,
        "rerank_max_depth": 0,
        "source_max_chunks_per_document": 3,
        "evidence_top_k": 5,
        "retrieval_algorithm_version": "hybrid-rrf-v1",
        "index_version": "pgvector-fts-v1",
        "retrieval_cache_artifact_version": "v2",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class RetrievalConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_dense_and_lexical_branches_overlap(self):
        service = RetrievalService(MagicMock())
        service.config = _service_config()
        service.embeddings = SimpleNamespace(embed_texts=AsyncMock(return_value=[[1.0, 0.0]]))

        async def delayed_dense(*args, **kwargs):
            await asyncio.sleep(0.05)
            return []

        async def delayed_lexical(*args, **kwargs):
            await asyncio.sleep(0.05)
            return []

        service.vector_store = SimpleNamespace(similarity_search=delayed_dense)
        service.repo = SimpleNamespace(lexical_search=delayed_lexical)
        with (
            patch(
                "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                AsyncMock(return_value=None),
            ),
            patch(
                "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                AsyncMock(),
            ),
        ):
            started = perf_counter()
            await service.retrieve("query", user_id="user", project_id=None)
        self.assertLess(perf_counter() - started, 0.09)

    async def test_branch_failure_still_degraded_with_evidence(self):
        service = RetrievalService(MagicMock())
        service.config = _service_config()
        service.embeddings = SimpleNamespace(embed_texts=AsyncMock(return_value=[[1.0, 0.0]]))
        service.vector_store = SimpleNamespace(
            similarity_search=AsyncMock(side_effect=RuntimeError("dense boom"))
        )
        service.repo = SimpleNamespace(
            lexical_search=AsyncMock(
                return_value=[
                    _chunk(
                        chunk_id="lex-1",
                        retrieval_sources=("lexical",),
                    )
                ]
            )
        )
        service.ranker = SimpleNamespace(
            name="none", version="0", rerank=lambda q, c, top_n: c[:top_n]
        )

        with (
            patch(
                "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                AsyncMock(return_value=None),
            ),
            patch(
                "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                AsyncMock(),
            ),
        ):
            outcome = await service.retrieve("query", user_id="user", project_id=None)

        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "dense_branch_failed")
        self.assertFalse(outcome.no_matches)
        self.assertEqual([chunk.chunk_id for chunk in outcome.chunks], ["lex-1"])

    async def test_both_branches_fail_marks_both_branches_failed(self):
        service = RetrievalService(MagicMock())
        service.config = _service_config()
        service.embeddings = SimpleNamespace(embed_texts=AsyncMock(return_value=[[1.0, 0.0]]))
        service.vector_store = SimpleNamespace(
            similarity_search=AsyncMock(side_effect=RuntimeError("dense boom"))
        )
        service.repo = SimpleNamespace(
            lexical_search=AsyncMock(side_effect=RuntimeError("lexical boom"))
        )

        with (
            patch(
                "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                AsyncMock(return_value=None),
            ),
            patch(
                "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                AsyncMock(),
            ),
        ):
            outcome = await service.retrieve("query", user_id="user", project_id=None)

        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "both_branches_failed")
        self.assertTrue(outcome.no_matches)
        self.assertEqual(outcome.chunks, [])

    async def test_concurrent_fan_out_preserves_deterministic_ranking_order(self):
        service = RetrievalService(MagicMock())
        service.config = _service_config(query_variant_concurrency=2)
        service.embeddings = SimpleNamespace(embed_texts=AsyncMock(return_value=[[1.0, 0.0]]))
        service.ranker = SimpleNamespace(
            name="none", version="0", rerank=lambda q, c, top_n: c[:top_n]
        )

        async def dense_by_query(query, **kwargs):
            await asyncio.sleep(0.01)
            mapping = {
                "alpha": _chunk(
                    chunk_id="d-alpha",
                    document_id="doc-a",
                    score=0.95,
                    retrieval_sources=("dense",),
                ),
                "beta": _chunk(
                    chunk_id="d-beta",
                    document_id="doc-b",
                    score=0.9,
                    retrieval_sources=("dense",),
                ),
                "gamma": _chunk(
                    chunk_id="d-gamma",
                    document_id="doc-c",
                    score=0.85,
                    retrieval_sources=("dense",),
                ),
            }
            return [mapping[query]]

        async def lexical_by_query(query, **kwargs):
            await asyncio.sleep(0.01)
            # Shared lexical text across variants → executed once via dedupe.
            return [
                _chunk(
                    chunk_id="l-shared",
                    document_id="doc-l",
                    score=0.8,
                    retrieval_sources=("lexical",),
                )
            ]

        service.vector_store = SimpleNamespace(similarity_search=dense_by_query)
        service.repo = SimpleNamespace(lexical_search=AsyncMock(side_effect=lexical_by_query))

        variants = [
            QueryVariant(label="a", text="alpha", dense_text="alpha", lexical_text="shared"),
            QueryVariant(label="b", text="beta", dense_text="beta", lexical_text="shared"),
            QueryVariant(label="c", text="gamma", dense_text="gamma", lexical_text="shared"),
        ]

        async def run_once():
            with (
                patch(
                    "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                    AsyncMock(return_value=None),
                ),
                patch(
                    "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                    AsyncMock(),
                ),
                patch.object(
                    service.query_expansion,
                    "expand_if_needed",
                    return_value=variants,
                ),
                patch(
                    "backend.modules.rag.application.retrieval_service.plan_retrieval",
                    return_value=SimpleNamespace(
                        intent=SimpleNamespace(value="evidence"),
                        top_k=10,
                        dense_candidates=5,
                        lexical_candidates=5,
                        diversify=False,
                        max_per_document=3,
                        multi_query=True,
                        lexical_phrase_boost=False,
                        rerank_depth=0,
                        rerank_method="noop",
                        to_dict=lambda: {"multi_query": True},
                    ),
                ),
            ):
                return await service.retrieve("seed", user_id="user", project_id=None)

        first = await run_once()
        second = await run_once()
        first_ids = [chunk.chunk_id for chunk in first.chunks]
        second_ids = [chunk.chunk_id for chunk in second.chunks]
        self.assertEqual(first_ids, second_ids)
        # Equal RRF ties preserve ranked-list insertion order (variant order).
        self.assertEqual(first_ids, ["d-alpha", "l-shared", "d-beta", "d-gamma"])
        self.assertEqual(service.repo.lexical_search.await_count, 2)
        self.assertEqual(service.embeddings.embed_texts.await_count, 6)

    async def test_shared_dense_text_embeds_once(self):
        service = RetrievalService(MagicMock())
        service.config = _service_config(query_variant_concurrency=3)
        embed = AsyncMock(return_value=[[1.0, 0.0]])
        service.embeddings = SimpleNamespace(embed_texts=embed)
        service.ranker = SimpleNamespace(
            name="none", version="0", rerank=lambda q, c, top_n: c[:top_n]
        )
        service.vector_store = SimpleNamespace(
            similarity_search=AsyncMock(
                return_value=[_chunk(chunk_id="d1", retrieval_sources=("dense",))]
            )
        )
        service.repo = SimpleNamespace(
            lexical_search=AsyncMock(
                return_value=[_chunk(chunk_id="l1", retrieval_sources=("lexical",))]
            )
        )
        variants = [
            QueryVariant(label="a", text="same", dense_text="same", lexical_text="lex-a"),
            QueryVariant(label="b", text="same", dense_text="same", lexical_text="lex-b"),
        ]
        with (
            patch(
                "backend.modules.rag.application.retrieval_service.get_cached_retrieval",
                AsyncMock(return_value=None),
            ),
            patch(
                "backend.modules.rag.application.retrieval_service.set_cached_retrieval",
                AsyncMock(),
            ),
            patch.object(service.query_expansion, "expand_if_needed", return_value=variants),
            patch(
                "backend.modules.rag.application.retrieval_service.plan_retrieval",
                return_value=SimpleNamespace(
                    intent=SimpleNamespace(value="evidence"),
                    top_k=5,
                    dense_candidates=5,
                    lexical_candidates=5,
                    diversify=False,
                    max_per_document=3,
                    multi_query=True,
                    lexical_phrase_boost=False,
                    rerank_depth=0,
                    rerank_method="noop",
                    to_dict=lambda: {"multi_query": True},
                ),
            ),
        ):
            await service.retrieve("seed", user_id="user", project_id=None)

        self.assertEqual(embed.await_count, 1)
        self.assertEqual(service.repo.lexical_search.await_count, 2)
