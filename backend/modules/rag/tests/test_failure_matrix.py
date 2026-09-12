"""Deterministic RAG failure-injection matrix (RAG-P1-021).

Each case asserts an explicit degraded/error/no-match outcome. None of these
modes may masquerade as a normal successful answer with invented evidence.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.identity_access.models import User
from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.application.evidence_revision import StaleEvidenceRevisionError
from backend.modules.rag.application.rag_answer_service import (
    CITATION_FAILURE_ANSWER,
    NO_CONTEXT_ANSWER,
    RagAnswerService,
)
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.domain.models import RetrievalOutcome, RetrievedChunk
from backend.modules.rag.infrastructure.repositories import RagRepository


def _chunk(**overrides) -> RetrievedChunk:
    base = {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "content": "safe evidence about governance",
        "score": 0.9,
        "filename": "notes.txt",
        "chunk_index": 0,
        "index_revision_id": "rev-live",
        "retrieval_sources": ("dense",),
    }
    base.update(overrides)
    return RetrievedChunk(**base)


def _retrieval_config(**overrides) -> SimpleNamespace:
    base = {
        "enabled": True,
        "top_k": 5,
        "evidence_top_k": 5,
        "embedding_dimensions": 2,
        "dense_candidates": 5,
        "lexical_candidates": 5,
        "source_max_chunks_per_document": 3,
        "retrieval_algorithm_version": "hybrid-rrf-v1",
        "index_version": "pgvector-fts-v1",
        "rrf_k": 60,
        "parent_context_enabled": False,
        "retrieval_cache_artifact_version": "v2",
        "rerank_enabled": False,
        "rerank_max_depth": 0,
        "rerank_heuristic_enabled": False,
        "fusion_method": "rrf",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FailingReranker:
    name = "failing"
    version = "v1"

    def rerank(self, query, chunks, *, limit):
        raise TimeoutError("reranker timeout")


class FailureMatrixRetrievalTests(unittest.IsolatedAsyncioTestCase):
    """Branch and scope failures must surface as degraded / no_matches / raise."""

    async def _retrieve(
        self,
        *,
        dense=None,
        lexical=None,
        embed_side_effect=None,
        config=None,
        filters=None,
        evidence_revision_hash: str | None = None,
        intent: RetrievalIntent | str | None = None,
        ranker=None,
    ) -> RetrievalOutcome:
        db = MagicMock()
        service = RetrievalService(db)
        service.config = config or _retrieval_config()
        service.embeddings = MagicMock()
        if embed_side_effect is not None:
            service.embeddings.embed_texts = AsyncMock(side_effect=embed_side_effect)
        else:
            service.embeddings.embed_texts = AsyncMock(return_value=[[1.0, 0.0]])
        service.vector_store = MagicMock()
        if isinstance(dense, Exception):
            service.vector_store.similarity_search = AsyncMock(side_effect=dense)
        else:
            service.vector_store.similarity_search = AsyncMock(return_value=dense or [])
        service.repo = MagicMock()
        if isinstance(lexical, Exception):
            service.repo.lexical_search = AsyncMock(side_effect=lexical)
        else:
            service.repo.lexical_search = AsyncMock(return_value=lexical or [])
        service.ranker = ranker or SimpleNamespace(
            name="none", version="0", rerank=lambda q, c, top_n=None, limit=None: c
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
            return await service.retrieve(
                "failure matrix query",
                user_id="user-1",
                project_id="proj-1",
                filters=filters,
                intent=intent,
                evidence_revision_hash=evidence_revision_hash,
                persist_trace=False,
            )

    async def test_embedding_timeout_degrades_dense_branch(self):
        outcome = await self._retrieve(
            embed_side_effect=TimeoutError("embedding provider timeout"),
            lexical=[_chunk(retrieval_sources=("lexical",))],
        )
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "dense_branch_failed")
        self.assertFalse(outcome.no_matches)
        self.assertEqual(len(outcome.chunks), 1)

    async def test_vector_db_failure_degrades_dense_branch(self):
        outcome = await self._retrieve(
            dense=RuntimeError("pgvector connection refused"),
            lexical=[_chunk(retrieval_sources=("lexical",))],
        )
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "dense_branch_failed")
        self.assertNotEqual(outcome.fusion_method, "rrf")

    async def test_fts_failure_degrades_lexical_branch(self):
        outcome = await self._retrieve(
            dense=[_chunk()],
            lexical=RuntimeError("fts tsquery failed"),
        )
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "lexical_branch_failed")
        self.assertFalse(outcome.no_matches)

    async def test_both_branches_fail_is_degraded_empty(self):
        outcome = await self._retrieve(
            dense=RuntimeError("vector boom"),
            lexical=RuntimeError("fts boom"),
        )
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.degradation_reason, "both_branches_failed")
        self.assertTrue(outcome.no_matches)
        self.assertEqual(outcome.chunks, [])

    async def test_empty_retrieval_is_no_matches_not_success(self):
        outcome = await self._retrieve(dense=[], lexical=[])
        self.assertFalse(outcome.degraded)
        self.assertTrue(outcome.no_matches)
        self.assertEqual(outcome.chunks, [])
        self.assertIn(outcome.fusion_method, {"none", "dense_only", "lexical_only"})

    async def test_reranker_failure_marks_degraded(self):
        outcome = await self._retrieve(
            dense=[_chunk(chunk_id="a"), _chunk(chunk_id="b", score=0.8)],
            lexical=[],
            config=_retrieval_config(rerank_enabled=True, rerank_max_depth=20),
            intent=RetrievalIntent.EVIDENCE,
            ranker=_FailingReranker(),
        )
        self.assertTrue(outcome.degraded)
        self.assertIsNotNone(outcome.degradation_reason)
        self.assertIn("reranker_failed", outcome.degradation_reason)
        self.assertGreaterEqual(len(outcome.chunks), 1)

    async def test_stale_revision_raises_not_success(self):
        with self.assertRaises(StaleEvidenceRevisionError) as ctx:
            await self._retrieve(
                dense=[_chunk(index_revision_id="live-revision")],
                lexical=[],
                filters={
                    "document_ids": ["doc-1"],
                    "owner_scoped": False,
                    "index_revision_ids": ["frozen-revision"],
                },
                evidence_revision_hash="frozen-hash",
            )
        self.assertEqual(ctx.exception.code, "stale_evidence_revision")

    async def test_missing_revision_on_chunk_raises(self):
        with self.assertRaises(StaleEvidenceRevisionError):
            await self._retrieve(
                dense=[_chunk(index_revision_id=None)],
                lexical=[],
                filters={
                    "document_ids": ["doc-1"],
                    "owner_scoped": False,
                    "index_revision_ids": ["frozen-revision"],
                },
                evidence_revision_hash="frozen-hash",
            )

    async def test_deleted_document_scope_yields_no_matches(self):
        """Soft-deleted docs are filtered in SQL; faked empty branches must not invent hits."""
        outcome = await self._retrieve(
            dense=[],
            lexical=[],
            filters={"document_ids": ["deleted-doc"], "owner_scoped": False},
        )
        self.assertTrue(outcome.no_matches)
        self.assertEqual(outcome.chunks, [])
        # Contract: repository scope always excludes deleted_at rows.
        params: dict = {}
        filters = RagRepository._retrieval_scope_filters(
            MagicMock(),
            user_id="user-1",
            project_id="proj-1",
            document_ids=["deleted-doc"],
            owner_scoped=False,
            params=params,
        )
        self.assertIsNotNone(filters)
        self.assertTrue(any("deleted_at IS NULL" in clause for clause in filters))

    async def test_authorization_empty_allow_list_never_widens(self):
        outcome = await self._retrieve(
            dense=[_chunk()],  # would be a success if allow-list were ignored
            lexical=[_chunk(chunk_id="c2")],
            filters={"document_ids": [], "owner_scoped": False},
        )
        self.assertTrue(outcome.no_matches)
        self.assertEqual(outcome.chunks, [])
        self.assertEqual(outcome.fusion_method, "none")
        self.assertFalse(outcome.degraded)


class FailureMatrixAnswerTests(unittest.IsolatedAsyncioTestCase):
    """Generation / citation failures must fail closed."""

    def _user(self) -> User:
        return User(id="user-a", email="user@example.com", password_hash="x")

    def _service(self) -> RagAnswerService:
        db = AsyncMock()
        db.commit = AsyncMock()
        service = RagAnswerService(db)
        service.config = SimpleNamespace(enabled=True, max_context_tokens=6000)
        service.memory_config = SimpleNamespace(enabled=False)
        service.repo = MagicMock()
        service.repo.create_query_record = AsyncMock()
        return service

    async def test_llm_timeout_is_not_normal_success(self):
        service = self._service()
        service.generation = MagicMock()
        service.generation.run_rag_answer = AsyncMock(
            side_effect=TimeoutError("LLM provider timeout")
        )
        outcome = RetrievalOutcome(chunks=[_chunk()])

        with self.assertRaises(TimeoutError):
            await service.answer_from_retrieval(
                "What happened?",
                outcome=outcome,
                user=self._user(),
                project_id=None,
                commit=False,
            )

    async def test_malformed_json_fails_citation_validation(self):
        service = self._service()
        service.generation = MagicMock()
        service.generation.run_rag_answer = AsyncMock(
            return_value=SimpleNamespace(
                id="run-bad",
                output_text="not-json at all {{{",
                model_name="fake-llm",
            )
        )
        result = await service.answer_from_retrieval(
            "What happened?",
            outcome=RetrievalOutcome(chunks=[_chunk()]),
            user=self._user(),
            project_id=None,
            commit=False,
        )
        self.assertTrue(result.citation_validation_failed)
        self.assertEqual(result.answer, CITATION_FAILURE_ANSWER)
        self.assertEqual(result.claims, [])
        self.assertFalse(any(c.used_in_answer for c in result.citations))

    async def test_malformed_json_validator_status_is_not_valid(self):
        validated = CitationValidationService().validate(
            raw_output="definitely not structured claims",
            retrieved_chunks=[_chunk()],
            allowed_document_ids=["doc-1"],
        )
        self.assertTrue(validated.citation_validation_failed)
        self.assertNotEqual(validated.citation_validation_status, "valid")

    async def test_empty_retrieval_answer_is_no_context(self):
        service = self._service()
        service.generation = MagicMock()
        result = await service.answer_from_retrieval(
            "anything",
            outcome=RetrievalOutcome(chunks=[], no_matches=True),
            user=self._user(),
            project_id=None,
            commit=False,
        )
        self.assertTrue(result.no_context_found)
        self.assertEqual(result.answer, NO_CONTEXT_ANSWER)
        self.assertEqual(result.citations, [])
        service.generation.run_rag_answer.assert_not_called()

    async def test_ask_route_timeout_maps_to_504_contract(self):
        """Document the HTTP fail-closed contract without live cloud providers."""
        from backend.modules.rag.api import routes as rag_routes

        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(3600)

        with (
            patch.object(rag_routes, "RagAnswerService") as mock_cls,
            patch.object(rag_routes.settings, "RAG_ASK_TIMEOUT_SECONDS", 0.01),
            patch.object(rag_routes, "_require_rag_enabled"),
        ):
            mock_cls.return_value.answer = _hang
            from fastapi import HTTPException

            with self.assertRaises(HTTPException) as ctx:
                await rag_routes.ask_rag(
                    payload=SimpleNamespace(
                        query="q",
                        project_id=None,
                        run_id=None,
                        agent_id=None,
                        document_ids=None,
                        intent=None,
                    ),
                    db=MagicMock(),
                    current_user=self._user(),
                    _rate_limit=None,
                )
            self.assertEqual(ctx.exception.status_code, 504)
            self.assertIn("timed out", str(ctx.exception.detail).lower())


if __name__ == "__main__":
    unittest.main()
