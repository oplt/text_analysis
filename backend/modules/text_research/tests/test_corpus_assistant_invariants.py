"""Corpus assistant scope and invariant-oriented unit tests."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.rag.domain.models import RetrievalOutcome
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService


class CorpusScopeServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_corpus_yields_empty_allow_list_never_none(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="Empty")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.repo.list_documents = AsyncMock(return_value=[])
        service.rag_repo.get_documents_by_ids = AsyncMock(return_value=[])

        scope = await service.resolve(corpus_id="corp-1", user_id="user-1")
        self.assertIsInstance(scope.rag_document_ids, list)
        self.assertEqual(scope.rag_document_ids, [])
        self.assertIn("Corpus has no documents", scope.warnings)

    async def test_only_indexed_docs_enter_allow_list(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="Mixed")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        docs = [
            MagicMock(id="cd1", rag_document_id="rag-ok"),
            MagicMock(id="cd2", rag_document_id="rag-pending"),
        ]
        service.repo.list_documents = AsyncMock(return_value=docs)
        ok = MagicMock(
            id="rag-ok",
            deleted_at=None,
            status=DocumentStatus.INDEXED.value,
        )
        pending = MagicMock(
            id="rag-pending",
            deleted_at=None,
            status=DocumentStatus.EMBEDDING.value,
        )
        service.rag_repo.get_documents_by_ids = AsyncMock(return_value=[ok, pending])

        scope = await service.resolve(corpus_id="corp-1", user_id="user-1")
        self.assertEqual(scope.rag_document_ids, ["rag-ok"])
        self.assertEqual(scope.unavailable_rag_document_ids, ["rag-pending"])

    async def test_subset_cannot_expand_beyond_corpus(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="A")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.repo.list_documents = AsyncMock(
            return_value=[MagicMock(id="cd1", rag_document_id="rag-1")]
        )
        with self.assertRaises(Exception):
            await service.resolve(
                corpus_id="corp-1",
                user_id="user-1",
                document_subset=["cd-foreign"],
            )


class CorpusAssistantSingleRetrieveTests(unittest.IsolatedAsyncioTestCase):
    async def test_ask_retrieves_once_then_answers_from_retrieval(self):
        from backend.modules.text_research.application.corpus_assistant_service import (
            CorpusAssistantService,
        )

        service = CorpusAssistantService(MagicMock())
        scope = MagicMock(
            corpus_id="corp-1",
            project_id="proj-1",
            corpus_name="C",
            rag_document_ids=["rag-1"],
            corpus_document_ids=["cd1"],
            indexed_rag_document_ids=["rag-1"],
            unavailable_rag_document_ids=[],
            scope_hash="hash",
            index_version="idx",
            retrieval_version="ret",
            to_dict=MagicMock(
                return_value={
                    "corpus_id": "corp-1",
                    "project_id": "proj-1",
                    "corpus_name": "C",
                    "rag_document_ids": ["rag-1"],
                    "corpus_document_ids": ["cd1"],
                    "indexed_rag_document_ids": ["rag-1"],
                    "unavailable_rag_document_ids": [],
                    "scope_hash": "hash",
                    "index_version": "idx",
                    "retrieval_version": "ret",
                    "total_documents": 1,
                    "indexed_count": 1,
                    "unavailable_count": 0,
                    "warnings": [],
                }
            ),
        )
        service.scope_service.resolve = AsyncMock(return_value=scope)
        thread = MagicMock(
            id="thread-1",
            corpus_id="corp-1",
            rag_conversation_id="conv-1",
            project_id="proj-1",
        )
        service.create_thread = AsyncMock(return_value=(thread, scope))
        service.rag_repo.list_messages = AsyncMock(return_value=([], 0))
        service.rag_repo.create_message = AsyncMock(
            side_effect=[
                MagicMock(id="m-user"),
                MagicMock(id="m-asst"),
            ]
        )
        service._rag_to_corpus_document_map = AsyncMock(return_value={"rag-1": "cd1"})
        outcome = RetrievalOutcome(chunks=[], no_matches=True, retrieval_trace_id="trace-1")
        service.retrieval.retrieve = AsyncMock(return_value=outcome)
        answer = MagicMock(
            answer="none",
            citations=[],
            claims=[],
            retrieved_chunk_ids=[],
            model_name="none",
            latency_ms=1,
            retrieval_trace_id="trace-1",
            no_context_found=True,
            retrieval_degraded=False,
            degradation_reason=None,
            citation_validation_failed=False,
            citation_validation_status="valid",
            injection_chunks_filtered=0,
            coverage=None,
            prompt_template_id=None,
            prompt_version_id=None,
            ai_run_id=None,
        )
        service.answers.answer_from_retrieval = AsyncMock(return_value=answer)
        service.answers.answer = AsyncMock()
        service.db.add = MagicMock()
        service.db.commit = AsyncMock()

        user = MagicMock(id="user-1")
        result = await service.ask(corpus_id="corp-1", user=user, query="What?")

        service.retrieval.retrieve.assert_awaited_once()
        call_filters = service.retrieval.retrieve.await_args.kwargs["filters"]
        self.assertEqual(call_filters["document_ids"], ["rag-1"])
        self.assertFalse(call_filters["owner_scoped"])
        service.answers.answer_from_retrieval.assert_awaited_once()
        service.answers.answer.assert_not_awaited()
        self.assertEqual(result["retrieval_trace_id"], "trace-1")


if __name__ == "__main__":
    unittest.main()
