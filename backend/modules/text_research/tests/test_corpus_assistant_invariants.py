"""Corpus assistant scope and invariant-oriented unit tests."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

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
        service.rag_repo.list_evidence_revision_chunks = AsyncMock(return_value=[])

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
        with self.assertRaises(HTTPException):
            await service.resolve(
                corpus_id="corp-1",
                user_id="user-1",
                document_subset=["cd-foreign"],
            )

    async def test_scope_paginates_beyond_the_legacy_ten_thousand_limit(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="Large")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        first_page = [
            SimpleNamespace(id=f"cd-{i}", rag_document_id=f"rag-{i}")
            for i in range(50)
        ]
        last_page = [SimpleNamespace(id="cd-50", rag_document_id="rag-50")]
        service.repo.list_documents = AsyncMock(side_effect=[first_page, last_page])
        service.rag_repo.get_documents_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=f"rag-{i}",
                    deleted_at=None,
                    status=DocumentStatus.INDEXED.value,
                    metadata_json="{}",
                )
                for i in range(51)
            ]
        )
        service.rag_repo.list_evidence_revision_chunks = AsyncMock(return_value=[])

        scope = await service.resolve(corpus_id="corp-1", user_id="user-1")

        self.assertEqual(scope.total_documents, 51)
        self.assertEqual(scope.indexed_count, 51)
        self.assertEqual(scope.unavailable_count, 0)
        self.assertEqual(
            [call.kwargs for call in service.repo.list_documents.await_args_list],
            [
                {"limit": 50, "offset": 0},
                {"limit": 50, "offset": 50},
            ],
        )

    async def test_unavailable_accounting_covers_every_corpus_document(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="Availability")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        documents = [
            SimpleNamespace(id="cd-indexed", rag_document_id="rag-indexed"),
            SimpleNamespace(id="cd-missing-id", rag_document_id=None),
            SimpleNamespace(id="cd-uploaded", rag_document_id="rag-uploaded"),
            SimpleNamespace(id="cd-failed", rag_document_id="rag-failed"),
            SimpleNamespace(id="cd-deleted", rag_document_id="rag-deleted"),
            SimpleNamespace(id="cd-missing-source", rag_document_id="rag-missing"),
        ]
        service.repo.list_documents = AsyncMock(return_value=documents)
        service.rag_repo.get_documents_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="rag-indexed",
                    deleted_at=None,
                    status=DocumentStatus.INDEXED.value,
                    metadata_json="{}",
                ),
                SimpleNamespace(
                    id="rag-uploaded",
                    deleted_at=None,
                    status=DocumentStatus.UPLOADED.value,
                    metadata_json="{}",
                ),
                SimpleNamespace(
                    id="rag-failed",
                    deleted_at=None,
                    status=DocumentStatus.FAILED.value,
                    metadata_json="{}",
                ),
                SimpleNamespace(
                    id="rag-deleted",
                    deleted_at=object(),
                    status=DocumentStatus.DELETED.value,
                    metadata_json="{}",
                ),
            ]
        )
        service.rag_repo.list_evidence_revision_chunks = AsyncMock(return_value=[])

        scope = await service.resolve(corpus_id="corp-1", user_id="user-1")

        self.assertEqual(scope.indexed_count + scope.unavailable_count, 6)
        self.assertEqual(scope.indexed_count, 1)
        self.assertEqual(scope.unavailable_count, 5)
        self.assertEqual(
            scope.unavailable_corpus_document_ids,
            [
                "cd-deleted",
                "cd-failed",
                "cd-missing-id",
                "cd-missing-source",
                "cd-uploaded",
            ],
        )
        self.assertEqual(
            scope.unavailable_reasons,
            {
                "cd-deleted": "rag_source_deleted",
                "cd-failed": "indexing_failed",
                "cd-missing-id": "missing_rag_document_id",
                "cd-missing-source": "rag_source_missing",
                "cd-uploaded": "uploaded_not_indexed",
            },
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
        self.assertFalse(
            service.answers.answer_from_retrieval.await_args.kwargs["commit"]
        )
        service.answers.answer.assert_not_awaited()
        self.assertEqual(result["retrieval_trace_id"], "trace-1")

    async def test_generation_failure_persists_failed_assistant_turn(self):
        from backend.modules.text_research.application.corpus_assistant_service import (
            CorpusAssistantService,
        )

        service = CorpusAssistantService(MagicMock())
        scope = SimpleNamespace(project_id="proj-1", rag_document_ids=["rag-1"])
        thread = SimpleNamespace(
            id="thread-1",
            corpus_id="corp-1",
            rag_conversation_id="conv-1",
        )
        service.create_thread = AsyncMock(return_value=(thread, scope))
        service.rag_repo.list_messages = AsyncMock(return_value=([], 0))
        pending_user = SimpleNamespace(
            id="m-user",
            metadata_json='{"turn_status":"pending"}',
        )
        failed_assistant = SimpleNamespace(id="m-failed")
        service.rag_repo.create_message = AsyncMock(
            side_effect=[pending_user, failed_assistant]
        )
        service.retrieval.retrieve = AsyncMock(
            return_value=RetrievalOutcome(chunks=[], no_matches=True)
        )
        service.answers.answer_from_retrieval = AsyncMock(
            side_effect=RuntimeError("provider unavailable")
        )
        service.db.add = MagicMock()
        service.db.commit = AsyncMock()
        service.db.rollback = AsyncMock()

        with self.assertRaises(HTTPException) as context:
            await service.ask(corpus_id="corp-1", user=SimpleNamespace(id="user-1"), query="What?")

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(service.db.commit.await_count, 2)
        service.db.rollback.assert_awaited_once()
        self.assertEqual(json.loads(pending_user.metadata_json)["turn_status"], "failed")
        self.assertEqual(
            json.loads(pending_user.metadata_json)["failed_assistant_message_id"],
            "m-failed",
        )
        failed_call = service.rag_repo.create_message.await_args_list[1]
        self.assertEqual(failed_call.kwargs["metadata"]["turn_status"], "failed")
        self.assertEqual(failed_call.kwargs["metadata"]["error_type"], "RuntimeError")

    async def test_existing_fixed_thread_keeps_persisted_subset_when_ids_omitted(self):
        from backend.modules.text_research.application.corpus_assistant_service import (
            CorpusAssistantService,
        )

        service = CorpusAssistantService(MagicMock())
        thread = SimpleNamespace(
            id="thread-1",
            corpus_id="corp-1",
            project_id="proj-1",
            user_id="user-1",
            rag_conversation_id="conv-1",
        )
        scope = MagicMock(
            project_id="proj-1",
            rag_document_ids=["rag-subset"],
            corpus_id="corp-1",
            scope_hash="subset-hash",
            corpus_document_ids=["cd-subset"],
            indexed_rag_document_ids=["rag-subset"],
            unavailable_rag_document_ids=[],
            index_version="index-v1",
            retrieval_version="retrieval-v1",
        )
        service._get_thread_or_404 = AsyncMock(return_value=thread)
        service.scope_service.resolve_for_thread = AsyncMock(return_value=scope)
        service.rag_repo.list_messages = AsyncMock(return_value=([], 0))
        service.rag_repo.create_message = AsyncMock(
            side_effect=[MagicMock(id="m-user"), MagicMock(id="m-asst")]
        )
        service._rag_to_corpus_document_map = AsyncMock(return_value={})
        service.retrieval.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                chunks=[], fusion_method="dense_only", retrieval_trace_id="trace-1"
            )
        )
        service.answers.answer_from_retrieval = AsyncMock(
            return_value=SimpleNamespace(
                answer="answer",
                citations=[],
                claims=[],
                retrieved_chunk_ids=[],
                model_name="model",
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
        )
        service.db.add = MagicMock()
        service.db.commit = AsyncMock()

        await service.ask(
            corpus_id="corp-1",
            user=SimpleNamespace(id="user-1"),
            query="follow up",
            thread_id="thread-1",
        )

        service.scope_service.resolve_for_thread.assert_awaited_once_with(
            thread, user_id="user-1"
        )
        filters = service.retrieval.retrieve.await_args.kwargs["filters"]
        self.assertEqual(filters["document_ids"], ["rag-subset"])

    async def test_existing_thread_rejects_inline_scope_change(self):
        from backend.modules.text_research.application.corpus_assistant_service import (
            CorpusAssistantService,
        )

        service = CorpusAssistantService(MagicMock())
        service._get_thread_or_404 = AsyncMock(
            return_value=SimpleNamespace(corpus_id="corp-1", project_id="proj-1")
        )
        with self.assertRaises(HTTPException) as ctx:
            await service.ask(
                corpus_id="corp-1",
                user=SimpleNamespace(id="user-1"),
                query="follow up",
                thread_id="thread-1",
                document_subset=["outside-subset"],
            )
        self.assertEqual(ctx.exception.status_code, 409)


class AssistantThreadScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_snapshot_cannot_cross_project_boundary(self):
        service = CorpusScopeService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(
            return_value=SimpleNamespace(id="corp-1", project_id="proj-2", name="Moved")
        )
        thread = SimpleNamespace(
            corpus_id="corp-1",
            project_id="proj-1",
            rag_conversation_id="conv-1",
            scope_mode="fixed",
            scope_snapshot_json='{"corpus_id":"corp-1","project_id":"proj-2","scope_hash":"h"}',
        )
        with self.assertRaises(HTTPException) as ctx:
            await service.resolve_for_thread(thread, user_id="user-1")
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_scope_update_records_explicit_audit_event(self):
        service = CorpusScopeService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(
            return_value=SimpleNamespace(id="corp-1", project_id="proj-1", name="Corpus")
        )
        scope = SimpleNamespace(
            scope_mode="fixed",
            corpus_id="corp-1",
            project_id="proj-1",
            scope_hash="new-hash",
            rag_document_ids=["rag-2"],
            corpus_document_ids=["cd-2"],
            to_dict=lambda: {
                "corpus_id": "corp-1",
                "project_id": "proj-1",
                "scope_hash": "new-hash",
                "rag_document_ids": ["rag-2"],
                "corpus_document_ids": ["cd-2"],
                "scope_mode": "fixed",
            },
        )
        service.resolve = AsyncMock(return_value=scope)
        service.rag_repo.get_conversation = AsyncMock(
            return_value=SimpleNamespace(scope_snapshot_json=None)
        )
        thread = SimpleNamespace(
            id="thread-1",
            corpus_id="corp-1",
            project_id="proj-1",
            rag_conversation_id="conv-1",
            scope_mode="fixed",
            scope_snapshot_json='{"scope_hash":"old-hash"}',
        )
        service.db.add = MagicMock()
        service.db.flush = AsyncMock()

        await service.update_thread_scope(
            thread=thread,
            user_id="user-1",
            document_subset=["cd-2"],
            scope_mode="fixed",
            reason="Narrow to reviewed sources",
        )

        event = service.db.add.call_args.args[0]
        self.assertEqual(event.action, "scope_updated")
        self.assertEqual(event.actor_id, "user-1")
        self.assertEqual(event.previous_scope_hash, "old-hash")
        self.assertEqual(event.new_scope_hash, "new-hash")
        self.assertEqual(event.reason, "Narrow to reviewed sources")


if __name__ == "__main__":
    unittest.main()
