"""Collaborator ACL matrix for Ask Corpus (invariant I2) + parent expand (I5)."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.rag.application.parent_context_service import expand_parent_chunks
from backend.modules.rag.application.retrieval_planner import plan_retrieval
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.text_research.application.corpus_assistant_service import (
    CorpusAssistantService,
)
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService


def _chunk(chunk_id: str, document_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"child-{chunk_id}",
        score=score,
        filename=f"{document_id}.txt",
        chunk_index=0,
    )


class CollaboratorAclMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_member_resolve_succeeds_after_project_access(self):
        service = CorpusScopeService(MagicMock())
        corpus = MagicMock(id="corp-1", project_id="proj-1", name="Shared")
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.repo.list_documents = AsyncMock(
            return_value=[MagicMock(id="cd1", rag_document_id="rag-1")]
        )
        rag_doc = MagicMock(id="rag-1", deleted_at=None, status="indexed")
        service.rag_repo.get_documents_by_ids = AsyncMock(return_value=[rag_doc])

        scope = await service.resolve(corpus_id="corp-1", user_id="collaborator")
        self.assertEqual(scope.rag_document_ids, ["rag-1"])
        service.get_corpus_or_404.assert_awaited_once_with("corp-1", user_id="collaborator")

    async def test_unrelated_user_denied_via_get_corpus_or_404(self):
        service = CorpusScopeService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(
            side_effect=HTTPException(status_code=404, detail="Research corpus not found")
        )
        with self.assertRaises(HTTPException) as ctx:
            await service.resolve(corpus_id="corp-1", user_id="stranger")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_assistant_retrieve_uses_owner_scoped_false_for_collaborators(self):
        service = CorpusAssistantService(MagicMock())
        scope = MagicMock(
            project_id="proj-1",
            rag_document_ids=["rag-peer"],
            to_dict=MagicMock(return_value={}),
        )
        service.scope_service.resolve = AsyncMock(return_value=scope)
        service.retrieval.retrieve = AsyncMock(
            return_value=MagicMock(chunks=[], retrieval_trace_id="t1")
        )
        user = MagicMock(id="collaborator")

        await service.retrieve(
            corpus_id="corp-1",
            user=user,
            query="accountability",
            retrieval_mode="dense",
        )
        filters = service.retrieval.retrieve.await_args.kwargs["filters"]
        self.assertFalse(filters["owner_scoped"])
        self.assertEqual(filters["document_ids"], ["rag-peer"])
        self.assertEqual(filters["retrieval_mode"], "dense")


class ParentExpandAllowListTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_outside_allow_list_not_expanded(self):
        child = _chunk("child-1", "doc-a")
        repo = MagicMock()
        child_row = MagicMock(id="child-1", parent_chunk_id="parent-1", document_id="doc-a")
        parent_row = MagicMock(
            id="parent-1",
            document_id="doc-OTHER",
            content="PARENT TEXT",
            chunk_index=0,
        )
        repo.get_chunks_by_ids = AsyncMock(side_effect=[[child_row], [parent_row]])

        expanded = await expand_parent_chunks(
            [child], repo=repo, document_ids=["doc-a"]
        )
        self.assertEqual(expanded[0].content, "child-child-1")

    async def test_parent_inside_allow_list_replaces_content(self):
        child = _chunk("child-1", "doc-a")
        repo = MagicMock()
        child_row = MagicMock(id="child-1", parent_chunk_id="parent-1", document_id="doc-a")
        parent_row = MagicMock(
            id="parent-1",
            document_id="doc-a",
            content="PARENT TEXT",
            chunk_index=0,
        )
        repo.get_chunks_by_ids = AsyncMock(side_effect=[[child_row], [parent_row]])

        expanded = await expand_parent_chunks(
            [child], repo=repo, document_ids=["doc-a"]
        )
        self.assertEqual(expanded[0].content, "PARENT TEXT")
        self.assertIn("parent_expand", expanded[0].retrieval_sources)


class RetrievalModePlannerTests(unittest.TestCase):
    def test_dense_mode_zeros_lexical(self):
        config = MagicMock(
            top_k=5,
            evidence_top_k=12,
            dense_candidates=40,
            lexical_candidates=40,
            source_max_chunks_per_document=3,
        )
        plan = plan_retrieval("semantic_search", config, retrieval_mode="dense")
        self.assertGreater(plan.dense_candidates, 0)
        self.assertEqual(plan.lexical_candidates, 0)

    def test_lexical_mode_zeros_dense(self):
        config = MagicMock(
            top_k=5,
            evidence_top_k=12,
            dense_candidates=40,
            lexical_candidates=40,
            source_max_chunks_per_document=3,
        )
        plan = plan_retrieval("semantic_search", config, retrieval_mode="lexical")
        self.assertEqual(plan.dense_candidates, 0)
        self.assertGreater(plan.lexical_candidates, 0)


class SynthesisEmptyShortCircuitTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_allow_list_never_enqueues(self):
        from backend.modules.text_research.application.corpus_synthesis_service import (
            CorpusSynthesisService,
        )

        service = CorpusSynthesisService(MagicMock())
        service.rag_config = MagicMock(enabled=True, synthesis_max_documents=25)
        empty_scope = MagicMock(
            rag_document_ids=[],
            indexed_count=0,
            to_dict=MagicMock(return_value={"indexed_count": 0}),
        )
        service.scope_service.resolve = AsyncMock(return_value=empty_scope)
        service._enqueue = AsyncMock()

        result = await service.synthesize(
            corpus_id="corp-1",
            user=MagicMock(id="u1"),
            query="summarize",
            async_mode=True,
        )
        self.assertEqual(result["mode"], "sync")
        self.assertEqual(result["document_findings"], [])
        service._enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
