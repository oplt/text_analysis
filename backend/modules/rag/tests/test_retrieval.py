import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.pgvector_adapter import PgVectorAdapter
from backend.lib.vectors import cosine_similarity


class RetrievalFilterTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = AsyncMock()
        self.config = SimpleNamespace(score_threshold=0.1, vector_backend="pgvector")
        self.adapter = PgVectorAdapter(self.db, self.config)
        self.adapter.repo = MagicMock()
        self.adapter.repo.similarity_search_indexed = AsyncMock(return_value=None)

    async def test_retrieval_filters_by_user_id(self):
        self.adapter.repo.similarity_search_json_fallback = AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="doc-a",
                    content="user a secret",
                    score=0.95,
                    filename="a.txt",
                    chunk_index=0,
                )
            ]
        )
        results = await self.adapter.similarity_search(
            "secret",
            user_id="user-a",
            project_id=None,
            top_k=5,
            query_embedding=[1.0, 0.0],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")
        self.adapter.repo.similarity_search_indexed.assert_awaited_once()

    async def test_retrieval_filters_by_project_id(self):
        self.adapter.repo.similarity_search_json_fallback = AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="doc-p",
                    content="project one",
                    score=0.9,
                    filename="p.txt",
                    chunk_index=0,
                    page_number=0,
                )
            ]
        )
        results = await self.adapter.similarity_search(
            "project",
            user_id="user-a",
            project_id="proj-1",
            top_k=5,
            query_embedding=[1.0, 0.0],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "project one")
        call_kwargs = self.adapter.repo.similarity_search_json_fallback.await_args.kwargs
        self.assertEqual(call_kwargs["project_id"], "proj-1")

    async def test_uses_indexed_search_when_available(self):
        indexed_hit = RetrievedChunk(
            chunk_id="idx-1",
            document_id="doc-a",
            content="indexed hit",
            score=0.99,
            filename="a.txt",
            chunk_index=0,
        )
        self.adapter.repo.similarity_search_indexed = AsyncMock(return_value=[indexed_hit])
        self.adapter.repo.similarity_search_json_fallback = AsyncMock()

        results = await self.adapter.similarity_search(
            "query",
            user_id="user-a",
            project_id=None,
            top_k=5,
            query_embedding=[0.1] * 1536,
        )
        self.assertEqual(results[0].chunk_id, "idx-1")
        self.adapter.repo.similarity_search_json_fallback.assert_not_called()

    async def test_deleted_document_not_in_indexed_list(self):
        self.adapter.repo.similarity_search_indexed = AsyncMock(return_value=[])
        self.adapter.repo.similarity_search_json_fallback = AsyncMock(return_value=[])
        results = await self.adapter.similarity_search(
            "query",
            user_id="user-a",
            project_id=None,
            top_k=5,
            query_embedding=[1.0, 0.0],
        )
        self.assertEqual(results, [])
        self.adapter.repo.similarity_search_json_fallback.assert_awaited_once()


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors_score_high(self):
        score = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(score, 1.0)
