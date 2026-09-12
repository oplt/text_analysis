from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from backend.lib.vector_search import DenseFallbackScopeTooLarge
from backend.modules.rag.infrastructure.repositories import RagRepository


class DenseJsonFallbackLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_scope_fails_instead_of_sampling_recent_rows(self):
        db = MagicMock()
        count = MagicMock()
        count.scalar.return_value = 6
        db.execute = AsyncMock(return_value=count)
        repo = RagRepository(db)
        repo._retrieval_scope_filters = MagicMock(return_value=["TRUE"])

        with self.assertRaises(DenseFallbackScopeTooLarge):
            await repo.similarity_search_json_fallback(
                user_id="user-1",
                project_id=None,
                document_ids=["doc-1"],
                query_embedding=[1.0, 0.0],
                top_k=2,
                score_threshold=0.0,
                exact_max_rows=5,
            )

    async def test_small_scope_fetches_without_recency_limit(self):
        db = MagicMock()
        count = MagicMock()
        count.scalar.return_value = 1
        rows = MagicMock()
        rows.mappings.return_value.all.return_value = [
            {
                "chunk_id": "old-relevant",
                "document_id": "doc-1",
                "content": "old evidence",
                "chunk_index": 0,
                "metadata_json": "{}",
                "revision_id": "revision-1",
                "embedding_json": "[1.0, 0.0]",
                "original_filename": "evidence.txt",
            }
        ]
        db.execute = AsyncMock(side_effect=[count, rows])
        repo = RagRepository(db)
        repo._retrieval_scope_filters = MagicMock(return_value=["TRUE"])

        results = await repo.similarity_search_json_fallback(
            user_id="user-1",
            project_id=None,
            document_ids=["doc-1"],
            query_embedding=[1.0, 0.0],
            top_k=1,
            score_threshold=0.0,
            exact_max_rows=5,
        )
        self.assertEqual([chunk.chunk_id for chunk in results], ["old-relevant"])
        self.assertNotIn("updated_at DESC", str(db.execute.await_args_list[1].args[0]))
