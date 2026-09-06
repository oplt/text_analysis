import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class RagChunkListingTest(unittest.IsolatedAsyncioTestCase):
    async def test_list_document_chunks_defaults_to_snippet_content(self):
        from backend.modules.rag.api.routes import list_document_chunks

        repo = MagicMock()
        repo.get_document = AsyncMock(
            return_value=SimpleNamespace(id="doc-1", user_id="user-a")
        )
        long_content = "x" * 500
        repo.list_chunks_for_document = AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        id="chunk-1",
                        document_id="doc-1",
                        chunk_index=0,
                        content=long_content,
                        token_count=120,
                        metadata_json=json.dumps({"page_number": 1}),
                    )
                ],
                1,
            )
        )

        with patch("backend.modules.rag.api.routes.RagRepository", return_value=repo):
            response = await list_document_chunks(
                document_id="doc-1",
                pagination=SimpleNamespace(limit=50, offset=0),
                content_mode="snippet",
                db=AsyncMock(),
                current_user=SimpleNamespace(id="user-a"),
            )

        self.assertEqual(response.total, 1)
        self.assertLess(len(response.items[0].content), len(long_content))
        self.assertTrue(response.items[0].content.endswith("…"))

    async def test_list_document_chunks_full_mode_returns_raw_content(self):
        from backend.modules.rag.api.routes import list_document_chunks

        repo = MagicMock()
        repo.get_document = AsyncMock(
            return_value=SimpleNamespace(id="doc-1", user_id="user-a")
        )
        long_content = "x" * 500
        repo.list_chunks_for_document = AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        id="chunk-1",
                        document_id="doc-1",
                        chunk_index=0,
                        content=long_content,
                        token_count=120,
                        metadata_json="{}",
                    )
                ],
                1,
            )
        )

        with patch("backend.modules.rag.api.routes.RagRepository", return_value=repo):
            response = await list_document_chunks(
                document_id="doc-1",
                pagination=SimpleNamespace(limit=50, offset=0),
                content_mode="full",
                db=AsyncMock(),
                current_user=SimpleNamespace(id="user-a"),
            )

        self.assertEqual(response.items[0].content, long_content)


if __name__ == "__main__":
    unittest.main()
