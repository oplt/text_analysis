"""Regression tests for corpus document pagination totals."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.pagination import paginated_response
from backend.modules.text_research.api import routes
from backend.modules.text_research.application.corpus_service import CorpusService


class DocumentPaginationContractTests(unittest.TestCase):
    def test_paginated_response_keeps_total_independent_of_page_length(self):
        page = paginated_response(["a", "b"], total=42, limit=2, offset=0)
        self.assertEqual(page.total, 42)
        self.assertEqual(len(page.items), 2)
        self.assertEqual(page.limit, 2)
        self.assertEqual(page.offset, 0)

    def test_list_documents_route_does_not_derive_total_from_page_length(self):
        source = inspect.getsource(routes.list_documents)
        self.assertNotIn("total = len(documents)", source)
        self.assertIn("paginate_documents", source)


class CorpusServicePaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_paginate_documents_returns_service_total_not_page_length(self):
        service = CorpusService(db=MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=MagicMock())
        page_items = [MagicMock(id="d1"), MagicMock(id="d2")]
        service.repo.paginate_documents = AsyncMock(return_value=(page_items, 50))

        documents, total = await service.paginate_documents(
            "corpus-1",
            user_id="user-1",
            limit=2,
            offset=0,
        )

        self.assertEqual(len(documents), 2)
        self.assertEqual(total, 50)
        service.repo.paginate_documents.assert_awaited_once_with(
            "corpus-1",
            organization=None,
            publication_year=None,
            region=None,
            cultural_sphere=None,
            language=None,
            search=None,
            sort_by="created_at",
            sort_dir="asc",
            limit=2,
            offset=0,
        )


class ListDocumentsRoutePaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_passes_through_database_total(self):
        documents = [MagicMock(), MagicMock()]
        pagination = MagicMock(limit=2, offset=4)
        current_user = MagicMock(id="user-1")
        db = MagicMock()

        with (
            patch.object(
                routes,
                "CorpusService",
                return_value=MagicMock(paginate_documents=AsyncMock(return_value=(documents, 17))),
            ),
            patch.object(
                routes,
                "_document_response",
                side_effect=lambda doc: {"id": id(doc)},
            ),
        ):
            response = await routes.list_documents(
                "corpus-1",
                pagination=pagination,
                db=db,
                current_user=current_user,
            )

        self.assertEqual(response.total, 17)
        self.assertEqual(response.limit, 2)
        self.assertEqual(response.offset, 4)
        self.assertEqual(len(response.items), 2)


if __name__ == "__main__":
    unittest.main()
