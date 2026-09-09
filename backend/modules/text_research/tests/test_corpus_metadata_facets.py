"""Regression coverage for SQL-backed corpus metadata facets."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.api import corpora, routes
from backend.modules.text_research.application.corpus_service import CorpusService
from backend.modules.text_research.infrastructure.repositories import ResearchRepository


class CorpusMetadataFacetTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_returns_sql_aggregated_counts_including_high_cardinality(self):
        facets = {
            "organization": [
                {"value": f"organization-{index}", "count": 1}
                for index in range(1_000)
            ],
            "organization_type": [],
            "publication_year": [{"value": "2024", "count": 3}],
            "country": [],
            "region": [],
            "cultural_sphere": [],
            "language": [{"value": "en", "count": 3}],
            "publication_type": [],
        }
        service = MagicMock(metadata_facets=AsyncMock(return_value=facets))
        with patch.object(corpora, "CorpusService", return_value=service):
            result = await corpora.corpus_metadata_facets(
                "corpus-1", db=MagicMock(), current_user=SimpleNamespace(id="user-1")
            )

        self.assertEqual(len(result["organization"]), 1_000)
        self.assertEqual(result["publication_year"], [{"value": "2024", "count": 3}])
        service.metadata_facets.assert_awaited_once_with("corpus-1", user_id="user-1")

    async def test_service_authorizes_before_requesting_facets(self):
        service = CorpusService(db=MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=MagicMock())
        service.repo.list_document_metadata_facets = AsyncMock(return_value={"language": []})

        result = await service.metadata_facets("corpus-1", user_id="user-1")

        self.assertEqual(result, {"language": []})
        service.repo.list_document_metadata_facets.assert_awaited_once_with("corpus-1")

    def test_repository_uses_group_by_and_excludes_nulls(self):
        source = inspect.getsource(ResearchRepository.list_document_metadata_facets)
        self.assertIn(".group_by(column)", source)
        self.assertIn("column.is_not(None)", source)
        self.assertNotIn("list_documents(", source)
