"""Streaming CSV route/service regression tests."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.text_research.api import routes
from backend.modules.text_research.application.export_service import ExportService


class StreamingCsvTests(unittest.IsolatedAsyncioTestCase):
    def test_contextual_import_does_not_read_entire_upload(self):
        source = inspect.getsource(routes.import_contextual_csv)
        self.assertNotIn("await file.read()", source)
        self.assertIn("TextIOWrapper", source)

    async def test_unit_export_yields_rows_incrementally(self):
        service = ExportService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=SimpleNamespace(id="corpus-1"))
        service.repo.list_documents = AsyncMock(
            return_value=[SimpleNamespace(id="doc-1", title="Document")]
        )
        service.repo.list_text_units_for_corpus = AsyncMock(
            return_value=[SimpleNamespace(id="unit-1", corpus_document_id="doc-1", text="Hello")]
        )

        rows = [
            row
            async for row in service.iter_units_csv(
                "corpus-1", user_id="user-1", unit_type="paragraph"
            )
        ]

        self.assertEqual(rows[0], "text_unit_id,corpus_document_id,document_title,text\r\n")
        self.assertEqual(rows[1], "unit-1,doc-1,Document,Hello\r\n")
