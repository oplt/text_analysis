"""Streaming CSV route/service regression tests."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.api import routes
from backend.modules.text_research.application.export_service import (
    ExportService,
    resolve_export_row_budget,
)


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

        async def _iter(_corpus_id, **_kwargs):
            yield [SimpleNamespace(id="unit-1", corpus_document_id="doc-1", text="Hello")]

        service.repo.iter_text_units_for_corpus = _iter

        rows = [
            row
            async for row in service.iter_units_csv(
                "corpus-1", user_id="user-1", unit_type="paragraph"
            )
        ]

        self.assertEqual(rows[0], "text_unit_id,corpus_document_id,document_title,text\r\n")
        self.assertEqual(rows[1], "unit-1,doc-1,Document,Hello\r\n")

    def test_resolve_export_row_budget_never_silently_truncates(self):
        full = resolve_export_row_budget(1_500_000, max_rows=None)
        self.assertFalse(full.truncated)
        self.assertEqual(full.exported_rows, 1_500_000)
        self.assertNotIn("X-Export-Truncated", full.as_headers())

        capped = resolve_export_row_budget(1_500_000, max_rows=1_000_000)
        self.assertTrue(capped.truncated)
        self.assertEqual(capped.exported_rows, 1_000_000)
        self.assertEqual(capped.as_headers()["X-Export-Truncated"], "true")
        self.assertEqual(capped.as_headers()["X-Export-Total-Rows"], "1500000")
        self.assertEqual(capped.as_manifest_note()["truncated"], True)

    async def test_prediction_meta_marks_explicit_million_plus_cap(self):
        service = ExportService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=SimpleNamespace(id="model-1"))
        service.repo.list_predictions_for_model = AsyncMock(return_value=([], 1_250_000))
        meta = await service.prediction_export_meta("model-1", user_id="u", max_rows=1_000_000)
        self.assertTrue(meta.truncated)
        self.assertEqual(meta.exported_rows, 1_000_000)
        self.assertEqual(meta.total_rows, 1_250_000)

        uncapped = await service.prediction_export_meta("model-1", user_id="u", max_rows=None)
        self.assertFalse(uncapped.truncated)
        self.assertEqual(uncapped.exported_rows, 1_250_000)

    async def test_prediction_export_route_attaches_truncation_headers(self):
        meta = resolve_export_row_budget(10, max_rows=3)

        async def _empty():
            if False:  # pragma: no cover
                yield ""

        with (
            patch.object(
                ExportService,
                "prediction_export_meta",
                new=AsyncMock(return_value=meta),
            ),
            patch.object(ExportService, "iter_predictions_csv", return_value=_empty()),
        ):
            response = await routes.export_predictions_csv(
                model_id="model-1",
                max_rows=3,
                db=MagicMock(),
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.headers["X-Export-Truncated"], "true")
        self.assertEqual(response.headers["X-Export-Total-Rows"], "10")
        self.assertEqual(response.headers["X-Export-Exported-Rows"], "3")

    def test_http_export_routes_use_iterators_not_materializers(self):
        annotations_src = inspect.getsource(routes.export_annotations_csv)
        predictions_src = inspect.getsource(routes.export_predictions_csv)
        units_src = inspect.getsource(routes.export_units_csv)
        self.assertIn("iter_annotations_csv", annotations_src)
        self.assertIn("iter_predictions_csv", predictions_src)
        self.assertIn("iter_units_csv", units_src)
        self.assertNotIn(".export_annotations_csv(", annotations_src)
        self.assertNotIn(".export_predictions_csv(", predictions_src)
        self.assertIn("as_headers", annotations_src)
        self.assertIn("as_headers", predictions_src)
        self.assertIn("annotation_export_meta", annotations_src)
        self.assertIn("prediction_export_meta", predictions_src)


if __name__ == "__main__":
    unittest.main()
