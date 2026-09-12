"""Focused Phase D/E contracts for safe replay and streamed exports."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.api.quantitative_routes import _reject_unsupported_async
from backend.modules.text_research.api.routes import get_run_results, get_run_results_artifact
from backend.modules.text_research.application.export_service import ExportService


class AsyncCapabilityTests(unittest.TestCase):
    def test_inline_only_operation_rejects_explicit_async(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _reject_unsupported_async("kwic", True)
        self.assertEqual(ctx.exception.status_code, 422)


class ResultArtifactAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_results_artifact_loads_full_payload(self) -> None:
        run = SimpleNamespace(
            results_json='{"results_artifact_id": "export:full-results"}',
            metrics_json="{}",
        )
        with (
            patch(
                "backend.modules.text_research.api.routes.RunService.get_run",
                new=AsyncMock(return_value=run),
            ),
            patch(
                "backend.modules.text_research.infrastructure.artifact_store.ArtifactStore.load",
                return_value={"rows": [{"id": "one"}], "total": 1},
            ) as load,
        ):
            payload = await get_run_results_artifact(
                "run-1",
                db=MagicMock(),
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(payload["total"], 1)
        load.assert_called_once_with("export:full-results")

    async def test_results_does_not_load_artifact_when_run_access_is_denied(self) -> None:
        with (
            patch(
                "backend.modules.text_research.api.routes.RunService.get_run",
                new=AsyncMock(side_effect=HTTPException(status_code=404, detail="not found")),
            ),
            patch(
                "backend.modules.text_research.api.routes.ArtifactStore.load",
            ) as load,
            self.assertRaises(HTTPException) as ctx,
        ):
            await get_run_results(
                "other-users-run",
                db=MagicMock(),
                current_user=SimpleNamespace(id="user-1"),
            )
        self.assertEqual(ctx.exception.status_code, 404)
        load.assert_not_called()


class AnnotationStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_annotations_csv_streams_header_before_page_fetch(self) -> None:
        service = ExportService(MagicMock())
        service.get_corpus_or_404 = AsyncMock()
        annotation = SimpleNamespace(
            text_unit_id="unit-1",
            label_id="label-1",
            annotator_id="annotator-1",
            value="yes",
            confidence=0.9,
            comment=None,
            codebook_version="1",
            created_at=None,
            updated_at=None,
        )
        call_order: list[str] = []

        async def _list_page(*_args, **_kwargs):
            call_order.append("page")
            if call_order.count("page") == 1:
                return [annotation]
            return []

        service.repo.list_annotations_for_corpus = AsyncMock(side_effect=_list_page)
        service.repo.list_labels_by_ids = AsyncMock(
            return_value=[SimpleNamespace(id="label-1", name="Important")]
        )

        agen = service.iter_annotations_csv("corpus-1", user_id="user-1", codebook_id=None)
        header = await agen.__anext__()
        call_order.append("got_header")
        self.assertIn("label_name", header)
        # Header must arrive before the first page fetch completes from the
        # consumer's perspective: we got header with zero page calls so far.
        self.assertEqual(call_order, ["got_header"])

        row = await agen.__anext__()
        self.assertIn("Important", row)
        # Drain remaining (empty page break).
        remaining = [line async for line in agen]
        self.assertEqual(remaining, [])
        self.assertGreaterEqual(call_order.count("page"), 1)
