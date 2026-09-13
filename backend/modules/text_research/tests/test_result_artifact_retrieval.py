"""LATEST-010: authorized paging / download for artifactized run results."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.api.routes import (
    download_run_results,
    get_run_results,
    get_run_results_artifact,
)
from backend.modules.text_research.application.result_artifacts import (
    load_full_run_results,
    maybe_artifactize_results,
    page_run_results,
    resolve_results_artifact_id,
    serialized_size_bytes,
)
from backend.modules.text_research.infrastructure.artifact_store import ArtifactStore


class ResultArtifactizePreviewTests(unittest.TestCase):
    def test_preview_stays_small_when_payload_is_artifactized(self) -> None:
        huge = {"pairs": [{"i": i, "pad": "x" * 80} for i in range(400)]}
        self.assertGreater(serialized_size_bytes(huge), 500)

        store = MagicMock(spec=ArtifactStore)
        store.put.return_value = SimpleNamespace(
            artifact_id="export:chk",
            checksum="chk",
        )
        inline, ref = maybe_artifactize_results(
            huge,
            max_inline_bytes=500,
            preview_rows=25,
            producing_run_id="run-9",
            store=store,
        )

        self.assertEqual(ref, "export:chk")
        self.assertTrue(inline["artifactized"])
        self.assertEqual(inline["results_artifact_id"], "export:chk")
        self.assertEqual(inline["pairs_total"], 400)
        self.assertEqual(len(inline["pairs_preview"]), 25)
        self.assertLess(serialized_size_bytes(inline), serialized_size_bytes(huge))
        store.put.assert_called_once()
        self.assertEqual(store.put.call_args.kwargs["producing_run_id"], "run-9")


class ResultPagingTests(unittest.TestCase):
    def test_large_array_pages_correctly_from_artifact(self) -> None:
        full = {"pairs": [{"id": i} for i in range(250)]}
        run = SimpleNamespace(
            results_json=(
                '{"artifactized": true, "results_artifact_id": "export:abc", '
                '"results_checksum": "abc", "results_bytes": 9999, "preview_rows": 50}'
            ),
            metrics_json="{}",
        )
        descriptor = SimpleNamespace(
            artifact_id="export:abc",
            checksum="abc",
            metadata={
                "kind": "analysis_results",
                "bytes": 9999,
                "schema_version": 1,
                "media_type": "application/json",
                "preview_rows": 50,
            },
        )
        store = MagicMock(spec=ArtifactStore)
        store.get.return_value = descriptor
        store.load.return_value = full

        page = page_run_results(run, key="pairs", limit=100, offset=100, store=store)
        self.assertTrue(page.artifactized)
        self.assertEqual(page.checksum, "abc")
        self.assertEqual(page.media_type, "application/json")
        self.assertEqual(page.schema_version, 1)
        self.assertEqual(page.byte_size, 9999)
        self.assertEqual(page.preview_rows, 50)
        self.assertEqual(page.total, 250)
        self.assertEqual(page.row_count, 250)
        self.assertEqual(page.items, [{"id": i} for i in range(100, 200)])

    def test_full_artifact_checksum_matches_descriptor(self) -> None:
        run = SimpleNamespace(
            results_json='{"results_artifact_id": "export:chk-1"}',
            metrics_json="{}",
        )
        descriptor = SimpleNamespace(
            artifact_id="export:chk-1",
            checksum="chk-1",
            metadata={"bytes": 12},
        )
        store = MagicMock(spec=ArtifactStore)
        store.get.return_value = descriptor
        store.load.return_value = {"ok": True}

        payload, artifact_id, loaded = load_full_run_results(run, store=store)
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(artifact_id, "export:chk-1")
        self.assertEqual(loaded.checksum, "chk-1")

    def test_resolve_prefers_explicit_results_artifact_over_path(self) -> None:
        run = SimpleNamespace(
            results_json='{"results_artifact_id": "export:from-results"}',
            metrics_json='{"results_artifact_id": "export:from-metrics"}',
            artifact_path="models/should-not-use",
        )
        self.assertEqual(resolve_results_artifact_id(run), "export:from-results")


class ResultArtifactRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_unauthorized_user_cannot_fetch_results(self) -> None:
        with (
            patch(
                "backend.modules.text_research.api.routes.RunService.get_run",
                new=AsyncMock(side_effect=HTTPException(status_code=404, detail="not found")),
            ),
            patch(
                "backend.modules.text_research.application.result_artifacts.ArtifactStore.load",
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
                "backend.modules.text_research.application.result_artifacts.load_full_run_results",
                return_value=({"rows": [{"id": "one"}], "total": 1}, "export:full-results", None),
            ),
        ):
            payload = await get_run_results_artifact(
                "run-1",
                db=MagicMock(),
                current_user=SimpleNamespace(id="user-1"),
            )
        self.assertEqual(payload["total"], 1)

    async def test_download_includes_checksum_header(self) -> None:
        run = SimpleNamespace(
            results_json='{"results_artifact_id": "export:dl"}',
            metrics_json="{}",
        )
        descriptor = SimpleNamespace(checksum="deadbeef")
        with (
            patch(
                "backend.modules.text_research.api.routes.RunService.get_run",
                new=AsyncMock(return_value=run),
            ),
            patch(
                "backend.modules.text_research.application.result_artifacts.load_full_run_results",
                return_value=({"pairs": [1]}, "export:dl", descriptor),
            ),
        ):
            response = await download_run_results(
                "run-dl",
                db=MagicMock(),
                current_user=SimpleNamespace(id="user-1"),
            )
        self.assertEqual(response.headers["X-Results-Artifact-Id"], "export:dl")
        self.assertEqual(response.headers["X-Results-Checksum"], "deadbeef")


if __name__ == "__main__":
    unittest.main()
