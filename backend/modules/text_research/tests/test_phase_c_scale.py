"""Phase C: streaming exports, prediction freeze, artifacts, dedup."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.export_service import ExportService
from backend.modules.text_research.application.prediction_service import (
    PredictionService,
    freeze_prediction_selection,
)
from backend.modules.text_research.application.result_artifacts import (
    maybe_artifactize_results,
    serialized_size_bytes,
)
from backend.modules.text_research.application.run_dedup import (
    build_computation_identity,
    find_reusable_run,
    identity_from_parameters,
    materialize_reuse_run,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps


class PredictionFreezeTests(unittest.TestCase):
    def test_selection_hash_stable_for_same_snapshot(self) -> None:
        left = freeze_prediction_selection(
            corpus_id="c1",
            unit_type="paragraph",
            only_unannotated=True,
            filters={"language": "en"},
            document_ids=["d2", "d1"],
            annotated_unit_ids={"u2", "u1"},
        )
        right = freeze_prediction_selection(
            corpus_id="c1",
            unit_type="paragraph",
            only_unannotated=True,
            filters={"language": "en"},
            document_ids=["d1", "d2"],
            annotated_unit_ids={"u1", "u2"},
        )
        self.assertEqual(left["selection_hash"], right["selection_hash"])
        changed = freeze_prediction_selection(
            corpus_id="c1",
            unit_type="paragraph",
            only_unannotated=False,
            filters={"language": "en"},
            document_ids=["d1", "d2"],
            annotated_unit_ids={"u1", "u2"},
        )
        self.assertNotEqual(left["selection_hash"], changed["selection_hash"])

    def test_execute_prediction_pages_instead_of_full_list(self) -> None:
        source = inspect.getsource(PredictionService.execute_prediction)
        self.assertIn("iter_text_units_for_corpus", source)
        self.assertIn("selection_snapshot", source)
        self.assertNotIn("list_text_units_for_corpus(", source)


class StreamingExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_iter_units_csv_consumes_pages_incrementally(self) -> None:
        service = ExportService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=SimpleNamespace(id="corpus-1"))
        service.repo.list_documents = AsyncMock(
            return_value=[SimpleNamespace(id="doc-1", title="Document")]
        )

        pages = [
            [SimpleNamespace(id="unit-1", corpus_document_id="doc-1", text="Hello")],
            [SimpleNamespace(id="unit-2", corpus_document_id="doc-1", text="World")],
        ]
        call_count = {"n": 0}

        async def _iter(_corpus_id, **_kwargs):
            for page in pages:
                call_count["n"] += 1
                yield page

        service.repo.iter_text_units_for_corpus = _iter

        gen = service.iter_units_csv("corpus-1", user_id="user-1", unit_type="paragraph")
        header = await gen.__anext__()
        self.assertIn("text_unit_id", header)
        first = await gen.__anext__()
        self.assertIn("unit-1", first)
        self.assertEqual(call_count["n"], 1)
        second = await gen.__anext__()
        self.assertIn("unit-2", second)
        self.assertEqual(call_count["n"], 2)

    async def test_prediction_export_pages_without_million_row_list(self) -> None:
        service = ExportService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=SimpleNamespace(id="model-1"))
        limits: list[int] = []

        async def _list_predictions(model_id, *, limit, offset):
            limits.append(limit)
            if offset > 0:
                return [], 0
            row = SimpleNamespace(
                text_unit_id="u1",
                trained_model_id=model_id,
                predicted_labels_json='["pos"]',
                scores_json='{"pos": 0.9}',
                uncertainty=0.1,
                created_at=None,
            )
            return [row], 1

        service.repo.list_predictions_for_model = AsyncMock(side_effect=_list_predictions)
        rows = [row async for row in service.iter_predictions_csv("model-1", user_id="u")]
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(limit <= service.PREDICTION_EXPORT_PAGE_SIZE for limit in limits))
        self.assertNotIn(1_000_000, limits)


class ResultArtifactTests(unittest.TestCase):
    def test_small_result_stays_inline(self) -> None:
        inline, ref = maybe_artifactize_results({"pairs": [{"a": 1}]}, max_inline_bytes=10_000)
        self.assertIsNone(ref)
        self.assertEqual(inline["pairs"], [{"a": 1}])

    def test_large_result_stores_artifact_and_preview(self) -> None:
        huge = {"pairs": [{"i": i, "score": 0.5} for i in range(5_000)], "item_count": 5_000}
        self.assertGreater(serialized_size_bytes(huge), 1_000)
        with patch(
            "backend.modules.text_research.application.result_artifacts.ArtifactStore"
        ) as store_cls:
            store = store_cls.return_value
            store.put.return_value = SimpleNamespace(
                artifact_id="export:abc", checksum="abc"
            )
            inline, ref = maybe_artifactize_results(huge, max_inline_bytes=500)
        self.assertEqual(ref, "export:abc")
        self.assertTrue(inline["artifactized"])
        self.assertEqual(inline["results_artifact_id"], "export:abc")
        self.assertIn("pairs_preview", inline)
        self.assertEqual(inline["pairs_total"], 5_000)
        self.assertEqual(inline["item_count"], 5_000)


class RunDedupTests(unittest.IsolatedAsyncioTestCase):
    def test_identity_requires_trustworthy_hashes(self) -> None:
        self.assertIsNone(
            build_computation_identity(analysis_spec_hash=None, corpus_checksum="c")
        )
        left = build_computation_identity(
            analysis_spec_hash="spec", corpus_checksum="corp", pipeline_checksum="pipe"
        )
        right = build_computation_identity(
            analysis_spec_hash="spec", corpus_checksum="corp", pipeline_checksum="pipe"
        )
        other = build_computation_identity(
            analysis_spec_hash="other", corpus_checksum="corp", pipeline_checksum="pipe"
        )
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)

    def test_identity_from_parameters(self) -> None:
        params = {
            "analysis_spec_hash": "spec",
            "corpus_checksum": "corp",
            "pipeline_checksum": "pipe",
            "engine_version": "text_research.pipeline/1",
        }
        identity = identity_from_parameters(params)
        self.assertIsNotNone(identity)
        self.assertEqual(
            identity,
            build_computation_identity(
                analysis_spec_hash="spec",
                corpus_checksum="corp",
                pipeline_checksum="pipe",
            ),
        )

    async def test_find_reusable_prefers_active_then_completed(self) -> None:
        repo = MagicMock()
        active = SimpleNamespace(id="active", status=AnalysisRunStatus.RUNNING.value)
        completed = SimpleNamespace(id="done", status=AnalysisRunStatus.COMPLETED.value)
        repo.find_run_by_computation_identity = AsyncMock(side_effect=[active, completed])
        found = await find_reusable_run(
            repo,
            project_id="p",
            corpus_id="c",
            computation_identity_value="cid",
        )
        self.assertEqual(found.id, "active")

    async def test_failed_never_reused_via_materialize_active_passthrough(self) -> None:
        repo = MagicMock()
        source = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.QUEUED.value,
            project_id="p",
            corpus_id="c",
            run_type="similarity",
            parameters_json=dumps({"computation_identity": "cid"}),
            metrics_json="{}",
            results_json="{}",
            artifact_path=None,
            random_seed=None,
            evidence_revision_hash=None,
            started_at=None,
            completed_at=None,
        )
        reused = await materialize_reuse_run(repo, source=source, user_id="u1")
        self.assertIs(reused, source)
        repo.create_run.assert_not_called()

    async def test_completed_reuse_creates_audit_run(self) -> None:
        repo = MagicMock()
        source = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.COMPLETED.value,
            project_id="p",
            corpus_id="c",
            run_type="similarity",
            parameters_json=dumps(
                {
                    "computation_identity": "cid",
                    "analysis_spec_hash": "spec",
                    "corpus_checksum": "corp",
                }
            ),
            metrics_json='{"ok": 1}',
            results_json='{"pairs": []}',
            artifact_path="export:1",
            random_seed=42,
            evidence_revision_hash=None,
            started_at=None,
            completed_at=None,
        )
        created = SimpleNamespace(id="run-2", status=AnalysisRunStatus.COMPLETED.value)
        repo.create_run = AsyncMock(return_value=created)
        reused = await materialize_reuse_run(repo, source=source, user_id="u1")
        self.assertEqual(reused.id, "run-2")
        created_arg = repo.create_run.await_args.args[0]
        self.assertEqual(created_arg.status, AnalysisRunStatus.COMPLETED.value)
        self.assertIn("reused_from_run_id", created_arg.parameters_json)


class LargeCorpusPredictionContractTests(unittest.TestCase):
    def test_prediction_service_uses_projected_annotation_ids(self) -> None:
        from backend.modules.text_research.application import prediction_service

        source = inspect.getsource(prediction_service.PredictionService.execute_prediction)
        self.assertIn("list_annotated_text_unit_ids", source)
        self.assertNotIn("list_annotations_for_corpus", source)
        self.assertIn("iter_text_units_for_corpus", source)


if __name__ == "__main__":
    unittest.main()
