"""Phase C: streaming exports, prediction freeze, artifacts, dedup."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.export_service import (
    ExportService,
    resolve_export_row_budget,
)
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
    completed_result_reuse_allowed,
    find_reusable_run,
    identity_from_parameters,
    is_raw_embedding_similarity,
    materialize_reuse_run,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps, loads


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

    def test_empty_annotated_snapshot_has_stable_digest(self) -> None:
        frozen = freeze_prediction_selection(
            corpus_id="c1",
            unit_type="paragraph",
            only_unannotated=True,
            filters=None,
            document_ids=None,
            annotated_unit_ids=set(),
        )
        self.assertTrue(frozen["frozen"])
        self.assertEqual(frozen["snapshot_version"], 1)
        self.assertEqual(frozen["annotated_unit_count"], 0)
        self.assertEqual(
            frozen["annotated_unit_ids_hash"],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        # None (not applicable) must not collide with empty frozen set.
        without = freeze_prediction_selection(
            corpus_id="c1",
            unit_type="paragraph",
            only_unannotated=False,
            filters=None,
            document_ids=None,
            annotated_unit_ids=None,
        )
        self.assertIsNone(without["annotated_unit_ids_hash"])
        self.assertNotEqual(frozen["selection_hash"], without["selection_hash"])

    def test_execute_prediction_pages_instead_of_full_list(self) -> None:
        source = inspect.getsource(PredictionService.execute_prediction)
        self.assertIn("iter_text_units_for_corpus", source)
        self.assertIn("selection_snapshot", source)
        self.assertIn('"annotated_unit_ids" in frozen_snapshot', source)
        self.assertNotIn("list_text_units_for_corpus(", source)
        # Must not gate freeze persistence on truthiness of the snapshot dict alone
        # in a way that treats empty annotated lists as missing.
        self.assertIn("selection_snapshot_is_frozen", source)


class EmptyPredictionFreezeResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_honors_empty_frozen_annotation_ids(self) -> None:
        frozen = freeze_prediction_selection(
            corpus_id="corpus-1",
            unit_type="paragraph",
            only_unannotated=True,
            filters=None,
            document_ids=None,
            annotated_unit_ids=set(),
        )
        run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.QUEUED.value,
            parameters_json=dumps(
                {
                    "model_id": "model-1",
                    "unit_type": "paragraph",
                    "only_unannotated": True,
                    "selection_snapshot": {**frozen, "annotated_unit_ids": []},
                }
            ),
            corpus_id="corpus-1",
            project_id="project-1",
            created_by="user-1",
        )
        model = SimpleNamespace(
            id="model-1",
            corpus_id="corpus-1",
            version=1,
            task_type="binary",
            vectorizer_artifact_path="vec.joblib",
            model_artifact_path="model.joblib",
            label_ids_json=dumps(["no", "yes"]),
            metrics_json=dumps({}),
            training_dataset_snapshot_id="snapshot-1",
        )
        unit_added_after_freeze = SimpleNamespace(
            id="newly-annotated", text="still selected", corpus_document_id="doc-1"
        )
        service = PredictionService(MagicMock())
        service.repo = MagicMock()
        service.db = MagicMock(commit=AsyncMock())
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run = AsyncMock(return_value=run)
        service.repo.update_run_if_active = AsyncMock(return_value=run)
        service.repo.get_model = AsyncMock(return_value=model)
        service.repo.list_documents = AsyncMock(return_value=[])
        service.repo.list_annotated_text_unit_ids = AsyncMock(return_value={"newly-annotated"})
        service.repo.count_text_units_for_corpus = AsyncMock(return_value=1)
        service.repo.bulk_upsert_predictions = AsyncMock()

        async def _pages(*_args, **_kwargs):
            yield [unit_added_after_freeze]

        service.repo.iter_text_units_for_corpus = _pages
        draft = SimpleNamespace(id="prediction-set-1", metadata_json=dumps({}))
        with (
            patch(
                "backend.modules.text_research.application.prediction_service.model_storage.load_artifact",
                return_value=object(),
            ),
            patch(
                "backend.modules.text_research.application.prediction_service.predict_with_uncertainty",
                return_value=[{"prediction": 1, "probability": 0.9, "uncertainty": 0.1}],
            ),
            patch(
                "backend.modules.text_research.application.prediction_set_service.PredictionSetService"
            ) as prediction_sets,
        ):
            prediction_sets.return_value.create_draft_from_run = AsyncMock(return_value=draft)
            prediction_sets.return_value.publish = AsyncMock(return_value=draft)
            await service.execute_prediction("run-1")

        service.repo.list_annotated_text_unit_ids.assert_not_awaited()
        rows = service.repo.bulk_upsert_predictions.await_args.args[0]
        self.assertEqual(rows[0]["text_unit_id"], "newly-annotated")
        self.assertEqual(rows[0]["prediction_set_id"], "prediction-set-1")

    async def test_first_freeze_persists_empty_annotated_ids_structurally(self) -> None:
        run = SimpleNamespace(
            id="run-2",
            status=AnalysisRunStatus.QUEUED.value,
            parameters_json=dumps(
                {
                    "model_id": "model-1",
                    "unit_type": "paragraph",
                    "only_unannotated": True,
                    "filters": {},
                }
            ),
            corpus_id="corpus-1",
            project_id="project-1",
            created_by="user-1",
        )
        model = SimpleNamespace(
            id="model-1",
            corpus_id="corpus-1",
            version=1,
            task_type="binary",
            vectorizer_artifact_path="vec.joblib",
            model_artifact_path="model.joblib",
            label_ids_json=dumps(["no", "yes"]),
            metrics_json=dumps({}),
            training_dataset_snapshot_id="snapshot-1",
        )
        service = PredictionService(MagicMock())
        service.repo = MagicMock()
        service.db = MagicMock(commit=AsyncMock())
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run = AsyncMock(return_value=run)
        service.repo.update_run_if_active = AsyncMock(return_value=run)
        service.repo.get_model = AsyncMock(return_value=model)
        service.repo.list_documents = AsyncMock(return_value=[])
        service.repo.list_annotated_text_unit_ids = AsyncMock(return_value=set())
        service.repo.count_text_units_for_corpus = AsyncMock(return_value=0)

        async def _pages(*_args, **_kwargs):
            for _ in ():
                yield []

        service.repo.iter_text_units_for_corpus = _pages
        draft = SimpleNamespace(id="prediction-set-2", metadata_json=dumps({}))
        with (
            patch(
                "backend.modules.text_research.application.prediction_service.model_storage.load_artifact",
                return_value=object(),
            ),
            patch(
                "backend.modules.text_research.application.prediction_set_service.PredictionSetService"
            ) as prediction_sets,
        ):
            prediction_sets.return_value.create_draft_from_run = AsyncMock(return_value=draft)
            prediction_sets.return_value.publish = AsyncMock(return_value=draft)
            await service.execute_prediction("run-2")

        persisted = None
        for call in service.repo.update_run.await_args_list:
            kwargs = call.kwargs
            if "parameters_json" in kwargs:
                persisted = loads(kwargs["parameters_json"], {})
        self.assertIsNotNone(persisted)
        snapshot = persisted["selection_snapshot"]
        self.assertTrue(snapshot["frozen"])
        self.assertEqual(snapshot["snapshot_version"], 1)
        self.assertIn("annotated_unit_ids", snapshot)
        self.assertEqual(snapshot["annotated_unit_ids"], [])
        self.assertEqual(snapshot["annotated_unit_count"], 0)


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

    async def test_prediction_export_honors_explicit_max_rows_without_silent_loss(self) -> None:
        service = ExportService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=SimpleNamespace(id="model-1"))

        async def _list_predictions(_model_id, *, limit, offset):
            rows = [
                SimpleNamespace(
                    text_unit_id=f"u{index}",
                    trained_model_id="model-1",
                    predicted_labels_json='["pos"]',
                    scores_json='{"pos": 0.9}',
                    uncertainty=0.1,
                    created_at=None,
                )
                for index in range(offset, min(offset + limit, 5))
            ]
            return rows, 5

        service.repo.list_predictions_for_model = AsyncMock(side_effect=_list_predictions)
        rows = [
            row async for row in service.iter_predictions_csv("model-1", user_id="u", max_rows=3)
        ]

        self.assertEqual(len(rows), 4)  # Header plus the caller's explicit three rows.
        self.assertEqual(service.repo.list_predictions_for_model.await_count, 1)
        self.assertEqual(service.repo.list_predictions_for_model.await_args.kwargs["limit"], 3)

    async def test_prediction_export_has_no_silent_million_row_default(self) -> None:
        source = inspect.getsource(ExportService.iter_predictions_csv)
        self.assertNotIn("1_000_000", source)
        self.assertNotIn("1000000", source)
        meta = resolve_export_row_budget(1_000_001, max_rows=None)
        self.assertFalse(meta.truncated)
        self.assertEqual(meta.exported_rows, 1_000_001)


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
            store.put.return_value = SimpleNamespace(artifact_id="export:abc", checksum="abc")
            inline, ref = maybe_artifactize_results(huge, max_inline_bytes=500)
        self.assertEqual(ref, "export:abc")
        self.assertTrue(inline["artifactized"])
        self.assertEqual(inline["results_artifact_id"], "export:abc")
        self.assertIn("pairs_preview", inline)
        self.assertEqual(inline["pairs_total"], 5_000)
        self.assertEqual(inline["item_count"], 5_000)


class RunDedupTests(unittest.IsolatedAsyncioTestCase):
    def test_identity_requires_trustworthy_hashes(self) -> None:
        self.assertIsNone(build_computation_identity(analysis_spec_hash=None, corpus_checksum="c"))
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
        params = dumps(
            {
                "analysis_specification": {"analysis": {"type": "frequencies"}},
                "analysis_spec_hash": "spec",
                "corpus_checksum": "corp",
                "pipeline_checksum": "pipe",
                "engine_version": "text_research.pipeline/1",
            }
        )
        active = SimpleNamespace(
            id="active", status=AnalysisRunStatus.RUNNING.value, parameters_json=params
        )
        completed = SimpleNamespace(
            id="done", status=AnalysisRunStatus.COMPLETED.value, parameters_json=params
        )
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

    async def test_parameter_change_uses_a_distinct_computation_identity(self) -> None:
        baseline = build_computation_identity(
            analysis_spec_hash="frequency-top-10",
            corpus_checksum="corp",
            pipeline_checksum="pipe",
        )
        changed = build_computation_identity(
            analysis_spec_hash="frequency-top-20",
            corpus_checksum="corp",
            pipeline_checksum="pipe",
        )
        self.assertNotEqual(baseline, changed)

    async def test_failed_and_cancelled_runs_are_not_reused(self) -> None:
        repo = MagicMock()
        params = dumps(
            {
                "analysis_specification": {"analysis": {"type": "frequencies"}},
                "analysis_spec_hash": "spec",
                "corpus_checksum": "corp",
                "pipeline_checksum": "pipe",
                "engine_version": "text_research.pipeline/1",
            }
        )
        for status in (AnalysisRunStatus.FAILED.value, AnalysisRunStatus.CANCELLED.value):
            with self.subTest(status=status):
                repo.find_run_by_computation_identity = AsyncMock(
                    side_effect=[
                        SimpleNamespace(id="active", status=status, parameters_json=params),
                        SimpleNamespace(id="completed", status=status, parameters_json=params),
                    ]
                )
                reused = await find_reusable_run(
                    repo,
                    project_id="p",
                    corpus_id="c",
                    computation_identity_value="cid",
                )
                self.assertIsNone(reused)

    async def test_completed_reuse_creates_audit_run(self) -> None:
        repo = MagicMock()
        source = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.COMPLETED.value,
            project_id="p",
            corpus_id="c",
            run_type="frequencies",
            parameters_json=dumps(
                {
                    "computation_identity": "cid",
                    "_quantitative_operation": "frequencies",
                    "analysis_specification": {"analysis": {"type": "frequencies"}},
                    "analysis_spec_hash": "spec",
                    "corpus_checksum": "corp",
                    "pipeline_checksum": "pipe",
                    "engine_version": "text_research.pipeline/1",
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

    def test_raw_embedding_similarity_never_allowlisted_for_reuse(self) -> None:
        raw = {
            "_quantitative_operation": "similarity",
            "method": "embedding_cosine",
            "has_embeddings": True,
            "analysis_spec_hash": "spec",
            "corpus_checksum": "corp",
            "pipeline_checksum": "pipe",
            "engine_version": "text_research.pipeline/1",
        }
        self.assertTrue(is_raw_embedding_similarity(raw))
        self.assertFalse(completed_result_reuse_allowed("similarity", raw))

        managed = {
            **raw,
            "embedding_artifact_id": "emb:1",
            "has_embeddings": True,
        }
        self.assertFalse(is_raw_embedding_similarity(managed))
        self.assertTrue(completed_result_reuse_allowed("similarity", managed))

    def test_incomplete_queue_params_cannot_authorize_reuse(self) -> None:
        incomplete = {
            "_quantitative_operation": "frequencies",
            "unit_type": "paragraph",
            "top_n": 50,
            "computation_identity": "stale-cid",
        }
        self.assertIsNone(identity_from_parameters(incomplete))
        self.assertFalse(completed_result_reuse_allowed("frequencies", incomplete))

    def test_unknown_operation_cannot_reuse(self) -> None:
        params = {
            "analysis_specification": {"analysis": {"type": "readability"}},
            "analysis_spec_hash": "spec",
            "corpus_checksum": "corp",
            "pipeline_checksum": "pipe",
            "engine_version": "text_research.pipeline/1",
        }
        self.assertFalse(completed_result_reuse_allowed("readability", params))

    async def test_request_without_hashes_skips_find_reusable(self) -> None:
        repo = MagicMock()
        repo.find_run_by_computation_identity = AsyncMock()
        found = await find_reusable_run(
            repo,
            project_id="p",
            corpus_id="c",
            computation_identity_value="cid",
            request_params={"_quantitative_operation": "frequencies", "top_n": 10},
        )
        self.assertIsNone(found)
        repo.find_run_by_computation_identity.assert_not_awaited()

    async def test_materialize_refuses_incomplete_completed_source(self) -> None:
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
                    "method": "embedding_cosine",
                    "has_embeddings": True,
                    "analysis_spec_hash": "spec",
                    "corpus_checksum": "corp",
                    "pipeline_checksum": "pipe",
                    "engine_version": "text_research.pipeline/1",
                    "analysis_specification": {"analysis": {"type": "similarity"}},
                }
            ),
        )
        with self.assertRaises(ValueError):
            await materialize_reuse_run(repo, source=source, user_id="u1")
        repo.create_run.assert_not_called()

    def test_similarity_min_score_and_query_change_identity(self) -> None:
        from backend.modules.text_research.application.analysis_executor import (
            build_quantitative_spec,
        )
        from backend.modules.text_research.infrastructure.preprocessing import (
            PreprocessingConfig,
        )

        config = PreprocessingConfig().to_dict()

        def _hash(params: dict[str, object]) -> str:
            return build_quantitative_spec(
                "similarity",
                "corpus-1",
                preprocessing_config=config,
                analysis_parameters=params,
            ).spec_hash()

        self.assertNotEqual(
            _hash({"method": "tfidf_cosine", "min_score": 0.1}),
            _hash({"method": "tfidf_cosine", "min_score": 0.2}),
        )
        self.assertNotEqual(
            _hash({"method": "tfidf_cosine", "query_text": "alpha"}),
            _hash({"method": "tfidf_cosine", "query_text": "beta"}),
        )


class LargeCorpusPredictionContractTests(unittest.TestCase):
    def test_prediction_service_uses_projected_annotation_ids(self) -> None:
        from backend.modules.text_research.application import prediction_service

        source = inspect.getsource(prediction_service.PredictionService.execute_prediction)
        self.assertIn("list_annotated_text_unit_ids", source)
        self.assertNotIn("list_annotations_for_corpus", source)
        self.assertIn("iter_text_units_for_corpus", source)


if __name__ == "__main__":
    unittest.main()
