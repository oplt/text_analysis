"""Phase 3: drift comparison over complete PredictionSets (no browser 500-row cap)."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.text_research.application.drift_service import DriftService
from backend.modules.text_research.infrastructure.drift_monitoring import (
    build_drift_report,
    compute_warning_level,
)


def _prediction(
    *,
    unit_id: str,
    labels: list[str],
    scores: dict[str, float],
    uncertainty: float | None = 0.2,
) -> SimpleNamespace:
    return SimpleNamespace(
        text_unit_id=unit_id,
        predicted_labels_json=json.dumps(labels),
        scores_json=json.dumps(scores),
        uncertainty=uncertainty,
    )


def _prediction_set(
    *,
    set_id: str,
    model_id: str,
    unit_ids: list[str],
    run_id: str = "run-1",
    snapshot_id: str = "snap-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=set_id,
        project_id="proj-1",
        corpus_id="corp-1",
        trained_model_id=model_id,
        dataset_snapshot_id=snapshot_id,
        analysis_run_id=run_id,
        metadata_json=json.dumps({"unit_ids": unit_ids, "unit_count": len(unit_ids)}),
    )


class PredictionSetDriftAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_prediction_sets_aggregates_more_than_500_rows(self) -> None:
        """Backend must load the full persisted set, not a 500-row browser sample."""
        n = 650
        unit_ids = [f"u-{i}" for i in range(n)]
        baseline_preds = [
            _prediction(
                unit_id=uid,
                labels=["yes" if i % 2 == 0 else "no"],
                scores={"yes": 0.9 if i % 2 == 0 else 0.1, "no": 0.1 if i % 2 == 0 else 0.9},
                uncertainty=0.1 + (i % 5) * 0.05,
            )
            for i, uid in enumerate(unit_ids)
        ]
        # Shifted current distribution (more "yes").
        current_preds = [
            _prediction(
                unit_id=uid,
                labels=["yes" if i % 5 != 0 else "no"],
                scores={"yes": 0.8 if i % 5 != 0 else 0.2, "no": 0.2 if i % 5 != 0 else 0.8},
                uncertainty=0.3 + (i % 3) * 0.05,
            )
            for i, uid in enumerate(unit_ids)
        ]
        baseline_set = _prediction_set(
            set_id="ps-base", model_id="model-a", unit_ids=unit_ids, run_id="run-base"
        )
        current_set = _prediction_set(
            set_id="ps-cur", model_id="model-a", unit_ids=unit_ids, run_id="run-cur"
        )
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")

        service = DriftService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_prediction_set_or_404 = AsyncMock(
            side_effect=lambda sid, user_id=None: baseline_set if sid == "ps-base" else current_set
        )

        call_count = {"n": 0}

        async def list_predictions_tracked(model_id, ids):
            call_count["n"] += 1
            self.assertEqual(len(ids), n)
            # First call baseline, second current (same model id).
            return baseline_preds if call_count["n"] == 1 else current_preds

        service.repo = MagicMock()
        service.repo.list_predictions_for_units = AsyncMock(side_effect=list_predictions_tracked)
        service.repo.list_adjudications_for_units = AsyncMock(return_value=[])

        created: dict = {}

        async def create_run(run):
            created["run"] = run
            run.id = "drift-run-large"
            return run

        service.repo.create_run = AsyncMock(side_effect=create_run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        report = await service.compare_prediction_sets(
            "corp-1",
            user_id="user-1",
            mode="PREDICTION_DRIFT",
            baseline_prediction_set_id="ps-base",
            current_prediction_set_id="ps-cur",
        )

        self.assertEqual(report["analysis_run_id"], "drift-run-large")
        self.assertEqual(report["mode"], "PREDICTION_DRIFT")
        self.assertEqual(report["provenance"]["n_baseline"], n)
        self.assertEqual(report["provenance"]["n_current"], n)
        self.assertEqual(report["provenance"]["n_observations"], 2 * n)
        self.assertEqual(report["provenance"]["aggregation"], "full_prediction_set")
        self.assertEqual(report["provenance"]["baseline_prediction_set_id"], "ps-base")
        self.assertEqual(report["provenance"]["current_prediction_set_id"], "ps-cur")
        self.assertEqual(report["provenance"]["baseline_analysis_run_id"], "run-base")
        self.assertEqual(report["provenance"]["current_analysis_run_id"], "run-cur")
        self.assertGreater(report["summary"]["n_observations"], 500)
        self.assertIn(report["summary"]["warning_level"], {"ok", "watch", "investigate"})
        self.assertIn("prediction_distribution", report["sections"])
        self.assertIn("score_distribution", report["sections"])
        self.assertIn("uncertainty_distribution", report["sections"])
        # Aggregation path must have seen the full unit list twice (baseline + current).
        self.assertEqual(service.repo.list_predictions_for_units.await_count, 2)
        first_ids = service.repo.list_predictions_for_units.await_args_list[0].args[1]
        self.assertEqual(len(first_ids), n)
        self.assertGreater(len(first_ids), 500)

    async def test_prediction_drift_rejects_different_models(self) -> None:
        service = DriftService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(
            return_value=SimpleNamespace(id="corp-1", project_id="proj-1")
        )
        baseline = _prediction_set(set_id="ps-a", model_id="model-a", unit_ids=["u1"])
        current = _prediction_set(set_id="ps-b", model_id="model-b", unit_ids=["u1"])
        service.get_prediction_set_or_404 = AsyncMock(
            side_effect=lambda sid, user_id=None: baseline if sid == "ps-a" else current
        )
        with self.assertRaises(HTTPException) as ctx:
            await service.compare_prediction_sets(
                "corp-1",
                user_id="user-1",
                mode="PREDICTION_DRIFT",
                baseline_prediction_set_id="ps-a",
                current_prediction_set_id="ps-b",
            )
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("MODEL_COMPARISON", str(ctx.exception.detail))

    async def test_aggregates_from_prediction_set_separates_confidence_and_uncertainty(
        self,
    ) -> None:
        service = DriftService(MagicMock())
        preds = [
            _prediction(unit_id="u1", labels=["a"], scores={"a": 0.9}, uncertainty=0.15),
            _prediction(unit_id="u2", labels=["b"], scores={"b": 0.7}, uncertainty=0.4),
        ]
        service.repo = MagicMock()
        service.repo.list_predictions_for_units = AsyncMock(return_value=preds)
        service.repo.list_adjudications_for_units = AsyncMock(return_value=[])
        ps = _prediction_set(set_id="ps-1", model_id="m1", unit_ids=["u1", "u2"])
        agg = await service._aggregates_from_prediction_set(ps)
        self.assertEqual(agg["n"], 2)
        self.assertEqual(sorted(agg["scores"]), [0.7, 0.9])
        self.assertEqual(sorted(agg["uncertainties"]), [0.15, 0.4])
        self.assertEqual(agg["label_counts"], {"a": 1, "b": 1})


class DriftReportWarningLevelTests(unittest.TestCase):
    def test_build_drift_report_includes_uncertainty_and_warning_level(self) -> None:
        report = build_drift_report(
            baseline={
                "label_counts": {"yes": 80, "no": 20},
                "scores": [0.9] * 50,
                "uncertainties": [0.1] * 50,
                "n": 100,
            },
            current={
                "label_counts": {"yes": 20, "no": 80},
                "scores": [0.4] * 50,
                "uncertainties": [0.6] * 50,
                "n": 100,
            },
        )
        self.assertIn("uncertainty_distribution", report["sections"])
        self.assertEqual(report["sections"]["score_distribution"]["kind"], "confidence")
        self.assertEqual(report["summary"]["n_observations"], 200)
        self.assertEqual(report["summary"]["warning_level"], "investigate")

    def test_compute_warning_level_ok_for_identical(self) -> None:
        report = build_drift_report(
            baseline={"label_counts": {"a": 50, "b": 50}, "n": 100},
            current={"label_counts": {"a": 50, "b": 50}, "n": 100},
        )
        self.assertEqual(compute_warning_level(report["sections"]), "ok")
        self.assertEqual(report["summary"]["warning_level"], "ok")


if __name__ == "__main__":
    unittest.main()
