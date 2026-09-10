"""Drift monitoring for production classifiers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.drift_monitoring import (
    build_drift_report,
    compute_warning_level,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aggregate_label_counts(predictions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in predictions:
        labels = loads(row.predicted_labels_json, [])
        if not labels:
            counts["__unlabeled__"] = counts.get("__unlabeled__", 0) + 1
            continue
        for label in labels:
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _aggregate_confidence_scores(predictions: list[Any]) -> list[float]:
    """Collect class confidence/score values (not uncertainty)."""
    scores: list[float] = []
    for row in predictions:
        payload = loads(row.scores_json, {})
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue
    return scores


def _aggregate_uncertainties(predictions: list[Any]) -> list[float]:
    values: list[float] = []
    for row in predictions:
        if row.uncertainty is not None:
            try:
                values.append(float(row.uncertainty))
            except (TypeError, ValueError):
                continue
    return values


def _aggregate_scores(predictions: list[Any]) -> list[float]:
    """Legacy combined score vector (confidence + uncertainty) for aggregate-body mode."""
    return _aggregate_confidence_scores(predictions) + _aggregate_uncertainties(predictions)


class DriftService(ResearchAccessMixin):
    async def compare_prediction_sets(
        self,
        corpus_id: str,
        *,
        user_id: str,
        mode: str,
        baseline_prediction_set_id: str,
        current_prediction_set_id: str,
    ) -> dict[str, Any]:
        """Compare complete persisted PredictionSets with explicit causal semantics."""
        allowed = {"DATA_DRIFT", "PREDICTION_DRIFT", "PERFORMANCE_DRIFT", "MODEL_COMPARISON"}
        if mode not in allowed:
            raise HTTPException(status_code=422, detail=f"Unsupported drift mode {mode!r}")
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        baseline_set = await self.get_prediction_set_or_404(
            baseline_prediction_set_id, user_id=user_id
        )
        current_set = await self.get_prediction_set_or_404(
            current_prediction_set_id, user_id=user_id
        )
        if baseline_set.corpus_id != corpus.id or current_set.corpus_id != corpus.id:
            raise HTTPException(
                status_code=422, detail="Prediction sets must belong to the requested corpus"
            )
        if (
            mode == "PREDICTION_DRIFT"
            and baseline_set.trained_model_id != current_set.trained_model_id
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "PREDICTION_DRIFT requires two PredictionSets from the same trained model; "
                    "use MODEL_COMPARISON instead."
                ),
            )
        baseline = await self._aggregates_from_prediction_set(baseline_set)
        current = await self._aggregates_from_prediction_set(current_set)
        report = build_drift_report(baseline=baseline, current=current)
        report["mode"] = mode
        report["provenance"] = {
            "baseline_prediction_set_id": baseline_set.id,
            "current_prediction_set_id": current_set.id,
            "baseline_analysis_run_id": baseline_set.analysis_run_id,
            "current_analysis_run_id": current_set.analysis_run_id,
            "baseline_trained_model_id": baseline_set.trained_model_id,
            "current_trained_model_id": current_set.trained_model_id,
            "baseline_snapshot_id": baseline_set.dataset_snapshot_id,
            "current_snapshot_id": current_set.dataset_snapshot_id,
            "n_baseline": baseline["n"],
            "n_current": current["n"],
            "n_observations": baseline["n"] + current["n"],
            "labeled_n": min(baseline.get("labeled_n", 0), current.get("labeled_n", 0)),
            "aggregation": "full_prediction_set",
        }
        summary = dict(report.get("summary") or {})
        summary["n_baseline"] = baseline["n"]
        summary["n_current"] = current["n"]
        summary["n_observations"] = baseline["n"] + current["n"]
        report["summary"] = summary
        if mode == "PERFORMANCE_DRIFT":
            if not baseline.get("labeled_n") or not current.get("labeled_n"):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "PERFORMANCE_DRIFT requires adjudicated gold labels in both PredictionSets."
                    ),
                )
            report["performance"] = {
                "baseline_accuracy": baseline["gold_accuracy"],
                "current_accuracy": current["gold_accuracy"],
                "difference": current["gold_accuracy"] - baseline["gold_accuracy"],
            }
            sections = dict(report.get("sections") or {})
            sections["performance"] = report["performance"]
            report["sections"] = sections
            report["summary"]["warning_level"] = compute_warning_level(report["sections"])
            report["summary"]["section_count"] = len(report["sections"])
            report["summary"]["has_performance_drift"] = True
        run = await self._persist_report(
            corpus=corpus,
            user_id=user_id,
            baseline_run_id=baseline_set.analysis_run_id,
            current_run_id=current_set.analysis_run_id,
            report=report,
            provenance=report["provenance"],
        )
        report["analysis_run_id"] = run.id
        return report

    async def _aggregates_from_prediction_set(self, prediction_set: Any) -> dict[str, Any]:
        """Aggregate ALL predictions belonging to a persisted PredictionSet (no row cap)."""
        unit_ids = list(loads(prediction_set.metadata_json, {}).get("unit_ids") or [])
        predictions = await self.repo.list_predictions_for_units(
            prediction_set.trained_model_id, unit_ids
        )
        adjudications = await self.repo.list_adjudications_for_units(unit_ids)
        gold_by_unit: dict[str, set[str]] = {}
        for adjudication in adjudications:
            gold_by_unit.setdefault(adjudication.text_unit_id, set()).add(adjudication.final_value)
        correct = 0
        labeled = 0
        for prediction in predictions:
            gold = gold_by_unit.get(prediction.text_unit_id)
            if gold:
                labeled += 1
                if set(str(label) for label in loads(prediction.predicted_labels_json, [])) == gold:
                    correct += 1
        return {
            "label_counts": _aggregate_label_counts(predictions),
            "scores": _aggregate_confidence_scores(predictions),
            "uncertainties": _aggregate_uncertainties(predictions),
            "n": len(predictions),
            "labeled_n": labeled,
            "gold_accuracy": correct / labeled if labeled else None,
        }

    async def compare_prediction_runs(
        self,
        corpus_id: str,
        *,
        user_id: str,
        baseline_run_id: str,
        current_run_id: str,
        baseline: dict[str, Any] | None = None,
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        baseline_run = await self.get_run_or_404(baseline_run_id, user_id=user_id)
        current_run = await self.get_run_or_404(current_run_id, user_id=user_id)

        if baseline_run.corpus_id != corpus.id or current_run.corpus_id != corpus.id:
            raise HTTPException(status_code=422, detail="Runs must belong to the requested corpus")

        baseline_payload = dict(baseline or {})
        current_payload = dict(current or {})

        if not baseline_payload:
            baseline_payload = await self._aggregates_from_run(baseline_run)
        if not current_payload:
            current_payload = await self._aggregates_from_run(current_run)

        if not baseline_payload or not current_payload:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Provide baseline/current aggregates in the request body when runs "
                    "do not store label distributions"
                ),
            )

        report = build_drift_report(baseline=baseline_payload, current=current_payload)
        run = await self._persist_report(
            corpus=corpus,
            user_id=user_id,
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            report=report,
        )
        report["analysis_run_id"] = run.id
        return report

    async def compare_distributions(
        self,
        corpus_id: str,
        *,
        user_id: str,
        baseline: dict[str, Any],
        current: dict[str, Any],
        baseline_run_id: str | None = None,
        current_run_id: str | None = None,
    ) -> dict[str, Any]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        report = build_drift_report(baseline=baseline, current=current)
        run = await self._persist_report(
            corpus=corpus,
            user_id=user_id,
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            report=report,
        )
        report["analysis_run_id"] = run.id
        return report

    async def _aggregates_from_run(self, run: AnalysisRun) -> dict[str, Any]:
        stored = loads(run.results_json, {}) if run.results_json else {}
        if isinstance(stored, dict) and stored.get("label_counts"):
            payload: dict[str, Any] = {"label_counts": stored["label_counts"]}
            if stored.get("scores"):
                payload["scores"] = stored["scores"]
            if stored.get("top_terms"):
                payload["top_terms"] = stored["top_terms"]
            return payload

        params = loads(run.parameters_json, {})
        model_id = params.get("model_id")
        if not model_id or run.run_type != AnalysisRunType.CLASSIFIER_PREDICTION.value:
            return {}

        predictions, _total = await self.repo.list_predictions_for_model(
            model_id, limit=100_000, offset=0
        )
        if not predictions:
            return {}

        return {
            "label_counts": _aggregate_label_counts(predictions),
            "scores": _aggregate_scores(predictions),
        }

    async def _persist_report(
        self,
        *,
        corpus: Any,
        user_id: str,
        baseline_run_id: str | None,
        current_run_id: str | None,
        report: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=AnalysisRunType.DRIFT_MONITORING.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(
                    {
                        "baseline_run_id": baseline_run_id,
                        "current_run_id": current_run_id,
                        **(provenance or {}),
                    }
                ),
                metrics_json=dumps(report.get("summary", {})),
                results_json=dumps(report),
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.db.commit()
        return run
