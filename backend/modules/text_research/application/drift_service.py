"""Drift monitoring for production classifiers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.drift_monitoring import build_drift_report


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


def _aggregate_scores(predictions: list[Any]) -> list[float]:
    scores: list[float] = []
    for row in predictions:
        payload = loads(row.scores_json, {})
        if isinstance(payload, dict):
            for value in payload.values():
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    continue
        if row.uncertainty is not None:
            scores.append(float(row.uncertainty))
    return scores


class DriftService(ResearchAccessMixin):
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
