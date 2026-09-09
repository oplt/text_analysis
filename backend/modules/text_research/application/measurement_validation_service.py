"""Application service for measurement triangulation (§51)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps
from backend.modules.text_research.infrastructure.measurement_validation import (
    compare_measurements,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MeasurementValidationService(ResearchAccessMixin):
    async def compare(
        self,
        corpus_id: str,
        *,
        user_id: str,
        source_a: str,
        values_a: list[Any],
        source_b: str,
        values_b: list[Any],
        ids: list[str] | None = None,
        value_kind: str = "categorical",
        subgroup: list[str] | None = None,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        if not source_a or not source_b:
            raise HTTPException(status_code=400, detail="source_a and source_b are required")
        try:
            comparison = compare_measurements(
                source_a,
                values_a,
                source_b,
                values_b,
                ids=ids,
                value_kind=value_kind,
                subgroup=subgroup,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run = AnalysisRun(
            project_id=corpus.project_id,
            corpus_id=corpus_id,
            run_type=AnalysisRunType.MEASUREMENT_VALIDATION.value,
            status=AnalysisRunStatus.COMPLETED.value,
            parameters_json=dumps(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "value_kind": value_kind,
                    "n_input": len(values_a),
                }
            ),
            metrics_json=dumps(
                {
                    "n_paired": comparison.get("n_paired"),
                    "agreement_rate": comparison.get("agreement_rate"),
                    "correlation": comparison.get("correlation"),
                }
            ),
            results_json=dumps(comparison),
            created_by=user_id,
            started_at=_utcnow(),
            finished_at=_utcnow(),
        )
        return await self.repo.create_run(run)
