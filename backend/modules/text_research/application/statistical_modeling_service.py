"""Fit user-specified OLS / logistic models on tabular research rows (§50)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps
from backend.modules.text_research.infrastructure.artifact_store import ArtifactStore
from backend.modules.text_research.infrastructure.statistical_modeling import (
    fit_statistical_model,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StatisticalModelingService(ResearchAccessMixin):
    async def fit(
        self,
        corpus_id: str,
        *,
        user_id: str,
        model: str,
        dependent_var: str,
        independent_vars: list[str],
        rows: list[dict[str, Any]],
        add_intercept: bool = True,
    ) -> AnalysisRun:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        if not dependent_var or not independent_vars:
            raise HTTPException(
                status_code=400,
                detail="dependent_var and independent_vars are required",
            )
        if not rows:
            raise HTTPException(status_code=400, detail="rows must be non-empty")
        required = {dependent_var, *independent_vars}
        for i, row in enumerate(rows):
            missing = [name for name in required if name not in row]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"row {i} missing fields: {missing}",
                )
        input_artifact = ArtifactStore().put(
            "manifest",
            {
                "model": model,
                "dependent_var": dependent_var,
                "independent_vars": independent_vars,
                "rows": rows,
                "add_intercept": add_intercept,
            },
            metadata={"kind": "statistical_model_input", "n_rows": len(rows)},
            payload_format="json",
        )
        return await self._fit(
            corpus_id,
            user_id=user_id,
            model=model,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            rows=rows,
            add_intercept=add_intercept,
            input_artifact_id=input_artifact.artifact_id,
            input_artifact_checksum=input_artifact.checksum,
        )

    async def fit_from_artifact(
        self,
        corpus_id: str | None,
        *,
        user_id: str,
        model: str,
        dependent_var: str,
        independent_vars: list[str],
        add_intercept: bool,
        input_artifact_id: str,
    ) -> AnalysisRun:
        if corpus_id is None:
            raise HTTPException(status_code=400, detail="Statistical run has no corpus.")
        descriptor = ArtifactStore().get(input_artifact_id)
        if descriptor is None:
            raise HTTPException(
                status_code=400,
                detail="Statistical input artifact is unavailable.",
            )
        payload = ArtifactStore().load(input_artifact_id)
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise HTTPException(status_code=400, detail="Statistical input artifact is invalid.")
        return await self._fit(
            corpus_id,
            user_id=user_id,
            model=model,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            rows=payload["rows"],
            add_intercept=add_intercept,
            input_artifact_id=input_artifact_id,
            input_artifact_checksum=descriptor.checksum,
        )

    async def _fit(
        self,
        corpus_id: str,
        *,
        user_id: str,
        model: str,
        dependent_var: str,
        independent_vars: list[str],
        rows: list[dict[str, Any]],
        add_intercept: bool,
        input_artifact_id: str,
        input_artifact_checksum: str,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        try:
            fitted = fit_statistical_model(
                rows,
                model=model,
                dependent_var=dependent_var,
                independent_vars=independent_vars,
                add_intercept=add_intercept,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        params = {
            "model": model,
            "dependent_var": dependent_var,
            "independent_vars": independent_vars,
            "add_intercept": add_intercept,
            "n_rows": len(rows),
            "input_artifact_id": input_artifact_id,
            "input_artifact_checksum": input_artifact_checksum,
        }
        spec = build_spec_from_request(
            "statistical_model",
            corpus_id,
            analysis_parameters={
                "model": model,
                "dependent_var": dependent_var,
                "independent_vars": independent_vars,
                "add_intercept": add_intercept,
                "input_artifact_id": input_artifact_id,
                "input_artifact_checksum": input_artifact_checksum,
            },
        )
        run = AnalysisRun(
            project_id=corpus.project_id,
            corpus_id=corpus_id,
            run_type=AnalysisRunType.STATISTICAL_MODEL.value,
            status=AnalysisRunStatus.COMPLETED.value,
            parameters_json=dumps(
                attach_run_identity(
                    params,
                    spec,
                    parent_artifact_checksums=[input_artifact_checksum],
                )
            ),
            metrics_json=dumps(
                {
                    "n_observations": fitted.get("n_observations"),
                    "r_squared": fitted.get("r_squared"),
                    "pseudo_r_squared": fitted.get("pseudo_r_squared"),
                }
            ),
            results_json=dumps(fitted),
            created_by=user_id,
            started_at=_utcnow(),
            completed_at=_utcnow(),
        )
        return await self.repo.create_run(run)
