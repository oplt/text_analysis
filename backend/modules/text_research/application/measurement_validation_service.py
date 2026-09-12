"""Application service for measurement triangulation (§51)."""

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
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        if not source_a or not source_b:
            raise HTTPException(status_code=400, detail="source_a and source_b are required")
        input_artifact = ArtifactStore().put(
            "manifest",
            {
                "source_a": source_a,
                "values_a": values_a,
                "source_b": source_b,
                "values_b": values_b,
                "ids": ids,
                "value_kind": value_kind,
                "subgroup": subgroup,
            },
            metadata={"kind": "measurement_comparison_input", "n_input": len(values_a)},
            payload_format="json",
        )
        return await self._compare(
            corpus_id,
            user_id=user_id,
            source_a=source_a,
            values_a=values_a,
            source_b=source_b,
            values_b=values_b,
            ids=ids,
            value_kind=value_kind,
            subgroup=subgroup,
            input_artifact_id=input_artifact.artifact_id,
            input_artifact_checksum=input_artifact.checksum,
        )

    async def compare_from_artifact(
        self, corpus_id: str | None, *, user_id: str, input_artifact_id: str
    ) -> AnalysisRun:
        if corpus_id is None:
            raise HTTPException(status_code=400, detail="Measurement run has no corpus.")
        descriptor = ArtifactStore().get(input_artifact_id)
        if descriptor is None:
            raise HTTPException(
                status_code=400,
                detail="Measurement input artifact is unavailable.",
            )
        payload = ArtifactStore().load(input_artifact_id)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Measurement input artifact is invalid.")
        return await self._compare(
            corpus_id,
            user_id=user_id,
            source_a=str(payload.get("source_a") or ""),
            values_a=list(payload.get("values_a") or []),
            source_b=str(payload.get("source_b") or ""),
            values_b=list(payload.get("values_b") or []),
            ids=payload.get("ids"),
            value_kind=str(payload.get("value_kind") or "categorical"),
            subgroup=payload.get("subgroup"),
            input_artifact_id=input_artifact_id,
            input_artifact_checksum=descriptor.checksum,
        )

    async def _compare(
        self,
        corpus_id: str,
        *,
        user_id: str,
        source_a: str,
        values_a: list[Any],
        source_b: str,
        values_b: list[Any],
        ids: list[str] | None,
        value_kind: str,
        subgroup: list[str] | None,
        input_artifact_id: str,
        input_artifact_checksum: str,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
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

        params = {
            "source_a": source_a,
            "source_b": source_b,
            "value_kind": value_kind,
            "n_input": len(values_a),
            "input_artifact_id": input_artifact_id,
            "input_artifact_checksum": input_artifact_checksum,
        }
        spec = build_spec_from_request(
            "measurement_validation",
            corpus_id,
            analysis_parameters={
                "source_a": source_a,
                "source_b": source_b,
                "value_kind": value_kind,
                "input_artifact_id": input_artifact_id,
                "input_artifact_checksum": input_artifact_checksum,
            },
        )
        run = AnalysisRun(
            project_id=corpus.project_id,
            corpus_id=corpus_id,
            run_type=AnalysisRunType.MEASUREMENT_VALIDATION.value,
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
                    "n_paired": comparison.get("n_paired"),
                    "agreement_rate": comparison.get("agreement_rate"),
                    "correlation": comparison.get("correlation"),
                }
            ),
            results_json=dumps(comparison),
            created_by=user_id,
            started_at=_utcnow(),
            completed_at=_utcnow(),
        )
        return await self.repo.create_run(run)
