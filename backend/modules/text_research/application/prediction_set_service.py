"""Persist and retrieve ``PredictionSet`` headers for classifier prediction runs."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import ResearchArtifactKind
from backend.modules.text_research.domain.models import (
    AnalysisRun,
    PredictionSet,
    TrainedModel,
    dumps,
    loads,
)


class PredictionSetService(ResearchAccessMixin):
    async def create_from_run(
        self,
        *,
        run: AnalysisRun,
        model: TrainedModel,
        unit_ids: list[str],
        created_by: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> PredictionSet:
        metadata: dict[str, Any] = {
            "kind": ResearchArtifactKind.PREDICTION_SET.value,
            "unit_ids": unit_ids,
            "unit_count": len(unit_ids),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        prediction_set = PredictionSet(
            project_id=run.project_id,
            corpus_id=model.corpus_id,
            trained_model_id=model.id,
            model_version=model.version,
            dataset_snapshot_id=model.training_dataset_snapshot_id,
            analysis_run_id=run.id,
            created_by=created_by,
            metadata_json=dumps(metadata),
        )
        return await self.repo.create_prediction_set(prediction_set)

    async def get(self, prediction_set_id: str, *, user_id: str) -> dict[str, Any]:
        prediction_set = await self.get_prediction_set_or_404(prediction_set_id, user_id=user_id)
        metadata = loads(prediction_set.metadata_json, {})
        unit_ids = list(metadata.get("unit_ids") or [])
        predictions = await self.repo.list_predictions_for_units(
            prediction_set.trained_model_id,
            unit_ids,
        )
        return {
            "prediction_set": prediction_set,
            "metadata": metadata,
            "predictions": predictions,
        }

    async def list(
        self,
        corpus_id: str,
        *,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PredictionSet], int]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return await self.repo.list_prediction_sets_for_corpus(
            corpus_id,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def metadata(prediction_set: PredictionSet) -> dict[str, Any]:
        return loads(prediction_set.metadata_json, {})
