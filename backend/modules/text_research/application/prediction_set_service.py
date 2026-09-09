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

    async def browse_predictions(
        self,
        prediction_set_id: str,
        *,
        user_id: str,
        limit: int,
        offset: int,
        predicted_label: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        min_uncertainty: float | None = None,
        max_uncertainty: float | None = None,
        review_status: str | None = None,
        human_disagreement: bool | None = None,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        """Server-side paging projection; prediction, human, and gold stay separate."""
        prediction_set = await self.get_prediction_set_or_404(prediction_set_id, user_id=user_id)
        unit_ids = list(loads(prediction_set.metadata_json, {}).get("unit_ids") or [])
        predictions = await self.repo.list_predictions_for_units(
            prediction_set.trained_model_id, unit_ids
        )
        annotations = await self.repo.list_annotations_for_units(unit_ids, campaign_id=campaign_id)
        adjudications = await self.repo.list_adjudications_for_units(
            unit_ids, campaign_id=campaign_id
        )
        annotations_by_unit: dict[str, list[Any]] = {}
        adjudications_by_unit: dict[str, list[Any]] = {}
        for annotation in annotations:
            annotations_by_unit.setdefault(annotation.text_unit_id, []).append(annotation)
        for adjudication in adjudications:
            adjudications_by_unit.setdefault(adjudication.text_unit_id, []).append(adjudication)

        rows: list[dict[str, Any]] = []
        for prediction in predictions:
            labels = [str(label) for label in loads(prediction.predicted_labels_json, [])]
            scores = {str(k): float(v) for k, v in loads(prediction.scores_json, {}).items()}
            confidence = max(scores.values(), default=None)
            unit_annotations = annotations_by_unit.get(prediction.text_unit_id, [])
            unit_adjudications = adjudications_by_unit.get(prediction.text_unit_id, [])
            values_by_label: dict[str, set[str]] = {}
            for annotation in unit_annotations:
                values_by_label.setdefault(annotation.label_id, set()).add(annotation.value)
            has_disagreement = any(len(values) > 1 for values in values_by_label.values())
            resolved_review_status = (
                "adjudicated"
                if unit_adjudications
                else "annotated"
                if unit_annotations
                else "unreviewed"
            )
            if predicted_label and predicted_label not in labels:
                continue
            if min_confidence is not None and (confidence is None or confidence < min_confidence):
                continue
            if max_confidence is not None and (confidence is None or confidence > max_confidence):
                continue
            if min_uncertainty is not None and (
                prediction.uncertainty is None or prediction.uncertainty < min_uncertainty
            ):
                continue
            if max_uncertainty is not None and (
                prediction.uncertainty is None or prediction.uncertainty > max_uncertainty
            ):
                continue
            if review_status and resolved_review_status != review_status:
                continue
            if human_disagreement and not has_disagreement:
                continue
            rows.append(
                {
                    "prediction": prediction,
                    "human_annotations": unit_annotations,
                    "adjudications": unit_adjudications,
                    "review_status": resolved_review_status,
                    "human_disagreement": has_disagreement,
                    "provenance_layers": {
                        "prediction_set_id": prediction_set.id,
                        "trained_model_id": prediction_set.trained_model_id,
                        "dataset_snapshot_id": prediction_set.dataset_snapshot_id,
                        "campaign_id": campaign_id,
                    },
                }
            )
        return {"items": rows[offset : offset + limit], "total": len(rows)}

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
