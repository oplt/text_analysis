"""Whole-corpus (or filtered subset) prediction using a validated `TrainedModel`.

The fitted vectorizer is only ever `.transform()`-ed here — never refit —
matching the leakage-prevention contract established during training.
Predictions are persisted separately from human annotations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.quantitative_analysis_service import (
    _apply_document_filters,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure.classifiers import predict_with_uncertainty


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PredictionService(ResearchAccessMixin):
    async def predict(
        self,
        model_id: str,
        *,
        user_id: str,
        unit_type: str,
        only_unannotated: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        params = {
            "model_id": model_id,
            "unit_type": unit_type,
            "only_unannotated": only_unannotated,
            "filters": filters or {},
        }
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=model.project_id,
                corpus_id=model.corpus_id,
                run_type=AnalysisRunType.CLASSIFIER_PREDICTION.value,
                status=AnalysisRunStatus.QUEUED.value,
                parameters_json=dumps(params),
                created_by=user_id,
            )
        )
        await self.db.commit()
        await self.execute_prediction(run.id)
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_prediction(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run, status=AnalysisRunStatus.RUNNING.value, progress_stage="predicting", started_at=_utcnow()
        )
        await self.db.commit()

        try:
            model = await self.repo.get_model(params["model_id"])
            if model is None:
                raise ValueError("Trained model no longer exists")

            documents = await self.repo.list_documents(model.corpus_id)
            filtered_docs = _apply_document_filters(documents, params.get("filters") or {})
            doc_ids = [d.id for d in filtered_docs] if params.get("filters") else None
            units = await self.repo.list_text_units_for_corpus(
                model.corpus_id, unit_type=params["unit_type"], document_ids=doc_ids
            )

            if params.get("only_unannotated"):
                annotated_unit_ids = {
                    a.text_unit_id for a in await self.repo.list_annotations_for_corpus(model.corpus_id)
                }
                units = [u for u in units if u.id not in annotated_unit_ids]

            vectorizer = model_storage.load_artifact(model.vectorizer_artifact_path)
            classifier = model_storage.load_artifact(model.model_artifact_path)
            label_names = loads(model.label_ids_json, [])
            texts = [u.text for u in units]

            predictions = (
                predict_with_uncertainty(classifier, vectorizer, texts, task_type=model.task_type)
                if texts
                else []
            )

            for unit, prediction in zip(units, predictions, strict=True):
                if model.task_type == "multilabel":
                    predicted_binary = prediction["prediction"]
                    predicted_labels = [
                        label_names[i] for i, flag in enumerate(predicted_binary) if flag
                    ]
                    probabilities = prediction.get("probabilities")
                    scores = (
                        {label_names[i]: float(probabilities[i]) for i in range(len(label_names))}
                        if probabilities is not None
                        else {}
                    )
                else:
                    predicted_labels = [str(prediction["prediction"])]
                    scores = {}
                uncertainty = prediction.get("uncertainty")
                await self.repo.upsert_prediction(
                    trained_model_id=model.id,
                    text_unit_id=unit.id,
                    predicted_labels_json=dumps(predicted_labels),
                    scores_json=dumps(scores),
                    uncertainty=uncertainty,
                )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps({"units_predicted": len(units)}),
                results_json=dumps({"unit_count": len(units)}),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run, status=AnalysisRunStatus.FAILED.value, completed_at=_utcnow(), error_message=str(exc)
            )
            await self.db.commit()
            raise

        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def list_predictions(
        self,
        model_id: str,
        *,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        order_by_uncertainty: bool = False,
    ):
        model = await self.get_model_or_404(model_id, user_id=user_id)
        predictions, _total = await self.repo.list_predictions_for_model(
            model.id, limit=limit, offset=offset, order_by_uncertainty=order_by_uncertainty
        )
        return predictions
