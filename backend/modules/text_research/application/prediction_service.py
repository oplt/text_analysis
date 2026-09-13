"""Whole-corpus (or filtered subset) prediction using a validated `TrainedModel`.

The fitted vectorizer is only ever `.transform()`-ed here — never refit —
matching the leakage-prevention contract established during training.
Predictions are persisted separately from human annotations.

Large corpora are processed via frozen selection + paged iteration so peak
memory stays bounded; annotation membership is snapshotted at run start so
mid-run annotation changes cannot alter the candidate set.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.quantitative_analysis_service import (
    _apply_document_filters,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TextUnit, dumps, loads
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure.classifiers import predict_with_uncertainty
from backend.modules.text_research.infrastructure.out_of_core import (
    resolve_batch_size,
    should_use_out_of_core,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _prediction_row(
    *,
    model_id: str,
    unit_id: str,
    task_type: str,
    label_names: list[str],
    prediction: dict[str, Any],
) -> dict[str, object]:
    if task_type == "multilabel":
        predicted_binary = prediction["prediction"]
        predicted_labels = [label_names[i] for i, flag in enumerate(predicted_binary) if flag]
        probabilities = prediction.get("probabilities")
        scores = (
            {label_names[i]: float(probabilities[i]) for i in range(len(label_names))}
            if probabilities is not None
            else {}
        )
    else:
        predicted_index = int(prediction["prediction"])
        predicted_label = (
            label_names[predicted_index]
            if label_names and 0 <= predicted_index < len(label_names)
            else str(predicted_index)
        )
        predicted_labels = [predicted_label]
        probabilities = prediction.get("probabilities")
        probability = prediction.get("probability")
        if task_type == "multiclass" and probabilities is not None:
            scores = {
                label_names[i]: float(probabilities[i])
                for i in range(min(len(label_names), len(probabilities)))
            }
        elif task_type == "binary" and probability is not None and len(label_names) > 1:
            scores = {label_names[1]: float(probability)}
        else:
            scores = {}
    return {
        "trained_model_id": model_id,
        "text_unit_id": unit_id,
        "predicted_labels_json": dumps(predicted_labels),
        "scores_json": dumps(scores),
        "uncertainty": prediction.get("uncertainty"),
    }


SELECTION_SNAPSHOT_VERSION = 1


def freeze_prediction_selection(
    *,
    corpus_id: str,
    unit_type: str,
    only_unannotated: bool,
    filters: dict[str, Any] | None,
    document_ids: list[str] | None,
    annotated_unit_ids: set[str] | None,
    corpus_revision_hash: str | None = None,
) -> dict[str, Any]:
    """Immutable selection identity for reproducible / restartable prediction.

    Empty annotation sets are first-class: callers must distinguish ``None``
    (annotation membership not applicable) from ``set()`` (frozen empty set)
    via the ``annotated_unit_ids`` key on the persisted snapshot, never via
    truthiness of the ID list itself.
    """
    # Do not use ``annotated_unit_ids or ()`` — empty set is falsy but intentional.
    annotated_sorted = sorted(annotated_unit_ids) if annotated_unit_ids is not None else None
    annotated_hash = (
        hashlib.sha256(",".join(annotated_sorted).encode("utf-8")).hexdigest()
        if annotated_sorted is not None
        else None
    )
    # ``None`` means unfiltered corpus scope; ``[]`` means explicitly no documents.
    doc_ids = sorted(document_ids) if document_ids is not None else None
    doc_hash = (
        hashlib.sha256(",".join(doc_ids).encode("utf-8")).hexdigest()
        if doc_ids is not None
        else "ALL"
    )
    payload: dict[str, Any] = {
        "snapshot_version": SELECTION_SNAPSHOT_VERSION,
        "frozen": True,
        "corpus_id": corpus_id,
        "unit_type": unit_type,
        "only_unannotated": bool(only_unannotated),
        "filters": filters or {},
        "document_ids": doc_ids,
        "document_ids_hash": doc_hash,
        "annotated_unit_ids_hash": annotated_hash,
        "annotated_unit_count": len(annotated_sorted) if annotated_sorted is not None else None,
        "corpus_revision_hash": corpus_revision_hash,
    }
    payload["selection_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def selection_snapshot_is_frozen(params: dict[str, Any]) -> bool:
    """True when a selection snapshot was persisted (including empty freezes)."""
    snapshot = params.get("selection_snapshot")
    return isinstance(snapshot, dict) and (
        snapshot.get("frozen") is True or "selection_hash" in snapshot
    )


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
        from backend.modules.text_research.workers import queue_prediction

        queue_prediction(run_id=run.id, user_id=user_id)
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_prediction(self, run_id: str) -> AnalysisRun:
        from backend.modules.text_research.application.run_lifecycle import (
            TERMINAL_RUN_STATUSES,
            RunCancelledError,
            complete_if_active,
            ensure_not_cancelled,
            fail_if_active,
        )

        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in TERMINAL_RUN_STATUSES:
            return run
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="loading_units",
            started_at=_utcnow(),
        )
        await self.db.commit()

        prediction_set = None
        try:
            run = await ensure_not_cancelled(self.repo, run)
            model = await self.repo.get_model(params["model_id"])
            if model is None:
                raise ValueError("Trained model no longer exists")

            filters = params.get("filters") or {}
            documents = await self.repo.list_documents(model.corpus_id)
            filtered_docs = _apply_document_filters(documents, filters)
            live_doc_ids = [d.id for d in filtered_docs] if filters else None

            frozen_snapshot = (
                params.get("selection_snapshot") if selection_snapshot_is_frozen(params) else None
            )

            # Freeze only-unannotated membership at run start for reproducibility.
            # Detect snapshot presence structurally — empty ``annotated_unit_ids: []``
            # must still win over a live re-query after new annotations appear.
            annotated_unit_ids: set[str] | None = None
            if params.get("only_unannotated"):
                if isinstance(frozen_snapshot, dict) and "annotated_unit_ids" in frozen_snapshot:
                    annotated_unit_ids = set(frozen_snapshot["annotated_unit_ids"] or ())
                else:
                    annotated_unit_ids = await self.repo.list_annotated_text_unit_ids(
                        model.corpus_id
                    )

            if isinstance(frozen_snapshot, dict) and "document_ids" in frozen_snapshot:
                doc_ids = frozen_snapshot.get("document_ids")
            else:
                doc_ids = live_doc_ids

            corpus_revision = getattr(model, "training_dataset_snapshot_id", None)
            if isinstance(frozen_snapshot, dict) and frozen_snapshot.get("corpus_revision_hash"):
                corpus_revision = frozen_snapshot.get("corpus_revision_hash")

            selection = (
                dict(frozen_snapshot)
                if isinstance(frozen_snapshot, dict) and frozen_snapshot.get("selection_hash")
                else freeze_prediction_selection(
                    corpus_id=model.corpus_id,
                    unit_type=params["unit_type"],
                    only_unannotated=bool(params.get("only_unannotated")),
                    filters=filters,
                    document_ids=doc_ids,
                    annotated_unit_ids=annotated_unit_ids,
                    corpus_revision_hash=str(corpus_revision) if corpus_revision else None,
                )
            )
            # Persist snapshot once (restart-safe). Always record annotated_unit_ids
            # when only_unannotated, including the empty list.
            if not selection_snapshot_is_frozen(params):
                snapshot = dict(selection)
                snapshot["frozen"] = True
                snapshot["snapshot_version"] = SELECTION_SNAPSHOT_VERSION
                if annotated_unit_ids is not None:
                    snapshot["annotated_unit_ids"] = sorted(annotated_unit_ids)
                params = {**params, "selection_snapshot": snapshot}
                await self.repo.update_run(run, parameters_json=dumps(params))
                await self.db.commit()

            exclude_ids = annotated_unit_ids if params.get("only_unannotated") else None
            unit_count = 0
            predicted_unit_ids: list[str] = []
            units_predicted = 0

            # Bound batch size from a count without materializing all units.
            approx_count = await self.repo.count_text_units_for_corpus(
                model.corpus_id, unit_type=params["unit_type"]
            )
            batch_size = resolve_batch_size(approx_count)
            use_batches = should_use_out_of_core(approx_count) or approx_count > batch_size

            run = await ensure_not_cancelled(self.repo, run)
            vectorizer = model_storage.load_artifact(model.vectorizer_artifact_path)
            classifier = model_storage.load_artifact(model.model_artifact_path)
            label_names = loads(model.label_ids_json, [])
            training_metrics = loads(model.metrics_json, {})
            thresholds = training_metrics.get("thresholds")
            from backend.modules.text_research.application.prediction_set_service import (
                PredictionSetService,
            )

            prediction_set_service = PredictionSetService(self.db)
            prediction_set = await prediction_set_service.create_draft_from_run(
                run=run,
                model=model,
                created_by=run.created_by,
                extra_metadata={"selection_snapshot": params.get("selection_snapshot")},
            )

            await self.repo.update_run(run, progress_stage="predicting")
            await self.db.commit()

            async def _persist_batch(
                batch_units: list[TextUnit], batch_predictions: list[dict[str, Any]]
            ) -> None:
                nonlocal units_predicted
                rows = [
                    _prediction_row(
                        model_id=model.id,
                        unit_id=unit.id,
                        task_type=model.task_type,
                        label_names=label_names,
                        prediction=prediction,
                    )
                    for unit, prediction in zip(batch_units, batch_predictions, strict=True)
                ]
                for row in rows:
                    row["prediction_set_id"] = prediction_set.id
                await self.repo.bulk_upsert_predictions(rows)
                # Commit each staged batch so a later discard/publish is a small
                # transaction, not one giant uncommitted write.
                await self.db.commit()
                predicted_unit_ids.extend(unit.id for unit in batch_units)
                units_predicted += len(batch_units)

            async for page in self.repo.iter_text_units_for_corpus(
                model.corpus_id,
                unit_type=params["unit_type"],
                document_ids=doc_ids,
                batch_size=batch_size,
            ):
                run = await ensure_not_cancelled(self.repo, run)
                if exclude_ids is not None:
                    page = [unit for unit in page if unit.id not in exclude_ids]
                if not page:
                    continue
                unit_count += len(page)
                texts = [unit.text for unit in page]
                batch_predictions = predict_with_uncertainty(
                    classifier,
                    vectorizer,
                    texts,
                    task_type=model.task_type,
                    label_names=label_names,
                    thresholds=thresholds,
                )
                await _persist_batch(list(page), batch_predictions)

            run = await ensure_not_cancelled(self.repo, run)
            await self.repo.update_run(run, progress_stage="saving")

            run = await ensure_not_cancelled(self.repo, run)
            completed = await complete_if_active(
                self.repo,
                run,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "units_predicted": units_predicted,
                        "units_selected": unit_count,
                        "selection_hash": selection.get("selection_hash"),
                    }
                ),
                results_json=dumps(
                    {
                        "unit_count": units_predicted,
                        "prediction_set_id": prediction_set.id,
                        "batch_size": batch_size if use_batches else max(unit_count, 1),
                        "selection_hash": selection.get("selection_hash"),
                    }
                ),
            )
            if completed is None:
                await self.repo.discard_prediction_set(prediction_set)
                await self.db.commit()
                refreshed = await self.repo.get_run(run_id)
                assert refreshed is not None
                return refreshed
            await prediction_set_service.publish(prediction_set, unit_ids=predicted_unit_ids)
            await self.db.commit()
        except RunCancelledError:
            if prediction_set is not None:
                await self.repo.discard_prediction_set(prediction_set)
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            if prediction_set is not None:
                await self.repo.discard_prediction_set(prediction_set)
            await fail_if_active(
                self.repo,
                run,
                completed_at=_utcnow(),
                error_message=str(exc),
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
