"""Dictionaries and training-dataset/classifier routes
(LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.route_serializers import (
    _dictionary_response,
    _loads,
    _model_response,
    _prediction_item_response,
    _prediction_set_response,
    _run_response,
    _snapshot_response,
)
from backend.modules.text_research.api.schemas import (
    ActiveLearningAssignRequest,
    AnalysisRunResponse,
    ClassifierPredictRequest,
    ClassifierTrainRequest,
    DatasetFreezeRequest,
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DictionaryCreate,
    DictionaryResponse,
    DictionaryUpdate,
    DriftMonitoringRequest,
    ModelLifecycleEventResponse,
    ModelLifecycleUpdateRequest,
    PredictionSetDetailResponse,
    PredictionSetPredictionRowResponse,
    PredictionSetPredictionsPageResponse,
    PredictionSetResponse,
    TrainedModelResponse,
    TrainingDatasetSnapshotResponse,
)
from backend.modules.text_research.application.active_learning_service import ActiveLearningService
from backend.modules.text_research.application.campaign_service import AnnotationCampaignService
from backend.modules.text_research.application.classification_service import ClassificationService
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.application.dictionary_service import DictionaryService
from backend.modules.text_research.application.drift_service import DriftService
from backend.modules.text_research.application.model_lifecycle_service import ModelLifecycleService
from backend.modules.text_research.application.prediction_service import PredictionService
from backend.modules.text_research.application.prediction_set_service import PredictionSetService

router = APIRouter()


# ------------------------------------------------------------------
# Dictionaries
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/dictionaries", response_model=DictionaryResponse, status_code=201
)
async def create_dictionary(
    project_id: str,
    body: DictionaryCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.terms is None and body.hierarchy is None:
        raise HTTPException(
            status_code=400,
            detail="Provide terms or hierarchy (dictionaries are user-defined only)",
        )
    dictionary = await DictionaryService(db).create_dictionary(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        language=body.language,
        terms=body.terms,
        hierarchy=body.hierarchy,
        exclusions=body.exclusions,
    )
    return _dictionary_response(dictionary)


@router.get("/projects/{project_id}/dictionaries", response_model=list[DictionaryResponse])
async def list_dictionaries(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dictionaries = await DictionaryService(db).list_dictionaries(
        project_id=project_id, user_id=current_user.id
    )
    return [_dictionary_response(d) for d in dictionaries]


@router.get("/dictionaries/{dictionary_id}", response_model=DictionaryResponse)
async def get_dictionary(
    dictionary_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dictionary = await DictionaryService(db).get_dictionary(dictionary_id, user_id=current_user.id)
    return _dictionary_response(dictionary)


@router.patch("/dictionaries/{dictionary_id}", response_model=DictionaryResponse)
async def update_dictionary(
    dictionary_id: str,
    body: DictionaryUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dictionary = await DictionaryService(db).update_dictionary(
        dictionary_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        language=body.language,
        terms=body.terms,
        hierarchy=body.hierarchy,
        exclusions=body.exclusions,
    )
    return _dictionary_response(dictionary)


@router.post(
    "/dictionaries/{dictionary_id}/versions", response_model=DictionaryResponse, status_code=201
)
async def create_dictionary_version(
    dictionary_id: str,
    new_version: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dictionary = await DictionaryService(db).create_version(
        dictionary_id, user_id=current_user.id, new_version=new_version
    )
    return _dictionary_response(dictionary)


# ------------------------------------------------------------------
# Training datasets & classifiers
# ------------------------------------------------------------------


@router.post("/classifiers/dataset-preview", response_model=DatasetPreviewResponse)
async def dataset_preview(
    body: DatasetPreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await DatasetBuilderService(db).preview(
        body.corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        annotation_source=body.annotation_source,
        selected_annotator_id=body.selected_annotator_id,
        minimum_agreement=body.minimum_agreement,
        annotation_campaign_id=body.annotation_campaign_id,
    )


@router.post(
    "/classifiers/dataset-snapshots",
    response_model=TrainingDatasetSnapshotResponse,
    status_code=201,
)
async def freeze_dataset(
    body: DatasetFreezeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    snapshot = await DatasetBuilderService(db).freeze(
        body.corpus_id,
        user_id=current_user.id,
        name=body.name,
        unit_type=body.unit_type,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        annotation_source=body.annotation_source,
        selected_annotator_id=body.selected_annotator_id,
        minimum_agreement=body.minimum_agreement,
        annotation_campaign_id=body.annotation_campaign_id,
    )
    return _snapshot_response(snapshot)


@router.get(
    "/projects/{project_id}/dataset-snapshots", response_model=list[TrainingDatasetSnapshotResponse]
)
async def list_dataset_snapshots(
    project_id: str,
    corpus_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    snapshots = await DatasetBuilderService(db).list_snapshots(
        project_id=project_id, user_id=current_user.id, corpus_id=corpus_id
    )
    return [_snapshot_response(s) for s in snapshots]


@router.get("/dataset-snapshots/{snapshot_id}", response_model=TrainingDatasetSnapshotResponse)
async def get_dataset_snapshot(
    snapshot_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    snapshot = await DatasetBuilderService(db).get_snapshot(snapshot_id, user_id=current_user.id)
    return _snapshot_response(snapshot)


@router.post("/classifiers/train", response_model=AnalysisRunResponse, status_code=202)
async def train_classifier(
    body: ClassifierTrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await ClassificationService(db).train(
        user_id=current_user.id,
        snapshot_id=body.snapshot_id,
        algorithm=body.algorithm,
        task_type=body.task_type,
        preprocessing_profile_id=body.preprocessing_profile_id,
        vectorizer=body.vectorizer,
        use_word_ngrams=body.use_word_ngrams,
        ngram_min=body.ngram_min,
        ngram_max=body.ngram_max,
        use_char_ngrams=body.use_char_ngrams,
        char_ngram_min=body.char_ngram_min,
        char_ngram_max=body.char_ngram_max,
        min_df=body.min_df,
        max_df=body.max_df,
        max_features=body.max_features,
        feature_selection_method=body.feature_selection_method,
        feature_selection_k=body.feature_selection_k,
        feature_selection_percentile=body.feature_selection_percentile,
        class_weight=body.class_weight,
        regularization_c=body.regularization_c,
        nb_alpha=body.nb_alpha,
        sgd_loss=body.sgd_loss,
        test_size=body.test_size,
        val_size=body.val_size,
        random_seed=body.random_seed,
        tune_hyperparameters=body.tune_hyperparameters,
        hyperparameter_search_type=body.hyperparameter_search_type,
        hyperparameter_param_grid=body.hyperparameter_param_grid,
        hyperparameter_n_iter=body.hyperparameter_n_iter,
        hyperparameter_scoring=body.hyperparameter_scoring,
        tune_thresholds=body.tune_thresholds,
        n_bootstrap=body.n_bootstrap,
        ci_confidence_level=body.ci_confidence_level,
        calibration_method=body.calibration_method,
        validation_strategy=body.validation_strategy,
        nested_cv_outer_splits=body.nested_cv_outer_splits,
        nested_cv_inner_splits=body.nested_cv_inner_splits,
        embedding_provider=body.embedding_provider,
        embedding_model_name=body.embedding_model_name,
        embedding_model_revision=body.embedding_model_revision,
        threshold_objective=body.threshold_objective,
        threshold_utility_tp=body.threshold_utility_tp,
        threshold_utility_tn=body.threshold_utility_tn,
        threshold_utility_fp=body.threshold_utility_fp,
        threshold_utility_fn=body.threshold_utility_fn,
        name=body.name,
        run_async=body.run_async,
    )
    return _run_response(run)


@router.get("/projects/{project_id}/classifiers", response_model=list[TrainedModelResponse])
async def list_classifiers(
    project_id: str,
    corpus_id: str | None = None,
    lifecycle_status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    models = await ClassificationService(db).list_models(
        project_id=project_id,
        user_id=current_user.id,
        corpus_id=corpus_id,
        lifecycle_status=lifecycle_status,
    )
    return [_model_response(m) for m in models]


@router.get("/projects/{project_id}/models", response_model=list[TrainedModelResponse])
async def list_models(
    project_id: str,
    corpus_id: str | None = None,
    lifecycle_status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    models = await ModelLifecycleService(db).list_by_status(
        project_id,
        user_id=current_user.id,
        status=lifecycle_status,
        corpus_id=corpus_id,
    )
    return [_model_response(m) for m in models]


@router.patch("/models/{model_id}/lifecycle", response_model=TrainedModelResponse)
async def update_model_lifecycle(
    model_id: str,
    body: ModelLifecycleUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model = await ModelLifecycleService(db).set_status(
        model_id,
        user_id=current_user.id,
        status=body.status,
        notes=body.notes,
        deprecate_others=body.deprecate_others,
    )
    return _model_response(model)


@router.get(
    "/models/{model_id}/lifecycle-events",
    response_model=list[ModelLifecycleEventResponse],
)
async def list_model_lifecycle_events(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    events = await ModelLifecycleService(db).list_events(model_id, user_id=current_user.id)
    return [
        ModelLifecycleEventResponse(
            id=event.id,
            model_id=event.model_id,
            from_status=event.from_status,
            to_status=event.to_status,
            actor_id=event.actor_id,
            reason=event.reason,
            run_id=event.run_id,
            metadata=_loads(event.metadata_json, {}),
            created_at=event.created_at,
        )
        for event in events
    ]


@router.get("/classifiers/{model_id}", response_model=TrainedModelResponse)
async def get_classifier(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model = await ClassificationService(db).get_model(model_id, user_id=current_user.id)
    return _model_response(model)


@router.get("/classifiers/{model_id}/coefficients")
async def get_classifier_coefficients(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ClassificationService(db).get_coefficients(model_id, user_id=current_user.id)


@router.post("/classifiers/{model_id}/clone")
async def clone_classifier_config(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ClassificationService(db).clone_config(model_id, user_id=current_user.id)


@router.post("/classifiers/{model_id}/predict", response_model=AnalysisRunResponse, status_code=202)
async def predict_classifier(
    model_id: str,
    body: ClassifierPredictRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await PredictionService(db).predict(
        model_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        only_unannotated=body.only_unannotated,
        filters=body.filters,
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/monitoring/drift")
async def compare_classifier_drift(
    corpus_id: str,
    body: DriftMonitoringRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.baseline_prediction_set_id and body.current_prediction_set_id:
        return await DriftService(db).compare_prediction_sets(
            corpus_id,
            user_id=current_user.id,
            mode=body.mode,
            baseline_prediction_set_id=body.baseline_prediction_set_id,
            current_prediction_set_id=body.current_prediction_set_id,
        )
    if body.baseline is None or body.current is None:
        raise HTTPException(
            status_code=422,
            detail="baseline/current aggregates or both PredictionSet IDs are required",
        )
    report = await DriftService(db).compare_distributions(
        corpus_id,
        user_id=current_user.id,
        baseline=body.baseline.model_dump(exclude_none=True),
        current=body.current.model_dump(exclude_none=True),
        baseline_run_id=body.baseline_run_id,
        current_run_id=body.current_run_id,
    )
    return report


@router.get("/classifiers/{model_id}/predictions")
async def list_predictions(
    model_id: str,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await PredictionService(db).list_predictions(
        model_id, user_id=current_user.id, limit=limit, offset=offset
    )


@router.get("/prediction-sets/{prediction_set_id}", response_model=PredictionSetDetailResponse)
async def get_prediction_set(
    prediction_set_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    payload = await PredictionSetService(db).get(
        prediction_set_id,
        user_id=current_user.id,
    )
    prediction_set = payload["prediction_set"]
    return PredictionSetDetailResponse(
        **_prediction_set_response(prediction_set).model_dump(),
        predictions=[
            _prediction_item_response(prediction) for prediction in payload["predictions"]
        ],
    )


@router.get(
    "/prediction-sets/{prediction_set_id}/predictions",
    response_model=PredictionSetPredictionsPageResponse,
)
async def browse_prediction_set_predictions(
    prediction_set_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    predicted_label: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    min_uncertainty: float | None = None,
    max_uncertainty: float | None = None,
    review_status: str | None = Query(default=None, pattern="^(unreviewed|annotated|adjudicated)$"),
    human_disagreement: bool | None = None,
    campaign_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    page = await PredictionSetService(db).browse_predictions(
        prediction_set_id,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        predicted_label=predicted_label,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        min_uncertainty=min_uncertainty,
        max_uncertainty=max_uncertainty,
        review_status=review_status,
        human_disagreement=human_disagreement,
        campaign_id=campaign_id,
    )
    return PredictionSetPredictionsPageResponse(
        items=[
            PredictionSetPredictionRowResponse(
                prediction=_prediction_item_response(row["prediction"]),
                human_annotations=[
                    {
                        "id": annotation.id,
                        "text_unit_id": annotation.text_unit_id,
                        "label_id": annotation.label_id,
                        "value": annotation.value,
                        "annotator_id": annotation.annotator_id,
                        "campaign_id": annotation.campaign_id,
                    }
                    for annotation in row["human_annotations"]
                ],
                adjudications=[
                    {
                        "id": adjudication.id,
                        "text_unit_id": adjudication.text_unit_id,
                        "label_id": adjudication.label_id,
                        "final_value": adjudication.final_value,
                        "campaign_id": adjudication.campaign_id,
                    }
                    for adjudication in row["adjudications"]
                ],
                review_status=row["review_status"],
                human_disagreement=row["human_disagreement"],
                provenance_layers=row["provenance_layers"],
            )
            for row in page["items"]
        ],
        total=page["total"],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/corpora/{corpus_id}/prediction-sets",
    response_model=list[PredictionSetResponse],
)
async def list_prediction_sets(
    corpus_id: str,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items, _total = await PredictionSetService(db).list(
        corpus_id,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    return [_prediction_set_response(item) for item in items]


@router.get("/classifiers/{model_id}/active-learning/queue")
async def list_uncertain_predictions(
    model_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    content_mode: str = Query(
        default="snippet",
        pattern="^(snippet|full)$",
        description="Return truncated text previews (snippet) or full unit text (full).",
    ),
    campaign_id: str | None = None,
    text_unit_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return persisted predictions ordered from most to least uncertain.

    This endpoint deliberately returns model output separately from human
    annotations; users explicitly choose which units enter an annotation task.

    When ``campaign_id`` or ``text_unit_id`` refers to a blind campaign for the
    current annotator, the queue is empty — predictions must not be fetched.
    """
    from backend.core.text_snippet import text_snippet

    if campaign_id:
        campaign = await AnnotationCampaignService(db).get_campaign(
            campaign_id, user_id=current_user.id
        )
        if campaign.blind_mode:
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
    if text_unit_id:
        policy = await AnnotationCampaignService(db).blind_policy_for_annotator_unit(
            text_unit_id=text_unit_id, annotator_id=current_user.id
        )
        if policy.get("hide_model_predictions"):
            return {"items": [], "total": 0, "limit": limit, "offset": offset}

    rows, total = await ActiveLearningService(db).uncertain_queue(
        model_id, user_id=current_user.id, limit=limit, offset=offset
    )

    def unit_text(raw: str) -> str:
        if content_mode == "full":
            return raw
        return text_snippet(raw)

    items = [
        {
            "prediction": {
                "id": prediction.id,
                "trained_model_id": prediction.trained_model_id,
                "text_unit_id": prediction.text_unit_id,
                "predicted_labels": _loads(prediction.predicted_labels_json, []),
                "scores": _loads(prediction.scores_json, {}),
                "uncertainty": prediction.uncertainty,
                "created_at": prediction.created_at,
            },
            "text_unit": {
                "id": unit.id,
                "corpus_document_id": unit.corpus_document_id,
                "unit_type": unit.unit_type,
                "position": unit.position,
                "text": unit_text(unit.text),
            },
        }
        for row in rows
        if (prediction := row["prediction"]) and (unit := row["text_unit"])
    ]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("/classifiers/{model_id}/active-learning/assign", status_code=201)
async def assign_uncertain_predictions(
    model_id: str,
    body: ActiveLearningAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = await ActiveLearningService(db).send_to_annotation(
        model_id,
        user_id=current_user.id,
        text_unit_ids=body.text_unit_ids,
        annotator_ids=body.annotator_ids,
    )
    return [
        {
            "id": task.id,
            "text_unit_id": task.text_unit_id,
            "annotator_id": task.annotator_id,
            "status": task.status,
            "assigned_at": task.assigned_at,
            "completed_at": task.completed_at,
        }
        for task in tasks
    ]
