"""Shared response serializers for text-research API routers (LATEST-017)."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.api.schemas import (
    AnalysisRunResponse,
    AnnotationLabelResponse,
    CleaningProfileResponse,
    CorpusDocumentResponse,
    DictionaryResponse,
    ModelPredictionItemResponse,
    PredictionSetResponse,
    PreprocessingProfileResponse,
    ResearchCorpusResponse,
    TrainedModelResponse,
    TrainingDatasetSnapshotResponse,
)
from backend.modules.text_research.domain.models import (
    AnalysisRun,
    AnnotationLabel,
    CleaningProfile,
    CorpusDocument,
    DictionaryDefinition,
    PredictionSet,
    PreprocessingProfile,
    ResearchCorpus,
    TrainedModel,
    TrainingDatasetSnapshot,
    loads,
)


def _loads(value: str | None, default: Any = None) -> Any:
    return loads(value, default)


def _corpus_response(corpus: ResearchCorpus) -> ResearchCorpusResponse:
    return ResearchCorpusResponse.model_validate(corpus)


def _document_response(document: CorpusDocument) -> CorpusDocumentResponse:
    return CorpusDocumentResponse(
        id=document.id,
        corpus_id=document.corpus_id,
        rag_document_id=document.rag_document_id,
        title=document.title,
        organization=document.organization,
        organization_type=document.organization_type,
        publication_year=document.publication_year,
        publication_type=document.publication_type,
        country=document.country,
        region=document.region,
        cultural_sphere=document.cultural_sphere,
        language=document.language,
        education_level=document.education_level,
        source_url=document.source_url,
        research_notes=document.research_notes,
        metadata_json=_loads(document.metadata_json),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _run_response(run: AnalysisRun) -> AnalysisRunResponse:
    from backend.modules.text_research.application.result_artifacts import (
        resolve_results_artifact_id,
    )
    from backend.modules.text_research.application.run_adapters import capability_for_run

    parameters = _loads(run.parameters_json)
    capability = capability_for_run(run, parameters if isinstance(parameters, dict) else {})
    return AnalysisRunResponse(
        id=run.id,
        project_id=run.project_id,
        corpus_id=run.corpus_id,
        run_type=run.run_type,
        status=run.status,
        run_version=int(getattr(run, "run_version", 1) or 1),
        evidence_revision_hash=getattr(run, "evidence_revision_hash", None),
        progress_stage=run.progress_stage,
        parameters=parameters,
        metrics=_loads(run.metrics_json),
        results=_loads(run.results_json),
        artifact_path=run.artifact_path,
        results_artifact_id=resolve_results_artifact_id(run),
        random_seed=run.random_seed,
        created_by=run.created_by,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        created_at=run.created_at,
        rerunnable=capability.rerunnable,
        rerun_block_reason=capability.block_reason,
        replayable=capability.replayable,
        exact_reproducible=capability.exact_reproducible,
        exact_reproduce_block_reason=capability.exact_reproduce_block_reason,
    )


def _run_event_name(
    current: AnalysisRunResponse,
    previous: AnalysisRunResponse | None,
) -> str:
    """Return the SSE event that describes the newest persisted run state."""
    from backend.modules.text_research.infrastructure.run_events import run_event_name

    prev = None
    if previous is not None:
        prev = {
            "status": previous.status,
            "progress_stage": previous.progress_stage,
            "artifact_path": previous.artifact_path,
        }
    return run_event_name(
        status=current.status,
        progress_stage=current.progress_stage,
        artifact_path=current.artifact_path,
        previous=prev,
    )


def _profile_response(profile: PreprocessingProfile) -> PreprocessingProfileResponse:
    return PreprocessingProfileResponse(
        id=profile.id,
        project_id=profile.project_id,
        name=profile.name,
        description=profile.description,
        config=_loads(profile.config_json, {}),
        created_by=profile.created_by,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _cleaning_profile_response(profile: CleaningProfile) -> CleaningProfileResponse:
    return CleaningProfileResponse(
        id=profile.id,
        project_id=profile.project_id,
        name=profile.name,
        description=profile.description,
        version=profile.version,
        config=_loads(profile.config_json, {}),
        created_by=profile.created_by,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _label_response(label: AnnotationLabel) -> AnnotationLabelResponse:
    return AnnotationLabelResponse(
        id=label.id,
        codebook_id=label.codebook_id,
        name=label.name,
        description=label.description,
        inclusion_criteria=label.inclusion_criteria,
        exclusion_criteria=label.exclusion_criteria,
        positive_examples=_loads(label.positive_examples_json),
        negative_examples=_loads(label.negative_examples_json),
        is_placeholder=label.is_placeholder,
        created_at=label.created_at,
    )


def _dictionary_response(dictionary: DictionaryDefinition) -> DictionaryResponse:
    from backend.modules.text_research.application.dictionary_service import DictionaryService

    spec = DictionaryService.get_spec(dictionary)
    return DictionaryResponse(
        id=dictionary.id,
        project_id=dictionary.project_id,
        name=dictionary.name,
        version=dictionary.version,
        description=dictionary.description,
        language=spec.language,
        terms=spec.flattened_terms(),
        hierarchy=DictionaryService.get_hierarchy(dictionary),
        exclusions=[e.to_dict() for e in spec.exclusions],
        format=spec.format,
        created_by=dictionary.created_by,
        created_at=dictionary.created_at,
    )


def _snapshot_response(snapshot: TrainingDatasetSnapshot) -> TrainingDatasetSnapshotResponse:
    return TrainingDatasetSnapshotResponse.model_validate(snapshot)


def _prediction_item_response(prediction) -> ModelPredictionItemResponse:
    return ModelPredictionItemResponse(
        id=prediction.id,
        trained_model_id=prediction.trained_model_id,
        text_unit_id=prediction.text_unit_id,
        predicted_labels=_loads(prediction.predicted_labels_json, []),
        scores=_loads(prediction.scores_json, {}),
        uncertainty=prediction.uncertainty,
        created_at=prediction.created_at,
    )


def _prediction_set_response(prediction_set: PredictionSet) -> PredictionSetResponse:
    from backend.modules.text_research.application.prediction_set_service import (
        PredictionSetService,
    )

    return PredictionSetResponse(
        id=prediction_set.id,
        project_id=prediction_set.project_id,
        corpus_id=prediction_set.corpus_id,
        trained_model_id=prediction_set.trained_model_id,
        model_version=prediction_set.model_version,
        dataset_snapshot_id=prediction_set.dataset_snapshot_id,
        analysis_run_id=prediction_set.analysis_run_id,
        created_by=prediction_set.created_by,
        created_at=prediction_set.created_at,
        metadata=PredictionSetService.metadata(prediction_set),
    )


def _model_response(model: TrainedModel) -> TrainedModelResponse:
    return TrainedModelResponse(
        id=model.id,
        project_id=model.project_id,
        corpus_id=model.corpus_id,
        analysis_run_id=model.analysis_run_id,
        training_dataset_snapshot_id=model.training_dataset_snapshot_id,
        model_family=model.model_family,
        task_type=model.task_type,
        label_ids=_loads(model.label_ids_json, []),
        feature_config=_loads(model.feature_config_json, {}),
        training_config=_loads(model.training_config_json, {}),
        metrics=_loads(model.metrics_json, {}),
        version=model.version,
        name=model.name,
        lifecycle_status=model.lifecycle_status,
        lifecycle_notes=model.lifecycle_notes,
        lifecycle_updated_at=model.lifecycle_updated_at,
        created_by=model.created_by,
        created_at=model.created_at,
    )
