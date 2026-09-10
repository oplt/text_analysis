"""HTTP routes for the text research workflow."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.db.session import SessionLocal
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.corpora import router as corpora_router
from backend.modules.text_research.api.schemas import (
    ActiveLearningAssignRequest,
    AdjudicationSaveRequest,
    AnalysisRequest,
    AnalysisRunResponse,
    AnnotationAssignRequest,
    AnnotationCampaignAssignRequest,
    AnnotationCampaignCreate,
    AnnotationCampaignResponse,
    AnnotationCampaignUpdate,
    AnnotationLabelCreate,
    AnnotationLabelResponse,
    AnnotationLabelUpdate,
    AnnotationResponse,
    AnnotationSaveRequest,
    ApplyCleaningRequest,
    BulkMetadataUpdate,
    ClassifierPredictRequest,
    ClassifierTrainRequest,
    CleaningPreviewRequest,
    CleaningPreviewResponse,
    CleaningProfileCreate,
    CleaningProfileResponse,
    CleaningProfileUpdate,
    ClusteringRequest,
    CodebookCreate,
    CodebookResponse,
    ComparativeAnalysisRequest,
    ContextualDatasetCreate,
    ContextualDatasetDetail,
    ContextualDatasetSummary,
    ContextualImportResponse,
    ContextualLinkRequest,
    ContextualObservationPage,
    CooccurrenceRequest,
    CorpusAnnotationAssignRequest,
    CorpusAnnotationAssignResponse,
    CorpusDocumentCreate,
    CorpusDocumentResponse,
    CorpusDocumentUpdate,
    DatasetFreezeRequest,
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DemoSeedRequest,
    DfmRequest,
    DictionaryAnalysisRequest,
    DictionaryCreate,
    DictionaryResponse,
    DictionaryUpdate,
    DimensionalityReductionRequest,
    DriftMonitoringRequest,
    DuplicateDetectionRequest,
    EngineComparisonRequest,
    ExportManifestResponse,
    FrequencyRequest,
    KeynessRequest,
    KwicRequest,
    MeasurementComparisonRequest,
    MetadataImportResponse,
    ModelLifecycleEventResponse,
    ModelLifecycleUpdateRequest,
    ModelPredictionItemResponse,
    NgramRequest,
    PredictionSetDetailResponse,
    PredictionSetPredictionRowResponse,
    PredictionSetPredictionsPageResponse,
    PredictionSetResponse,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingProfileCreate,
    PreprocessingProfileResponse,
    PreprocessingProfileUpdate,
    QuantedaScriptResponse,
    ReadabilityRequest,
    ReliabilityRequest,
    ResearchCorpusCreate,
    ResearchCorpusResponse,
    ResearchCorpusUpdate,
    RobustnessRequest,
    SegmentRequest,
    SimilarityRequest,
    SourceTextResponse,
    StatisticalModelRequest,
    TextUnitContextResponse,
    TopicKSweepRequest,
    TopicLabelRequest,
    TopicSeedStabilityRequest,
    TopicTrainRequest,
    TrainedModelResponse,
    TrainingDatasetSnapshotResponse,
)
from backend.modules.text_research.application.active_learning_service import ActiveLearningService
from backend.modules.text_research.application.adjudication_service import AdjudicationService
from backend.modules.text_research.application.annotation_service import AnnotationService
from backend.modules.text_research.application.campaign_service import AnnotationCampaignService
from backend.modules.text_research.application.classification_service import ClassificationService
from backend.modules.text_research.application.cleaning_service import CleaningProfileService
from backend.modules.text_research.application.codebook_service import CodebookService
from backend.modules.text_research.application.comparative_analysis_service import (
    ComparativeAnalysisService,
)
from backend.modules.text_research.application.contextual_dataset_service import (
    ContextualDatasetService,
)
from backend.modules.text_research.application.corpus_service import CorpusService
from backend.modules.text_research.application.dashboard_service import DashboardService
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.application.demo_seed_service import DemoSeedService
from backend.modules.text_research.application.dictionary_service import DictionaryService
from backend.modules.text_research.application.drift_service import DriftService
from backend.modules.text_research.application.export_service import ExportService
from backend.modules.text_research.application.ingestion_qa_service import IngestionQaService
from backend.modules.text_research.application.measurement_validation_service import (
    MeasurementValidationService,
)
from backend.modules.text_research.application.model_lifecycle_service import ModelLifecycleService
from backend.modules.text_research.application.prediction_service import PredictionService
from backend.modules.text_research.application.prediction_set_service import PredictionSetService
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.application.quantitative_analysis_service import (
    QuantitativeAnalysisService,
)
from backend.modules.text_research.application.reliability_service import ReliabilityService
from backend.modules.text_research.application.robustness_service import RobustnessService
from backend.modules.text_research.application.run_service import RunService
from backend.modules.text_research.application.segmentation_service import SegmentationService
from backend.modules.text_research.application.statistical_modeling_service import (
    StatisticalModelingService,
)
from backend.modules.text_research.application.topic_model_service import TopicModelService
from backend.modules.text_research.domain.analysis_specification import ANALYSIS_TYPES
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

router = APIRouter()
router.include_router(corpora_router)


@router.get("/analysis-engines")
async def analysis_engines(
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict[str, Any]]]:
    """Advertise runtime capabilities without requiring clients to probe R."""
    del current_user
    from backend.modules.text_research.infrastructure.engines.python_engine import (
        PythonAnalysisEngine,
    )
    from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine

    python = PythonAnalysisEngine()
    r_engine = RAnalysisEngine()
    r_available = r_engine.available()
    return {
        "engines": [
            {
                "name": python.name,
                "implementation": "python",
                "available": True,
                "analyses": sorted(name for name in ANALYSIS_TYPES if python.supports(name)),
            },
            {
                "name": r_engine.name,
                "implementation": "quanteda",
                "available": r_available,
                "analyses": sorted(r_engine.supported_analyses) if r_available else [],
            },
        ]
    }


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
    return AnalysisRunResponse(
        id=run.id,
        project_id=run.project_id,
        corpus_id=run.corpus_id,
        run_type=run.run_type,
        status=run.status,
        run_version=int(getattr(run, "run_version", 1) or 1),
        progress_stage=run.progress_stage,
        parameters=_loads(run.parameters_json),
        metrics=_loads(run.metrics_json),
        results=_loads(run.results_json),
        artifact_path=run.artifact_path,
        random_seed=run.random_seed,
        created_by=run.created_by,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        created_at=run.created_at,
    )


_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


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


# ------------------------------------------------------------------
# Demo seed
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/demo-seed", response_model=ResearchCorpusResponse, status_code=201
)
async def seed_demo_corpus(
    project_id: str,
    body: DemoSeedRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await DemoSeedService(db).seed_demo_corpus(
        project_id=project_id,
        user_id=current_user.id,
        corpus_name=body.corpus_name or "Demo Corpus (Synthetic)",
    )
    return _corpus_response(corpus)


# ------------------------------------------------------------------
# Corpora
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/corpora", response_model=ResearchCorpusResponse, status_code=201
)
async def create_corpus(
    project_id: str,
    body: ResearchCorpusCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).create_corpus(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    return _corpus_response(corpus)


@router.get("/projects/{project_id}/corpora", response_model=list[ResearchCorpusResponse])
async def list_corpora(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpora = await CorpusService(db).list_corpora(project_id=project_id, user_id=current_user.id)
    return [_corpus_response(c) for c in corpora]


@router.get("/corpora/{corpus_id}", response_model=ResearchCorpusResponse)
async def get_corpus(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).get_corpus(corpus_id, user_id=current_user.id)
    return _corpus_response(corpus)


@router.patch("/corpora/{corpus_id}", response_model=ResearchCorpusResponse)
async def update_corpus(
    corpus_id: str,
    body: ResearchCorpusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).update_corpus(
        corpus_id, user_id=current_user.id, name=body.name, description=body.description
    )
    return _corpus_response(corpus)


@router.delete("/corpora/{corpus_id}", status_code=204)
async def delete_corpus(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CorpusService(db).delete_corpus(corpus_id, user_id=current_user.id)


# ------------------------------------------------------------------
# Corpus documents
# ------------------------------------------------------------------


@router.post(
    "/corpora/{corpus_id}/documents", response_model=CorpusDocumentResponse, status_code=201
)
async def add_document(
    corpus_id: str,
    body: CorpusDocumentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = await CorpusService(db).add_document(
        corpus_id,
        user_id=current_user.id,
        rag_document_id=body.rag_document_id,
        title=body.title,
        organization=body.organization,
        organization_type=body.organization_type,
        publication_year=body.publication_year,
        publication_type=body.publication_type,
        country=body.country,
        region=body.region,
        cultural_sphere=body.cultural_sphere,
        language=body.language,
        education_level=body.education_level,
        source_url=body.source_url,
        research_notes=body.research_notes,
        extra_metadata=body.metadata_json,
    )
    return _document_response(document)


@router.get(
    "/corpora/{corpus_id}/documents", response_model=PaginatedResponse[CorpusDocumentResponse]
)
async def list_documents(
    corpus_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    organization: str | None = Query(default=None),
    publication_year: int | None = Query(default=None),
    region: str | None = Query(default=None),
    cultural_sphere: str | None = Query(default=None),
    language: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="asc"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents, total = await CorpusService(db).paginate_documents(
        corpus_id,
        user_id=current_user.id,
        organization=organization,
        publication_year=publication_year,
        region=region,
        cultural_sphere=cultural_sphere,
        language=language,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_document_response(d) for d in documents],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/documents/{document_id}", response_model=CorpusDocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = await CorpusService(db).get_document(document_id, user_id=current_user.id)
    return _document_response(document)


@router.patch("/documents/{document_id}", response_model=CorpusDocumentResponse)
async def update_document(
    document_id: str,
    body: CorpusDocumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fields = body.model_dump(exclude_unset=True)
    if "metadata_json" in fields and fields["metadata_json"] is not None:
        fields["extra_metadata"] = fields.pop("metadata_json")
    document = await CorpusService(db).update_document_metadata(
        document_id, user_id=current_user.id, **fields
    )
    return _document_response(document)


@router.post(
    "/corpora/{corpus_id}/documents/metadata/bulk", response_model=list[CorpusDocumentResponse]
)
async def bulk_update_metadata(
    corpus_id: str,
    body: BulkMetadataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents = await CorpusService(db).bulk_update_metadata(
        corpus_id, user_id=current_user.id, document_ids=body.document_ids, fields=body.fields
    )
    return [_document_response(d) for d in documents]


@router.post(
    "/corpora/{corpus_id}/documents/metadata/import", response_model=MetadataImportResponse
)
async def import_metadata_csv(
    corpus_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    content = (await file.read()).decode("utf-8")
    result = await CorpusService(db).import_metadata_csv(
        corpus_id, user_id=current_user.id, csv_content=content
    )
    return MetadataImportResponse(**result)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CorpusService(db).delete_document(document_id, user_id=current_user.id)


@router.get("/documents/{document_id}/source-text", response_model=SourceTextResponse)
async def get_source_text(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = CorpusService(db)
    source = await service.get_canonical_source(document_id, user_id=current_user.id)
    return SourceTextResponse(
        document_id=document_id,
        text=source.canonical_text,
        canonical_text_checksum=source.canonical_text_checksum,
        parser_name=source.parser_name,
        parser_version=source.parser_version,
        source="canonical",
    )


@router.post("/corpora/{corpus_id}/ingestion-qa", response_model=AnalysisRunResponse)
async def run_ingestion_qa(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Scan corpus documents for extraction/quality issues without mutating text."""
    run = await IngestionQaService(db).run_corpus_qa(corpus_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/documents/{document_id}/ingestion-qa")
async def get_document_ingestion_qa(
    document_id: str,
    run_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Inspect QA findings for one document from a (latest) ingestion QA run."""
    return await IngestionQaService(db).get_document_qa_slice(
        document_id, user_id=current_user.id, run_id=run_id
    )


# ------------------------------------------------------------------
# Document cleaning profiles
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/cleaning-profiles",
    response_model=CleaningProfileResponse,
    status_code=201,
)
async def create_cleaning_profile(
    project_id: str,
    body: CleaningProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).create_profile(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        config=body.config,
    )
    return _cleaning_profile_response(profile)


@router.get(
    "/projects/{project_id}/cleaning-profiles",
    response_model=list[CleaningProfileResponse],
)
async def list_cleaning_profiles(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profiles = await CleaningProfileService(db).list_profiles(
        project_id=project_id, user_id=current_user.id
    )
    return [_cleaning_profile_response(p) for p in profiles]


@router.get("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)
async def get_cleaning_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).get_profile(profile_id, user_id=current_user.id)
    return _cleaning_profile_response(profile)


@router.patch("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)
async def update_cleaning_profile(
    profile_id: str,
    body: CleaningProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).update_profile(
        profile_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        config=body.config,
    )
    return _cleaning_profile_response(profile)


@router.delete("/cleaning-profiles/{profile_id}", status_code=204)
async def delete_cleaning_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CleaningProfileService(db).delete_profile(profile_id, user_id=current_user.id)


@router.post("/cleaning/preview", response_model=CleaningPreviewResponse)
async def preview_cleaning_config(
    body: CleaningPreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await CleaningProfileService(db).preview(
        user_id=current_user.id,
        project_id=body.project_id,
        texts=body.texts,
        document_id=body.document_id,
        config=body.config,
        cleaning_profile_id=body.cleaning_profile_id,
    )


@router.post("/corpora/{corpus_id}/clean", response_model=AnalysisRunResponse)
async def apply_document_cleaning(
    corpus_id: str,
    body: ApplyCleaningRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Apply a cleaning profile to corpus documents. Raw extracts stay intact."""
    run = await CleaningProfileService(db).apply_to_corpus(
        corpus_id,
        user_id=current_user.id,
        cleaning_profile_id=body.cleaning_profile_id,
        document_ids=body.document_ids,
    )
    return _run_response(run)


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------


@router.post("/corpora/{corpus_id}/segment", response_model=AnalysisRunResponse, status_code=202)
async def segment_corpus(
    corpus_id: str,
    body: SegmentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await SegmentationService(db).start_segmentation(
        corpus_id, user_id=current_user.id, unit_type=body.unit_type
    )
    return _run_response(run)


# ------------------------------------------------------------------
# Preprocessing profiles
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/preprocessing-profiles",
    response_model=PreprocessingProfileResponse,
    status_code=201,
)
async def create_preprocessing_profile(
    project_id: str,
    body: PreprocessingProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).create_profile(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    return _profile_response(profile)


@router.get(
    "/projects/{project_id}/preprocessing-profiles",
    response_model=list[PreprocessingProfileResponse],
)
async def list_preprocessing_profiles(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profiles = await PreprocessingProfileService(db).list_profiles(
        project_id=project_id, user_id=current_user.id
    )
    return [_profile_response(p) for p in profiles]


@router.get("/preprocessing-profiles/{profile_id}", response_model=PreprocessingProfileResponse)
async def get_preprocessing_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).get_profile(profile_id, user_id=current_user.id)
    return _profile_response(profile)


@router.patch("/preprocessing-profiles/{profile_id}", response_model=PreprocessingProfileResponse)
async def update_preprocessing_profile(
    profile_id: str,
    body: PreprocessingProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).update_profile(
        profile_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    return _profile_response(profile)


@router.delete("/preprocessing-profiles/{profile_id}", status_code=204)
async def delete_preprocessing_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await PreprocessingProfileService(db).delete_profile(profile_id, user_id=current_user.id)


@router.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
async def preview_preprocessing_config(
    body: PreprocessingPreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await PreprocessingProfileService(db).preview(
        user_id=current_user.id,
        project_id=body.project_id,
        corpus_id=body.corpus_id,
        unit_type=body.unit_type,
        texts=body.texts,
        config=body.config,
        preprocessing_profile_id=body.preprocessing_profile_id,
        sample_size=body.sample_size,
    )


# ------------------------------------------------------------------
# Codebooks & labels
# ------------------------------------------------------------------


@router.post("/projects/{project_id}/codebooks", response_model=CodebookResponse, status_code=201)
async def create_codebook(
    project_id: str,
    body: CodebookCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).create_codebook(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        seed_placeholder_labels=body.seed_demo_labels,
    )
    return CodebookResponse.model_validate(codebook)


@router.get("/projects/{project_id}/codebooks", response_model=list[CodebookResponse])
async def list_codebooks(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebooks = await CodebookService(db).list_codebooks(
        project_id=project_id, user_id=current_user.id
    )
    return [CodebookResponse.model_validate(c) for c in codebooks]


@router.get("/codebooks/{codebook_id}", response_model=CodebookResponse)
async def get_codebook(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).get_codebook(codebook_id, user_id=current_user.id)
    return CodebookResponse.model_validate(codebook)


@router.post("/codebooks/{codebook_id}/versions", response_model=CodebookResponse, status_code=201)
async def create_codebook_version(
    codebook_id: str,
    new_version: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).create_version(
        codebook_id, user_id=current_user.id, new_version=new_version
    )
    return CodebookResponse.model_validate(codebook)


@router.post("/codebooks/{codebook_id}/freeze", response_model=CodebookResponse)
async def freeze_codebook(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).freeze_codebook(codebook_id, user_id=current_user.id)
    return CodebookResponse.model_validate(codebook)


@router.post(
    "/codebooks/{codebook_id}/labels", response_model=AnnotationLabelResponse, status_code=201
)
async def add_label(
    codebook_id: str,
    body: AnnotationLabelCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    label = await CodebookService(db).add_label(
        codebook_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        inclusion_criteria=body.inclusion_criteria,
        exclusion_criteria=body.exclusion_criteria,
        positive_examples=body.positive_examples,
        negative_examples=body.negative_examples,
    )
    return _label_response(label)


@router.get("/codebooks/{codebook_id}/labels", response_model=list[AnnotationLabelResponse])
async def list_labels(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    labels = await CodebookService(db).list_labels(codebook_id, user_id=current_user.id)
    return [_label_response(label) for label in labels]


@router.patch("/labels/{label_id}", response_model=AnnotationLabelResponse)
async def update_label(
    label_id: str,
    body: AnnotationLabelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fields = body.model_dump(exclude_unset=True)
    label = await CodebookService(db).update_label(label_id, user_id=current_user.id, **fields)
    return _label_response(label)


# ------------------------------------------------------------------
# Annotation
# ------------------------------------------------------------------


@router.post("/annotations/assign", status_code=201)
async def assign_annotation_tasks(
    body: AnnotationAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = await AnnotationService(db).assign_tasks(
        user_id=current_user.id,
        text_unit_ids=body.text_unit_ids,
        annotator_ids=body.annotator_ids,
    )
    return {"assigned_count": len(tasks)}


@router.post(
    "/corpora/{corpus_id}/annotations/assign",
    response_model=CorpusAnnotationAssignResponse,
    status_code=201,
)
async def assign_corpus_annotation_tasks(
    corpus_id: str,
    body: CorpusAnnotationAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample_size = body.sample_size if body.limit is None else body.limit
    return await AnnotationService(db).assign_corpus_tasks(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        annotator_ids=body.annotator_ids,
        sample_size=sample_size,
        strategy=body.strategy,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        random_seed=body.random_seed,
        stratify_by=body.stratify_by,
        stratum_mode=body.stratum_mode,
        sampling_level=body.sampling_level,
        max_units_per_document=body.max_units_per_document,
        campaign_id=body.campaign_id,
        create_campaign=body.create_campaign and body.campaign_id is None,
        campaign_name=body.campaign_name,
        campaign_description=body.campaign_description,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        codebook_id=body.codebook_id,
    )


@router.post(
    "/projects/{project_id}/annotation-campaigns",
    response_model=AnnotationCampaignResponse,
    status_code=201,
)
async def create_annotation_campaign(
    project_id: str,
    body: AnnotationCampaignCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.create_campaign_committed(
        user_id=current_user.id,
        project_id=project_id,
        corpus_id=body.corpus_id,
        name=body.name,
        description=body.description,
        codebook_id=body.codebook_id,
        unit_type=body.unit_type,
        sampling_strategy=body.sampling_strategy,
        assignment_strategy=body.assignment_strategy,
        sample_size=body.sample_size,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        annotator_ids=body.annotator_ids,
        metadata=body.metadata,
    )
    return await service.campaign_payload(campaign)


@router.get(
    "/projects/{project_id}/annotation-campaigns",
    response_model=list[AnnotationCampaignResponse],
)
async def list_annotation_campaigns(
    project_id: str,
    corpus_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaigns = await service.list_campaigns(
        project_id, user_id=current_user.id, corpus_id=corpus_id
    )
    return [await service.campaign_payload(campaign) for campaign in campaigns]


@router.get(
    "/annotation-campaigns/{campaign_id}",
    response_model=AnnotationCampaignResponse,
)
async def get_annotation_campaign(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.get_campaign(campaign_id, user_id=current_user.id)
    return await service.campaign_payload(campaign)


@router.patch(
    "/annotation-campaigns/{campaign_id}",
    response_model=AnnotationCampaignResponse,
)
async def patch_annotation_campaign(
    campaign_id: str,
    body: AnnotationCampaignUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.patch_campaign(
        campaign_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        status=body.status,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        metadata=body.metadata,
    )
    return await service.campaign_payload(campaign)


@router.post("/annotation-campaigns/{campaign_id}/assign", status_code=201)
async def assign_annotation_campaign(
    campaign_id: str,
    body: AnnotationCampaignAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationCampaignService(db).assign(
        campaign_id,
        user_id=current_user.id,
        annotator_ids=body.annotator_ids,
        sample_size=body.sample_size,
        strategy=body.strategy,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        random_seed=body.random_seed,
        stratify_by=body.stratify_by,
        stratum_mode=body.stratum_mode,
        sampling_level=body.sampling_level,
        max_units_per_document=body.max_units_per_document,
    )


@router.get("/annotation-campaigns/{campaign_id}/progress")
async def annotation_campaign_progress(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationCampaignService(db).progress(campaign_id, user_id=current_user.id)


@router.get("/text-units/{text_unit_id}/blind-policy")
async def text_unit_blind_policy(
    text_unit_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await AnnotationService(db).get_text_unit_or_404(text_unit_id, user_id=current_user.id)
    return await AnnotationCampaignService(db).blind_policy_for_annotator_unit(
        text_unit_id=text_unit_id, annotator_id=current_user.id
    )


@router.get("/annotations/queue", response_model=PaginatedResponse[dict[str, Any]])
async def list_annotation_queue(
    status: str | None = None,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    queue, total = await AnnotationService(db).list_queue(
        requesting_user_id=current_user.id,
        status=status,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(queue, total=total, limit=pagination.limit, offset=pagination.offset)


@router.post("/annotations", response_model=list[AnnotationResponse])
async def save_annotations(
    body: AnnotationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    annotations = await AnnotationService(db).save_annotations(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        codebook_id=body.codebook_id,
        values=body.values,
        campaign_id=body.campaign_id,
        mark_task_complete=body.mark_task_complete,
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get("/corpora/{corpus_id}/annotations/progress")
async def annotation_progress(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationService(db).progress(corpus_id, user_id=current_user.id)


@router.get("/text-units/{text_unit_id}/annotations")
async def list_unit_annotations(
    text_unit_id: str,
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    annotations = await AnnotationService(db).list_annotations_for_unit(
        text_unit_id, user_id=current_user.id, campaign_id=campaign_id
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get(
    "/corpora/{corpus_id}/annotations",
    response_model=list[AnnotationResponse],
)
async def list_corpus_annotations_for_units(
    corpus_id: str,
    text_unit_ids: list[str] = Query(default=[]),
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List human annotations for specific units.

    Predictions and adjudications are never returned here — callers must keep
    MODEL PREDICTION / HUMAN ANNOTATION / ADJUDICATED GOLD layers separate.
    """
    annotations = await AnnotationService(db).list_annotations_for_units(
        corpus_id,
        user_id=current_user.id,
        text_unit_ids=text_unit_ids,
        campaign_id=campaign_id,
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get("/text-units/{text_unit_id}/context", response_model=TextUnitContextResponse)
async def get_text_unit_context(
    text_unit_id: str,
    window: int = Query(default=2, ge=0, le=10),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationService(db).get_unit_context(
        text_unit_id, user_id=current_user.id, window=window
    )


# ------------------------------------------------------------------
# Reliability & adjudication
# ------------------------------------------------------------------


@router.post("/corpora/{corpus_id}/reliability", response_model=AnalysisRunResponse)
async def compute_reliability(
    corpus_id: str,
    body: ReliabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await ReliabilityService(db).compute_reliability(
        corpus_id,
        user_id=current_user.id,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        campaign_id=body.campaign_id,
        unit_type=body.unit_type,
        annotator_ids=body.annotator_ids,
        bootstrap_samples=body.bootstrap_samples,
        confidence_level=body.confidence_level,
        random_seed=body.random_seed,
    )
    return _run_response(run)


@router.post(
    "/annotation-campaigns/{campaign_id}/reliability",
    response_model=AnalysisRunResponse,
)
async def compute_campaign_reliability(
    campaign_id: str,
    body: ReliabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    codebook_id = body.codebook_id or campaign.codebook_id
    if not codebook_id:
        raise HTTPException(
            status_code=422, detail="codebook_id is required when the campaign has none"
        )
    run = await ReliabilityService(db).compute_reliability(
        campaign.corpus_id,
        user_id=current_user.id,
        codebook_id=codebook_id,
        label_ids=body.label_ids,
        campaign_id=campaign_id,
        unit_type=body.unit_type or campaign.unit_type,
        annotator_ids=body.annotator_ids,
        bootstrap_samples=body.bootstrap_samples,
        confidence_level=body.confidence_level,
        random_seed=body.random_seed,
    )
    return _run_response(run)


@router.get("/corpora/{corpus_id}/adjudication/disagreements")
async def list_disagreements(
    corpus_id: str,
    codebook_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AdjudicationService(db).list_disagreements(
        corpus_id, user_id=current_user.id, codebook_id=codebook_id
    )


@router.post("/adjudication", status_code=201)
async def save_adjudication(
    body: AdjudicationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adjudication = await AdjudicationService(db).save_adjudication(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        label_id=body.label_id,
        final_value=body.final_value,
        comment=body.comment,
        campaign_id=body.campaign_id,
    )
    return {"id": adjudication.id}


@router.get("/corpora/{corpus_id}/adjudications")
async def list_adjudications(
    corpus_id: str,
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AdjudicationService(db).list_adjudications(
        corpus_id, user_id=current_user.id, campaign_id=campaign_id
    )


@router.get("/annotation-campaigns/{campaign_id}/disagreements")
async def list_campaign_disagreements(
    campaign_id: str,
    codebook_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    resolved_codebook_id = codebook_id or campaign.codebook_id
    if not resolved_codebook_id:
        raise HTTPException(status_code=422, detail="Campaign has no codebook")
    return await AdjudicationService(db).list_disagreements(
        campaign.corpus_id,
        user_id=current_user.id,
        codebook_id=resolved_codebook_id,
        campaign_id=campaign_id,
    )


@router.post("/annotation-campaigns/{campaign_id}/adjudications", status_code=201)
async def save_campaign_adjudication(
    campaign_id: str,
    body: AdjudicationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adjudication = await AdjudicationService(db).save_adjudication(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        label_id=body.label_id,
        final_value=body.final_value,
        comment=body.comment,
        campaign_id=campaign_id,
    )
    return {"id": adjudication.id}


@router.get("/annotation-campaigns/{campaign_id}/adjudications")
async def list_campaign_adjudications(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    return await AdjudicationService(db).list_adjudications(
        campaign.corpus_id, user_id=current_user.id, campaign_id=campaign_id
    )


# ------------------------------------------------------------------
# Quantitative analysis
# ------------------------------------------------------------------


def _analysis_filters(body: AnalysisRequest) -> dict[str, Any]:
    excluded = {
        "unit_type",
        "preprocessing_profile_id",
        "top_n",
        "n",
        "weighting",
        "k1",
        "b",
        "smooth_idf",
        "rate_per",
        "skip",
        "group_by",
        "force_sparse_only",
        "trim",
        "run_async",
        "engine",
        "keyword",
        "window_size",
        "case_sensitive",
        "query_mode",
        "language",
        "token_attribute",
        "max_matches",
        "dictionary_id",
        "dictionary_terms",
        "hierarchy",
        "exclusions",
        "dictionary_language",
        "association_method",
        "directional",
        "min_frequency",
        "min_count",
        "include_network",
        "method",
        "mode",
        "top_k",
        "min_score",
        "centroid_target",
        "query_text",
        "query_unit_id",
        "embeddings",
        "query_embedding",
        "methods",
        "lexical_threshold",
        "char_ngram_size",
        "use_minhash",
        "minhash_num_perm",
        "minhash_shingle_size",
        "minhash_threshold",
        "max_pairs",
    }
    return {k: v for k, v in body.model_dump().items() if k not in excluded and v is not None}


@router.post("/corpora/{corpus_id}/analysis/corpus-stats", response_model=AnalysisRunResponse)
async def corpus_stats(
    corpus_id: str,
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).corpus_stats(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/frequencies", response_model=AnalysisRunResponse)
async def frequencies(
    corpus_id: str,
    body: FrequencyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.engine.runtime == "r":
        run = await QuantitativeAnalysisService(db).r_analysis(
            corpus_id,
            user_id=current_user.id,
            analysis_type="frequencies",
            unit_type=body.unit_type,
            preprocessing_profile_id=body.preprocessing_profile_id,
            analysis_parameters={
                "top_n": body.top_n,
                "rate_per": body.rate_per,
                "group_by": body.group_by,
            },
            **_analysis_filters(body),
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).frequencies(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
        rate_per=body.rate_per,
        group_by=body.group_by,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/ngrams", response_model=AnalysisRunResponse)
async def ngrams(
    corpus_id: str,
    body: NgramRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).ngrams(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        n=body.n,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
        rate_per=body.rate_per,
        skip=body.skip,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/dfm", response_model=AnalysisRunResponse)
async def dfm(
    corpus_id: str,
    body: DfmRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.engine.runtime == "r":
        run = await QuantitativeAnalysisService(db).r_analysis(
            corpus_id,
            user_id=current_user.id,
            analysis_type="dfm",
            unit_type=body.unit_type,
            preprocessing_profile_id=body.preprocessing_profile_id,
            analysis_parameters={"weighting": body.weighting},
            **_analysis_filters(body),
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).dfm(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        weighting=body.weighting,
        k1=body.k1,
        b=body.b,
        smooth_idf=body.smooth_idf,
        preprocessing_profile_id=body.preprocessing_profile_id,
        force_sparse_only=body.force_sparse_only,
        trim=body.trim.model_dump(exclude_none=True) if body.trim else None,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/kwic", response_model=AnalysisRunResponse)
async def kwic(
    corpus_id: str,
    body: KwicRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.engine.runtime == "r":
        run = await QuantitativeAnalysisService(db).r_analysis(
            corpus_id,
            user_id=current_user.id,
            analysis_type="kwic",
            unit_type=body.unit_type,
            preprocessing_profile_id=body.preprocessing_profile_id,
            analysis_parameters={
                "keyword": body.keyword,
                "window_size": body.window_size,
                "case_sensitive": body.case_sensitive,
                "query_mode": body.query_mode,
            },
            **_analysis_filters(body),
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).kwic(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        keyword=body.keyword,
        window_size=body.window_size,
        case_sensitive=body.case_sensitive,
        query_mode=body.query_mode,
        language=body.language,
        token_attribute=body.token_attribute,
        max_matches=body.max_matches,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/dictionary", response_model=AnalysisRunResponse)
async def dictionary_analysis(
    corpus_id: str,
    body: DictionaryAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not body.dictionary_id and not body.dictionary_terms and not body.hierarchy:
        raise HTTPException(
            status_code=400,
            detail="dictionary_id, dictionary_terms, or hierarchy required (user-defined only)",
        )
    if body.engine.runtime == "r":
        if body.group_by:
            raise HTTPException(
                status_code=422,
                detail="R dictionary does not yet support group_by",
            )
        if body.dictionary_id:
            from backend.modules.text_research.application.dictionary_service import (
                DictionaryService,
            )

            definition = await DictionaryService(db).get_dictionary(
                body.dictionary_id,
                user_id=current_user.id,
            )
            corpus = await CorpusService(db).get_corpus(corpus_id, user_id=current_user.id)
            if definition.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary does not belong to this corpus project",
                )
            dictionary_payload = DictionaryService.get_spec(definition).to_dict()
            analysis_parameters = {
                "entries": dictionary_payload["entries"],
                "exclusions": dictionary_payload["exclusions"],
                "case_sensitive": body.case_sensitive,
                "rate_per": body.rate_per,
            }
        else:
            analysis_parameters = {
                "hierarchy": body.hierarchy,
                "terms": body.dictionary_terms,
                "exclusions": body.exclusions,
                "case_sensitive": body.case_sensitive,
                "rate_per": body.rate_per,
            }
        run = await QuantitativeAnalysisService(db).r_analysis(
            corpus_id,
            user_id=current_user.id,
            analysis_type="dictionary",
            unit_type=body.unit_type,
            preprocessing_profile_id=body.preprocessing_profile_id,
            analysis_parameters=analysis_parameters,
            **_analysis_filters(body),
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).dictionary(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        dictionary_terms=body.dictionary_terms or [],
        dictionary_id=body.dictionary_id,
        hierarchy=body.hierarchy,
        exclusions=body.exclusions,
        dictionary_language=body.dictionary_language,
        case_sensitive=body.case_sensitive,
        rate_per=body.rate_per,
        group_by=body.group_by,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/keyness", response_model=AnalysisRunResponse)
async def keyness(
    corpus_id: str,
    body: KeynessRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.engine.runtime == "r":
        run = await QuantitativeAnalysisService(db).r_keyness(
            corpus_id,
            user_id=current_user.id,
            unit_type=body.unit_type,
            filters_a=body.filters_a,
            filters_b=body.filters_b,
            group_field=body.group_field,
            method=body.method,
            correction=body.correction,
            min_frequency=body.min_frequency,
            preprocessing_profile_id=body.preprocessing_profile_id,
            top_n=body.top_n,
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).keyness(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        filters_a=body.filters_a,
        filters_b=body.filters_b,
        group_field=body.group_field,
        method=body.method,
        correction=body.correction,
        min_frequency=body.min_frequency,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/cooccurrence", response_model=AnalysisRunResponse)
async def cooccurrence(
    corpus_id: str,
    body: CooccurrenceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.engine.runtime == "r":
        run = await QuantitativeAnalysisService(db).r_analysis(
            corpus_id,
            user_id=current_user.id,
            analysis_type="cooccurrence",
            unit_type=body.unit_type,
            preprocessing_profile_id=body.preprocessing_profile_id,
            analysis_parameters={
                "window_size": body.window_size,
                "top_n": body.top_n,
                "association_method": body.association_method,
                "directional": body.directional,
                "min_frequency": body.min_frequency,
                "min_count": body.min_count,
                "include_network": body.include_network,
            },
            **_analysis_filters(body),
        )
        return _run_response(run)
    run = await QuantitativeAnalysisService(db).cooccurrence(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        window_size=body.window_size,
        top_n=body.top_n,
        association_method=body.association_method,
        directional=body.directional,
        min_frequency=body.min_frequency,
        min_count=body.min_count,
        include_network=body.include_network,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/engine-comparison", response_model=AnalysisRunResponse)
async def engine_comparison(
    corpus_id: str,
    body: EngineComparisonRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).engine_comparison(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        analysis_type=body.analysis_type,
        analysis_parameters=body.analysis_parameters,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/similarity", response_model=AnalysisRunResponse)
async def similarity(
    corpus_id: str,
    body: SimilarityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Document-to-document / unit-to-unit / query-to-document / group-centroid
    similarity (cosine-on-TFIDF, Jaccard, or caller-supplied embeddings)."""
    run = await QuantitativeAnalysisService(db).similarity(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        method=body.method,
        mode=body.mode,
        top_k=body.top_k,
        min_score=body.min_score,
        group_by=body.group_by,
        centroid_target=body.centroid_target,
        query_text=body.query_text,
        query_unit_id=body.query_unit_id,
        embeddings=body.embeddings,
        query_embedding=body.query_embedding,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post(
    "/corpora/{corpus_id}/analysis/duplicate-detection", response_model=AnalysisRunResponse
)
async def duplicate_detection(
    corpus_id: str,
    body: DuplicateDetectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Exact / normalized checksum, lexical near-dup, and optional MinHash
    duplicate detection — the same engine ingestion QA uses, run explicitly."""
    run = await QuantitativeAnalysisService(db).duplicate_detection(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        methods=body.methods,
        lexical_threshold=body.lexical_threshold,
        char_ngram_size=body.char_ngram_size,
        use_minhash=body.use_minhash,
        minhash_num_perm=body.minhash_num_perm,
        minhash_shingle_size=body.minhash_shingle_size,
        minhash_threshold=body.minhash_threshold,
        max_pairs=body.max_pairs,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/clustering", response_model=AnalysisRunResponse)
async def clustering(
    corpus_id: str,
    body: ClusteringRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).clustering(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        n_clusters=body.n_clusters,
        algorithm=body.algorithm,
        use_svd=body.use_svd,
        n_svd_components=body.n_svd_components,
        top_terms=body.top_terms,
        random_seed=body.random_seed,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post(
    "/corpora/{corpus_id}/analysis/dimensionality-reduction",
    response_model=AnalysisRunResponse,
)
async def dimensionality_reduction(
    corpus_id: str,
    body: DimensionalityReductionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).dimensionality_reduction(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        method=body.method,
        n_components=body.n_components,
        random_seed=body.random_seed,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/readability", response_model=AnalysisRunResponse)
async def readability(
    corpus_id: str,
    body: ReadabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).readability(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        **_analysis_filters(body),
    )
    return _run_response(run)


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


# ------------------------------------------------------------------
# Topic models
# ------------------------------------------------------------------


def _topic_filters(body: TopicTrainRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "n_topics",
            "preprocessing_profile_id",
            "max_iterations",
            "random_seed",
            "group_by",
            "holdout_fraction",
            "holdout_unit_ids",
            "embedding_provider",
            "embedding_model_name",
            "persist_embedding_artifacts",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/train", response_model=AnalysisRunResponse, status_code=202
)
async def train_topic_model(
    corpus_id: str,
    body: TopicTrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).train(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        n_topics=body.n_topics,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        random_seed=body.random_seed,
        group_by=body.group_by,
        holdout_fraction=body.holdout_fraction,
        holdout_unit_ids=body.holdout_unit_ids,
        embedding_provider=body.embedding_provider,
        embedding_model_name=body.embedding_model_name,
        persist_embedding_artifacts=body.persist_embedding_artifacts,
        run_async=body.run_async,
        **_topic_filters(body),
    )
    return _run_response(run)


def _topic_ksweep_filters(body: TopicKSweepRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "k_values",
            "preprocessing_profile_id",
            "max_iterations",
            "random_seed",
            "holdout_fraction",
            "holdout_unit_ids",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/k-sweep", response_model=AnalysisRunResponse, status_code=202
)
async def topic_k_sweep(
    corpus_id: str,
    body: TopicKSweepRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).run_k_sweep(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        k_values=body.k_values,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        random_seed=body.random_seed,
        holdout_fraction=body.holdout_fraction,
        holdout_unit_ids=body.holdout_unit_ids,
        run_async=body.run_async,
        **_topic_ksweep_filters(body),
    )
    return _run_response(run)


def _topic_seed_stability_filters(body: TopicSeedStabilityRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "n_topics",
            "seeds",
            "preprocessing_profile_id",
            "max_iterations",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/seed-stability",
    response_model=AnalysisRunResponse,
    status_code=202,
)
async def topic_seed_stability(
    corpus_id: str,
    body: TopicSeedStabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).run_seed_stability(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        n_topics=body.n_topics,
        seeds=body.seeds,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        run_async=body.run_async,
        **_topic_seed_stability_filters(body),
    )
    return _run_response(run)


@router.post("/topics/{run_id}/labels", status_code=201)
async def name_topic(
    run_id: str,
    body: TopicLabelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    label = await TopicModelService(db).name_topic(
        run_id,
        user_id=current_user.id,
        topic_id=body.topic_id,
        human_name=body.human_name,
    )
    return {"id": label.id, "topic_id": label.topic_id, "human_name": label.human_name}


@router.get("/topics/{run_id}/labels")
async def list_topic_labels(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    labels = await TopicModelService(db).get_topic_labels(run_id, user_id=current_user.id)
    return [
        {
            "id": label.id,
            "topic_id": label.topic_id,
            "human_name": label.human_name,
            "created_at": label.created_at,
        }
        for label in labels
    ]


# ------------------------------------------------------------------
# Robustness, comparative, dashboard
# ------------------------------------------------------------------


@router.post("/robustness/sweep", response_model=AnalysisRunResponse, status_code=202)
async def run_robustness_sweep(
    body: RobustnessRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RobustnessService(db).run_sweep(
        body.snapshot_id,
        user_id=current_user.id,
        algorithm=body.algorithm,
        seeds=body.seeds,
        cv_folds=body.cv_folds,
        class_weights=body.class_weights,
        test_size=body.test_size,
        group_field=body.group_field,
        max_groups=body.max_groups,
        temporal_field=body.temporal_field,
        temporal_windows=body.temporal_windows,
        transfer_field=body.transfer_field,
        transfer_train_values=body.transfer_train_values,
        transfer_test_values=body.transfer_test_values,
        run_async=body.run_async,
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/analysis/statistical-model", response_model=AnalysisRunResponse)
async def statistical_model(
    corpus_id: str,
    body: StatisticalModelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await StatisticalModelingService(db).fit(
        corpus_id,
        user_id=current_user.id,
        model=body.model,
        dependent_var=body.dependent_var,
        independent_vars=body.independent_vars,
        rows=body.rows,
        add_intercept=body.add_intercept,
    )
    return _run_response(run)


@router.post(
    "/corpora/{corpus_id}/analysis/measurement-comparison",
    response_model=AnalysisRunResponse,
)
async def measurement_comparison(
    corpus_id: str,
    body: MeasurementComparisonRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await MeasurementValidationService(db).compare(
        corpus_id,
        user_id=current_user.id,
        source_a=body.source_a,
        values_a=body.values_a,
        source_b=body.source_b,
        values_b=body.values_b,
        ids=body.ids,
        value_kind=body.value_kind,
        subgroup=body.subgroup,
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/comparative/prevalence", response_model=AnalysisRunResponse)
async def comparative_prevalence(
    corpus_id: str,
    body: ComparativeAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filters = {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "codebook_id",
            "label_ids",
            "group_by",
            "provenance_mode",
            "model_id",
        }
        and v is not None
    }
    run = await ComparativeAnalysisService(db).prevalence_by_metadata(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        group_by=body.group_by,
        provenance_mode=body.provenance_mode,
        model_id=body.model_id,
        **filters,
    )
    return _run_response(run)


@router.get("/corpora/{corpus_id}/dashboard")
async def dashboard_summary(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await DashboardService(db).summary(corpus_id, user_id=current_user.id)


# ------------------------------------------------------------------
# Runs & exports
# ------------------------------------------------------------------


@router.get("/projects/{project_id}/runs", response_model=PaginatedResponse[AnalysisRunResponse])
async def list_runs(
    project_id: str,
    corpus_id: str | None = None,
    run_type: str | None = None,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    runs, total = await RunService(db).list_runs(
        project_id=project_id,
        user_id=current_user.id,
        corpus_id=corpus_id,
        run_type=run_type,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_run_response(r) for r in runs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/runs/{run_id}", response_model=AnalysisRunResponse)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RunService(db).get_run(run_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    """Stream run state via Redis Pub/Sub, with rare DB reconcile + FE poll fallback.

    Workers persist durable state in PostgreSQL and publish snapshots to Redis.
    This endpoint is the transport only — it does not poll the DB every second.
    """
    from backend.core.cache import redis_client
    from backend.core.config import settings
    from backend.modules.text_research.infrastructure.run_events import (
        TERMINAL_RUN_STATUSES,
        run_events_channel,
    )

    # Authorize once up front.
    async with SessionLocal() as session:
        await RunService(session).get_run(run_id, user_id=current_user.id)

    async def _load_response() -> AnalysisRunResponse:
        async with SessionLocal() as session:
            run = await RunService(session).get_run(run_id, user_id=current_user.id)
            return _run_response(run)

    async def events():
        previous: AnalysisRunResponse | None = None
        try:
            from backend.observability.prometheus_metrics import research_sse_active_connections

            research_sse_active_connections.inc()
        except Exception:
            pass
        try:
            current = await _load_response()
            event_name = _run_event_name(current, previous)
            payload = json.dumps(current.model_dump(mode="json"), separators=(",", ":"))
            if current.artifact_path:
                yield f"event: artifact-created\ndata: {payload}\n\n"
            yield f"event: {event_name}\ndata: {payload}\n\n"
            previous = current
            if current.status in TERMINAL_RUN_STATUSES:
                return

            channel = run_events_channel(run_id)
            pubsub = None
            use_redis = bool(getattr(settings, "CACHE_ENABLED", True))
            last_db_reconcile = asyncio.get_running_loop().time()
            db_reconcile_every = 15.0

            try:
                if use_redis:
                    pubsub = redis_client.pubsub()
                    await pubsub.subscribe(channel)

                while True:
                    message = None
                    if pubsub is not None:
                        try:
                            message = await pubsub.get_message(
                                ignore_subscribe_messages=True, timeout=1.0
                            )
                        except Exception:
                            # Redis hiccup → fall back to DB polling for this stream.
                            pubsub = None
                            use_redis = False

                    if message and message.get("type") == "message":
                        raw = message.get("data")
                        try:
                            envelope = json.loads(raw) if isinstance(raw, str) else raw
                            run_payload = (
                                envelope.get("run") if isinstance(envelope, dict) else None
                            )
                            redis_event = (
                                envelope.get("event") if isinstance(envelope, dict) else None
                            )
                        except (TypeError, json.JSONDecodeError):
                            run_payload = None
                            redis_event = None

                        if isinstance(run_payload, dict):
                            current = AnalysisRunResponse.model_validate(run_payload)
                            payload = json.dumps(
                                current.model_dump(mode="json"), separators=(",", ":")
                            )
                            if current.artifact_path and (
                                previous is None or current.artifact_path != previous.artifact_path
                            ):
                                yield f"event: artifact-created\ndata: {payload}\n\n"
                            name = redis_event or _run_event_name(current, previous)
                            if name != "artifact-created":
                                yield f"event: {name}\ndata: {payload}\n\n"
                            elif (
                                previous is not None
                                and current.artifact_path == previous.artifact_path
                            ):
                                # Redis said artifact-created but path unchanged.
                                yield f"event: progress\ndata: {payload}\n\n"
                            previous = current
                            if current.status in TERMINAL_RUN_STATUSES:
                                return
                            continue

                    now = asyncio.get_running_loop().time()
                    # Rare DB reconcile (missed publish / Redis down) — not 1Hz polling.
                    should_reconcile = (not use_redis) or (
                        now - last_db_reconcile >= db_reconcile_every
                    )
                    if should_reconcile:
                        last_db_reconcile = now
                        current = await _load_response()
                        if previous != current:
                            payload = json.dumps(
                                current.model_dump(mode="json"), separators=(",", ":")
                            )
                            if current.artifact_path and (
                                previous is None or current.artifact_path != previous.artifact_path
                            ):
                                yield f"event: artifact-created\ndata: {payload}\n\n"
                            yield (
                                f"event: {_run_event_name(current, previous)}\ndata: {payload}\n\n"
                            )
                            previous = current
                        if current.status in TERMINAL_RUN_STATUSES:
                            return
                        if not use_redis:
                            await asyncio.sleep(2)
                            continue

                    # SSE comment heartbeat keeps proxies from buffering/closing idle streams.
                    yield ": keepalive\n\n"
            finally:
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        await pubsub.unsubscribe(channel)
                        await pubsub.aclose()
        finally:
            try:
                from backend.observability.prometheus_metrics import research_sse_active_connections

                research_sse_active_connections.dec()
            except Exception:
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/clone-parameters")
async def clone_run_parameters(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RunService(db).clone_parameters(run_id, user_id=current_user.id)


@router.get("/runs/{run_id}/provenance")
async def get_run_provenance(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full provenance block + one-click reproduce payload for a persisted run."""
    return await RunService(db).get_provenance(run_id, user_id=current_user.id)


@router.post("/runs/{run_id}/rerun", response_model=AnalysisRunResponse, status_code=202)
async def rerun(
    run_id: str,
    run_async: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """One-click reproducible re-execution of a prior analysis run."""
    run = await RunService(db).rerun(run_id, user_id=current_user.id, run_async=run_async)
    return _run_response(run)


@router.post("/runs/{run_id}/cancel", response_model=AnalysisRunResponse)
async def cancel_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RunService(db).cancel_run(run_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/runs/{run_a_id}/compare/{run_b_id}")
async def compare_runs(
    run_a_id: str,
    run_b_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RunService(db).compare_runs(run_a_id, run_b_id, user_id=current_user.id)


@router.get("/runs/{run_id}/export.json")
async def export_run_json(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_run_json(run_id, user_id=current_user.id)


@router.get("/codebooks/{codebook_id}/export.json")
async def export_codebook_json(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_codebook_json(codebook_id, user_id=current_user.id)


@router.get("/classifiers/{model_id}/export/metrics.json")
async def export_model_metrics(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_model_metrics(model_id, user_id=current_user.id)


@router.get("/preprocessing-profiles/{profile_id}/export.json")
async def export_preprocessing_profile_json(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_preprocessing_profile_json(
        profile_id, user_id=current_user.id
    )


@router.get("/corpora/{corpus_id}/export/manifest", response_model=ExportManifestResponse)
async def export_manifest(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    manifest = await ExportService(db).build_manifest(corpus_id, user_id=current_user.id)
    return ExportManifestResponse(manifest=manifest)


@router.get("/corpora/{corpus_id}/export/quanteda-script", response_model=QuantedaScriptResponse)
async def export_quanteda_script(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    script = await ExportService(db).build_quanteda_script(corpus_id, user_id=current_user.id)
    return QuantedaScriptResponse(script=script)


@router.get("/corpora/{corpus_id}/export/units.csv")
async def export_units_csv(
    corpus_id: str,
    unit_type: str = Query(default="paragraph"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        ExportService(db).iter_units_csv(corpus_id, user_id=current_user.id, unit_type=unit_type),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{corpus_id}-units.csv"'},
    )


@router.get("/corpora/{corpus_id}/export/annotations.csv", response_class=PlainTextResponse)
async def export_annotations_csv(
    corpus_id: str,
    codebook_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_annotations_csv(
        corpus_id, user_id=current_user.id, codebook_id=codebook_id
    )


@router.get("/classifiers/{model_id}/export/predictions.csv", response_class=PlainTextResponse)
async def export_predictions_csv(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_predictions_csv(model_id, user_id=current_user.id)


# ------------------------------------------------------------------
# Contextual / mixed-method datasets
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/contextual-datasets",
    response_model=ContextualDatasetSummary,
    status_code=201,
)
async def create_contextual_dataset(
    project_id: str,
    body: ContextualDatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = await ContextualDatasetService(db).create_dataset(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    return ContextualDatasetSummary(
        id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        description=dataset.description,
        created_by=dataset.created_by,
        created_at=dataset.created_at,
        observation_count=0,
    )


@router.get(
    "/projects/{project_id}/contextual-datasets",
    response_model=list[ContextualDatasetSummary],
)
async def list_contextual_datasets(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await ContextualDatasetService(db).list_datasets(
        project_id=project_id, user_id=current_user.id
    )
    return [ContextualDatasetSummary.model_validate(row) for row in rows]


@router.get("/contextual-datasets/{dataset_id}", response_model=ContextualDatasetDetail)
async def get_contextual_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    detail = await ContextualDatasetService(db).get_dataset(dataset_id, user_id=current_user.id)
    return ContextualDatasetDetail.model_validate(detail)


@router.get(
    "/contextual-datasets/{dataset_id}/observations",
    response_model=ContextualObservationPage,
)
async def list_contextual_observations(
    dataset_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    page = await ContextualDatasetService(db).list_observations(
        dataset_id, user_id=current_user.id, limit=limit, offset=offset
    )
    return ContextualObservationPage.model_validate(page)


@router.delete("/contextual-datasets/{dataset_id}", status_code=204)
async def delete_contextual_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ContextualDatasetService(db).delete_dataset(dataset_id, user_id=current_user.id)


@router.post(
    "/contextual-datasets/{dataset_id}/import-csv",
    response_model=ContextualImportResponse,
)
async def import_contextual_csv(
    dataset_id: str,
    file: UploadFile = File(...),
    replace_existing: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        reader = csv.DictReader(io.TextIOWrapper(file.file, encoding="utf-8-sig", newline=""))
    except UnicodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded.") from exc
    result = await ContextualDatasetService(db).import_csv(
        dataset_id,
        user_id=current_user.id,
        reader=reader,
        replace_existing=replace_existing,
    )
    return ContextualImportResponse.model_validate(result)


@router.post("/contextual-datasets/{dataset_id}/link-discourse")
async def link_contextual_discourse(
    dataset_id: str,
    body: ContextualLinkRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ContextualDatasetService(db).link_discourse(
        dataset_id,
        user_id=current_user.id,
        corpus_id=body.corpus_id,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        indicator_key=body.indicator_key,
        unit_type=body.unit_type,
        group_by=body.group_by,
        join_on_year=body.join_on_year,
        provenance_mode=body.provenance_mode,
        model_id=body.model_id,
    )
