"""Pydantic schemas for the Policy Text Lab research API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResearchCorpusCreate(BaseModel):
    name: str
    description: str | None = None


class ResearchCorpusUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ResearchCorpusResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CorpusDocumentCreate(BaseModel):
    rag_document_id: str
    title: str | None = None
    organization: str | None = None
    organization_type: str | None = None
    publication_year: int | None = None
    publication_type: str | None = None
    country: str | None = None
    region: str | None = None
    cultural_sphere: str | None = None
    language: str | None = None
    education_level: str | None = None
    source_url: str | None = None
    research_notes: str | None = None
    metadata_json: dict[str, Any] | None = None


class CorpusDocumentUpdate(BaseModel):
    title: str | None = None
    organization: str | None = None
    organization_type: str | None = None
    publication_year: int | None = None
    publication_type: str | None = None
    country: str | None = None
    region: str | None = None
    cultural_sphere: str | None = None
    language: str | None = None
    education_level: str | None = None
    source_url: str | None = None
    research_notes: str | None = None
    metadata_json: dict[str, Any] | None = None


class CorpusDocumentResponse(BaseModel):
    id: str
    corpus_id: str
    rag_document_id: str
    title: str | None
    organization: str | None
    organization_type: str | None
    publication_year: int | None
    publication_type: str | None
    country: str | None
    region: str | None
    cultural_sphere: str | None
    language: str | None
    education_level: str | None
    source_url: str | None
    research_notes: str | None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkMetadataUpdate(BaseModel):
    document_ids: list[str]
    fields: dict[str, Any]


class MetadataImportResponse(BaseModel):
    updated: int
    errors: list[str]
    rows_processed: int


class SourceTextResponse(BaseModel):
    document_id: str
    text: str


class SegmentRequest(BaseModel):
    unit_type: str = Field(description="document | paragraph | sentence")


class PreprocessingProfileCreate(BaseModel):
    name: str
    description: str | None = None
    config: dict[str, Any]


class PreprocessingProfileUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None


class PreprocessingProfileResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    config: dict[str, Any]
    created_by: str
    created_at: datetime
    updated_at: datetime


class PreprocessingPreviewRequest(BaseModel):
    project_id: str | None = None
    corpus_id: str | None = None
    unit_type: str | None = None
    texts: list[str] | None = None
    config: dict[str, Any] | None = None
    preprocessing_profile_id: str | None = None
    sample_size: int = Field(default=5, ge=1, le=20)


class PreprocessingPreviewResponse(BaseModel):
    rows: list[dict[str, Any]]
    token_count_before: int
    token_count_after: int
    vocabulary_size: int
    most_frequently_removed_terms: list[dict[str, Any]]
    config: dict[str, Any]
    stemmer: str
    lemmatization_supported: bool
    profile_name: str | None = None
    profile_updated_at: str | None = None


class CodebookCreate(BaseModel):
    name: str
    description: str | None = None
    seed_demo_labels: bool = True


class CodebookResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    version: str
    is_frozen: bool
    created_by: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AnnotationLabelCreate(BaseModel):
    name: str
    description: str | None = None
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    positive_examples: list[str] | None = None
    negative_examples: list[str] | None = None


class AnnotationLabelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    positive_examples: list[str] | None = None
    negative_examples: list[str] | None = None


class AnnotationLabelResponse(BaseModel):
    id: str
    codebook_id: str
    name: str
    description: str | None
    inclusion_criteria: str | None
    exclusion_criteria: str | None
    positive_examples: list[str] | None = None
    negative_examples: list[str] | None = None
    is_placeholder: bool
    created_at: datetime


class AnnotationAssignRequest(BaseModel):
    text_unit_ids: list[str]
    annotator_ids: list[str]


class CorpusAnnotationAssignRequest(BaseModel):
    unit_type: str = "paragraph"
    annotator_ids: list[str] | None = None
    sample_size: int = Field(default=50, ge=1, le=5000)
    # Backward-compatible alias for sample_size
    limit: int | None = Field(default=None, ge=1, le=5000)
    strategy: str = Field(
        default="overlap",
        description="shared | disjoint | overlap",
    )
    overlap_count: int | None = Field(default=None, ge=0, le=5000)
    overlap_percent: float | None = Field(default=None, ge=0, le=100)


class CorpusAnnotationAssignResponse(BaseModel):
    assigned_count: int
    unique_units: int
    overlap_units: int
    strategy: str
    unit_type: str
    per_annotator: dict[str, int]


class TextUnitContextResponse(BaseModel):
    unit: dict[str, Any]
    document: dict[str, Any] | None = None
    before: list[dict[str, Any]] = Field(default_factory=list)
    after: list[dict[str, Any]] = Field(default_factory=list)


class AnnotationSaveRequest(BaseModel):
    text_unit_id: str
    codebook_id: str
    values: list[dict[str, Any]]
    mark_task_complete: bool = True


class AnnotationResponse(BaseModel):
    id: str
    text_unit_id: str
    label_id: str
    annotator_id: str
    value: str
    confidence: float | None
    comment: str | None
    codebook_version: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReliabilityRequest(BaseModel):
    codebook_id: str
    label_ids: list[str] | None = None


class AdjudicationSaveRequest(BaseModel):
    text_unit_id: str
    label_id: str
    final_value: str
    comment: str | None = None


class DatasetPreviewRequest(BaseModel):
    corpus_id: str
    unit_type: str
    codebook_id: str
    label_ids: list[str]
    annotation_source: str = "adjudicated_only"
    selected_annotator_id: str | None = None
    minimum_agreement: float | None = None


class DatasetFreezeRequest(DatasetPreviewRequest):
    name: str


class DatasetPreviewResponse(BaseModel):
    unit_count: int
    document_count: int
    unit_ids: list[str]
    document_ids: list[str]
    class_distribution: dict[str, dict[str, int]]
    unit_labels: dict[str, list[str]]
    missing_labels: list[dict[str, Any]]
    excluded_disagreements: list[dict[str, Any]]
    annotator_coverage: list[str]
    warnings: list[str]


class TrainingDatasetSnapshotResponse(BaseModel):
    id: str
    project_id: str
    corpus_id: str
    name: str
    unit_type: str
    codebook_id: str
    codebook_version: str
    annotation_source: str
    minimum_agreement: float | None
    created_by: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CorpusFilters(BaseModel):
    organization: str | None = None
    organization_type: str | None = None
    publication_year_min: int | None = None
    publication_year_max: int | None = None
    region: str | None = None
    cultural_sphere: str | None = None
    language: str | None = None
    publication_type: str | None = None


class AnalysisRequest(CorpusFilters):
    unit_type: str
    preprocessing_profile_id: str | None = None
    run_async: bool = False


class FrequencyRequest(AnalysisRequest):
    top_n: int = 50


class NgramRequest(AnalysisRequest):
    n: int = 2
    top_n: int = 50


class DfmRequest(AnalysisRequest):
    weighting: str = "count"
    run_async: bool = True


class KwicRequest(AnalysisRequest):
    keyword: str
    window_size: int = 5
    case_sensitive: bool = False


class DictionaryAnalysisRequest(AnalysisRequest):
    dictionary_id: str | None = None
    dictionary_terms: list[str] | None = None
    group_by: str | None = None
    run_async: bool = True


class KeynessRequest(BaseModel):
    unit_type: str
    preprocessing_profile_id: str | None = None
    filters_a: dict[str, Any]
    filters_b: dict[str, Any]
    top_n: int = 50
    run_async: bool = True


class CooccurrenceRequest(AnalysisRequest):
    window_size: int = 5
    top_n: int = 50
    run_async: bool = True


class ClassifierTrainRequest(BaseModel):
    snapshot_id: str
    algorithm: str = "logistic_regression"
    preprocessing_profile_id: str | None = None
    ngram_max: int = 1
    min_df: float | int = 1
    max_df: float | int = 1.0
    max_features: int | None = None
    class_weight: str | None = None
    regularization_c: float = 1.0
    test_size: float = 0.25
    random_seed: int = 42
    name: str | None = None
    run_async: bool = True


class ClassifierPredictRequest(BaseModel):
    unit_type: str
    only_unannotated: bool = False
    filters: dict[str, Any] | None = None


class ActiveLearningAssignRequest(BaseModel):
    """Assign uncertainty-ranked predictions to one or more annotators."""

    text_unit_ids: list[str] = Field(min_length=1)
    annotator_ids: list[str] = Field(min_length=1)


class TrainedModelResponse(BaseModel):
    id: str
    project_id: str
    corpus_id: str
    analysis_run_id: str
    training_dataset_snapshot_id: str
    model_family: str
    task_type: str
    label_ids: list[str]
    feature_config: dict[str, Any]
    training_config: dict[str, Any]
    metrics: dict[str, Any]
    version: int
    name: str | None
    created_by: str
    created_at: datetime


class TopicTrainRequest(CorpusFilters):
    unit_type: str
    algorithm: str = "lda"
    n_topics: int = 5
    preprocessing_profile_id: str | None = None
    max_iterations: int = 25
    random_seed: int = 42
    run_async: bool = True


class TopicLabelRequest(BaseModel):
    topic_id: int
    human_name: str


class RobustnessRequest(BaseModel):
    snapshot_id: str
    algorithm: str = "logistic_regression"
    seeds: list[int] | None = None
    cv_folds: int = 5
    class_weights: list[str | None] | None = None
    test_size: float = 0.25
    run_async: bool = True


class ComparativeAnalysisRequest(CorpusFilters):
    unit_type: str
    codebook_id: str
    label_ids: list[str]
    group_by: str
    provenance_mode: str = "human_only"
    model_id: str | None = None


class DictionaryCreate(BaseModel):
    name: str
    description: str | None = None
    terms: list[str]


class DictionaryUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    terms: list[str] | None = None


class DictionaryResponse(BaseModel):
    id: str
    project_id: str
    name: str
    version: str
    description: str | None
    terms: list[str]
    created_by: str
    created_at: datetime


class AnalysisRunResponse(BaseModel):
    id: str
    project_id: str
    corpus_id: str | None
    run_type: str
    status: str
    progress_stage: str | None
    parameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    artifact_path: str | None
    random_seed: int | None
    created_by: str
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime


class DemoSeedRequest(BaseModel):
    corpus_name: str | None = None


class ExportManifestResponse(BaseModel):
    manifest: dict[str, Any]


class QuantedaScriptResponse(BaseModel):
    script: str


class ContextualDatasetCreate(BaseModel):
    name: str
    description: str | None = None


class ContextualDatasetSummary(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    created_by: str
    created_at: datetime
    observation_count: int


class ContextualObservationResponse(BaseModel):
    id: str
    country: str | None
    year: int | None
    values: dict[str, Any]
    created_at: datetime


class ContextualDatasetDetail(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    created_by: str
    created_at: datetime
    observation_count: int
    indicator_keys: list[str]


class ContextualObservationPage(BaseModel):
    items: list[ContextualObservationResponse]
    total: int
    limit: int
    offset: int


class ContextualImportResponse(BaseModel):
    dataset_id: str
    imported: int
    skipped: int
    indicator_keys: list[str]
    replaced: bool


class ContextualLinkRequest(BaseModel):
    corpus_id: str
    codebook_id: str
    label_ids: list[str]
    indicator_key: str
    unit_type: str = "paragraph"
    group_by: str = "country"
    join_on_year: bool = True
    provenance_mode: str = "human_only"
    model_id: str | None = None
