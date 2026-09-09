"""Pydantic schemas for the text research API."""

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
    canonical_text_checksum: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    source: str = "canonical"


class SegmentRequest(BaseModel):
    unit_type: str = Field(description="document | paragraph | sentence")


class CleaningProfileCreate(BaseModel):
    name: str
    description: str | None = None
    version: str = "1.0"
    config: dict[str, Any]


class CleaningProfileUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    version: str | None = None
    config: dict[str, Any] | None = None


class CleaningProfileResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    version: str
    config: dict[str, Any]
    created_by: str
    created_at: datetime
    updated_at: datetime


class CleaningPreviewRequest(BaseModel):
    project_id: str | None = None
    texts: list[str] | None = None
    document_id: str | None = None
    config: dict[str, Any] | None = None
    cleaning_profile_id: str | None = None


class CleaningPreviewResponse(BaseModel):
    rows: list[dict[str, Any]]
    config: dict[str, Any]
    engine: str
    engine_version: str
    profile_name: str | None = None
    profile_version: str | None = None


class ApplyCleaningRequest(BaseModel):
    cleaning_profile_id: str
    document_ids: list[str] | None = None


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
    stemmer: str | None
    lemmatization_supported: bool
    implementation: dict[str, Any] | None = None
    profile_name: str | None = None
    profile_updated_at: str | None = None


class CodebookCreate(BaseModel):
    name: str
    description: str | None = None
    # Accepted for API compatibility; ignored — labels are always user-defined.
    seed_demo_labels: bool = False


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
    # Stratified annotation sampling (prompt.txt §24). All optional and
    # backward compatible: with no `stratify_by`, sampling is a seeded
    # random draw of `sample_size` units.
    random_seed: int | None = Field(
        default=None, description="Seed for reproducible sampling; auto-generated if omitted."
    )
    stratify_by: list[str] | None = Field(
        default=None,
        description="Metadata field names to stratify by, e.g. ['field_1', 'field_2']. Generic — any supported document metadata field works.",
    )
    stratum_mode: str = Field(default="proportional", description="proportional | equal")
    sampling_level: str = Field(default="unit", description="unit | document")
    max_units_per_document: int | None = Field(default=None, ge=1)


class CorpusAnnotationAssignResponse(BaseModel):
    assigned_count: int
    unique_units: int
    overlap_units: int
    strategy: str
    unit_type: str
    per_annotator: dict[str, int]
    sampling_plan: dict[str, Any] | None = None


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
    text_hashes: dict[str, str] = {}
    corpus_checksums: dict[str, str | None] = {}
    corpus_checksum_aggregate: str | None = None


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
    rate_per: float = Field(
        default=1000,
        gt=0,
        description="Normalize rates per this many tokens (100, 1000, 10000, or arbitrary).",
    )
    group_by: str | None = Field(
        default=None,
        description="Optional document metadata field for group token totals (e.g. organization).",
    )


class NgramRequest(AnalysisRequest):
    n: int = Field(default=2, ge=1, le=10, description="N-gram order (1=unigram … 10 max).")
    top_n: int = 50
    rate_per: float = Field(
        default=1000,
        gt=0,
        description="Normalize rates per this many n-gram tokens.",
    )
    skip: int = Field(
        default=0,
        ge=0,
        description="Skip-gram gap size. Only 0 (contiguous) is supported currently.",
    )


class DfmTrimConfig(BaseModel):
    """Optional post-build DFM feature trimming (quanteda ``dfm_trim`` analogue)."""

    min_term_frequency: float | None = None
    max_term_frequency: float | None = None
    term_frequency_type: str = Field(
        default="count",
        description="count | prop | rank | quantile",
    )
    min_document_frequency: float | None = None
    max_document_frequency: float | None = None
    document_frequency_type: str = Field(
        default="count",
        description="count | prop | rank | quantile",
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="Keep top-N features by term frequency (rank convenience).",
    )


class DfmRequest(AnalysisRequest):
    weighting: str = Field(
        default="count",
        description="count | binary | tf | tfidf | sublinear_tf | log_count | bm25",
    )
    k1: float | None = Field(
        default=None,
        gt=0,
        description="BM25 k1 (term-frequency saturation); default 1.5 when weighting=bm25.",
    )
    b: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="BM25 b (length normalization); default 0.75 when weighting=bm25.",
    )
    smooth_idf: bool | None = Field(
        default=None,
        description="Smooth IDF for tfidf / sublinear_tf (sklearn-compatible when true).",
    )
    force_sparse_only: bool = Field(
        default=False,
        description="If true, never attach a dense_matrix even for small DFMs.",
    )
    trim: DfmTrimConfig | None = None
    run_async: bool = True


class KwicRequest(AnalysisRequest):
    keyword: str = Field(description="Query string (word, phrase, regex, or wildcard).")
    window_size: int = Field(default=5, ge=0, description="Token context width on each side.")
    case_sensitive: bool = False
    query_mode: str = Field(
        default="auto",
        description="auto | word | phrase | exact_phrase | regex | wildcard | lemma",
    )
    language: str | None = Field(
        default=None,
        description="Language code for lemma matching (required when query_mode=lemma).",
    )
    token_attribute: str | None = Field(
        default=None,
        description="Optional token attribute: surface (default) or lemma.",
    )
    max_matches: int | None = Field(default=None, ge=1)


class DictionaryAnalysisRequest(AnalysisRequest):
    dictionary_id: str | None = None
    dictionary_terms: list[str] | None = None
    hierarchy: dict[str, Any] | None = Field(
        default=None,
        description="Optional inline hierarchical dictionary (user-defined).",
    )
    exclusions: list[Any] | None = None
    dictionary_language: str | None = Field(
        default=None,
        description="Optional language for the dictionary definition (not a corpus filter).",
    )
    case_sensitive: bool = False
    rate_per: float = Field(default=1000.0, gt=0)
    group_by: str | None = None
    run_async: bool = True


class KeynessRequest(BaseModel):
    unit_type: str
    preprocessing_profile_id: str | None = None
    filters_a: dict[str, Any] = Field(
        description="Metadata filters selecting group A (from corpus facets; not hardcoded)."
    )
    filters_b: dict[str, Any] = Field(
        description="Metadata filters selecting group B (from corpus facets; not hardcoded)."
    )
    group_field: str | None = Field(
        default=None,
        description="Optional metadata field name used for the comparison (for provenance).",
    )
    method: str = Field(
        default="log_likelihood",
        description="log_likelihood | chi_square | fisher",
    )
    correction: str = Field(
        default="bh",
        description="Multiple-testing correction: bh (Benjamini–Hochberg) | none",
    )
    min_frequency: int = Field(default=1, ge=0)
    top_n: int = 50
    run_async: bool = True


class CooccurrenceRequest(AnalysisRequest):
    window_size: int = Field(default=5, ge=1)
    top_n: int = 50
    association_method: str = Field(
        default="pmi",
        description="count | pmi | npmi | dice | log_dice | t_score",
    )
    directional: bool = Field(
        default=False,
        description="If true, keep ordered (forward-window) pairs; else undirected.",
    )
    min_frequency: int = Field(
        default=1,
        ge=0,
        description="Minimum unigram frequency for each term in a pair.",
    )
    min_count: int = Field(
        default=1,
        ge=0,
        description="Minimum raw co-occurrence count for a pair.",
    )
    include_network: bool = Field(
        default=True,
        description="If true, include graph-ready nodes/edges in the results.",
    )
    run_async: bool = True


class SimilarityRequest(AnalysisRequest):
    method: str = Field(
        default="tfidf_cosine",
        description="tfidf_cosine | jaccard | embedding_cosine",
    )
    mode: str = Field(
        default="pairwise",
        description=(
            "pairwise (document-to-document / unit-to-unit) | query "
            "(query-to-document) | group_centroid"
        ),
    )
    top_k: int | None = Field(default=20, ge=1, description="Limit returned rows per request.")
    min_score: float | None = Field(default=None, description="Drop rows below this score.")
    group_by: str | None = Field(
        default=None,
        description="Metadata field defining groups when mode=group_centroid.",
    )
    centroid_target: str = Field(
        default="between_groups",
        description="between_groups | item_to_own_group (only used when mode=group_centroid).",
    )
    query_text: str | None = Field(default=None, description="Ad hoc query text when mode=query.")
    query_unit_id: str | None = Field(
        default=None,
        description="Alternative to query_text: use an existing text unit's text as the query.",
    )
    embeddings: dict[str, list[float]] | None = Field(
        default=None,
        description=(
            "Required when method=embedding_cosine: mapping of text_unit_id -> vector. "
            "Never computed by this platform; supply vectors from an existing provider."
        ),
    )
    query_embedding: list[float] | None = Field(
        default=None,
        description="Required when mode=query and method=embedding_cosine.",
    )
    run_async: bool = False


class DuplicateDetectionRequest(AnalysisRequest):
    methods: list[str] | None = Field(
        default=None,
        description=(
            "Any of exact | normalized | lexical | minhash "
            "(default: exact, normalized, lexical)."
        ),
    )
    lexical_threshold: float = Field(default=0.85, ge=0, le=1)
    char_ngram_size: int = Field(default=5, ge=1)
    use_minhash: bool = Field(
        default=False, description="Convenience flag to add 'minhash' to methods."
    )
    minhash_num_perm: int = Field(default=64, ge=1)
    minhash_shingle_size: int = Field(default=3, ge=1)
    minhash_threshold: float = Field(default=0.8, ge=0, le=1)
    max_pairs: int | None = Field(default=1000, ge=1)
    run_async: bool = False


class ClusteringRequest(AnalysisRequest):
    n_clusters: int = Field(default=5, ge=2)
    algorithm: str = Field(default="kmeans", description="kmeans | minibatch_kmeans")
    use_svd: bool = False
    n_svd_components: int = Field(default=50, ge=2)
    top_terms: int = Field(default=10, ge=1)
    random_seed: int = 42
    run_async: bool = False


class DimensionalityReductionRequest(AnalysisRequest):
    method: str = Field(default="svd", description="svd | pca")
    n_components: int = Field(default=2, ge=2, le=3)
    random_seed: int = 42
    run_async: bool = False


class ReadabilityRequest(AnalysisRequest):
    run_async: bool = False


class ClassifierTrainRequest(BaseModel):
    snapshot_id: str
    algorithm: str = "logistic_regression"
    # User/config-driven task type (§27): "binary" | "multiclass" |
    # "multilabel". When omitted, the effective task type is inferred from
    # unambiguous gold-label shape at training time — never silently forced
    # to multilabel.
    task_type: str | None = None
    preprocessing_profile_id: str | None = None
    # Feature representation (§29): word and/or character n-grams, count or
    # TF-IDF weighted. Combinations are additive to the pre-existing fields
    # below (min_df/max_df/max_features apply to both feature families).
    vectorizer: str = "tfidf"
    use_word_ngrams: bool = True
    ngram_min: int = 1
    ngram_max: int = 1
    use_char_ngrams: bool = False
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    min_df: float | int = 1
    max_df: float | int = 1.0
    max_features: int | None = None
    class_weight: str | None = None
    regularization_c: float = 1.0
    # Naive Bayes (multinomial/complement) smoothing parameter (§31/§32).
    nb_alpha: float = 1.0
    # Only used when algorithm == "sgd_classifier".
    sgd_loss: str = "log_loss"
    test_size: float = 0.2
    # Fraction of the remaining (non-test) groups held out for validation
    # (§30). Set to 0 to disable the validation partition.
    val_size: float = 0.2
    random_seed: int = 42
    # §31 hyperparameter tuning: small grid/random search over C/alpha/
    # max_features, scored on VALIDATION only (or GroupKFold on train+val
    # groups when no validation partition exists). Never tunes on TEST.
    tune_hyperparameters: bool = False
    hyperparameter_search_type: str = "grid"  # "grid" | "random"
    hyperparameter_param_grid: dict[str, list[Any]] | None = None
    hyperparameter_n_iter: int = 10
    hyperparameter_scoring: str = "f1_macro"
    # §33 per-class/per-label threshold tuning, fit on VALIDATION only and
    # applied when scoring TEST and when predicting. Automatically skipped
    # (with a persisted note) when there is no validation partition.
    tune_thresholds: bool = True
    # §35 group-level bootstrap confidence intervals on TEST predictions.
    n_bootstrap: int = 200
    ci_confidence_level: float = 0.95
    # §36 calibration: diagnostics (ECE/reliability curve) are always
    # computed on TEST when probabilities are available; this selects the
    # method for the *optional* VAL-fit recalibration step.
    calibration_method: str = "sigmoid"  # "sigmoid" | "isotonic"
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


class TopicKSweepRequest(CorpusFilters):
    unit_type: str
    algorithm: str = "lda"
    k_values: list[int] = Field(..., min_length=1, description="n_topics values to compare.")
    preprocessing_profile_id: str | None = None
    max_iterations: int = 25
    random_seed: int = 42


class TopicSeedStabilityRequest(CorpusFilters):
    unit_type: str
    algorithm: str = "lda"
    n_topics: int = 5
    seeds: list[int] = Field(..., min_length=2, description="At least 2 seeds to compare.")
    preprocessing_profile_id: str | None = None
    max_iterations: int = 25


class RobustnessRequest(BaseModel):
    snapshot_id: str
    algorithm: str = "logistic_regression"
    seeds: list[int] | None = None
    cv_folds: int = 5
    class_weights: list[str | None] | None = None
    test_size: float = 0.25
    group_field: str = Field(
        default="organization",
        description=(
            "Document metadata field for leave-one-group-out (any CorpusDocument "
            "facet field or custom metadata_json key; NOT limited to organization)."
        ),
    )
    max_groups: int | None = Field(
        default=25, description="Cap on distinct group values swept by leave-one-group-out."
    )
    temporal_field: str = Field(
        default="publication_year",
        description="Document metadata field used for temporal holdout/expanding-window validation.",
    )
    temporal_windows: bool = Field(
        default=False, description="Also run expanding-window temporal validation."
    )
    transfer_field: str | None = Field(
        default=None,
        description="Optional metadata field for a generic transfer test (train on A, test on B).",
    )
    transfer_train_values: list[str] | None = Field(
        default=None, description="Values of transfer_field that define the train filter (A)."
    )
    transfer_test_values: list[str] | None = Field(
        default=None, description="Values of transfer_field that define the test filter (B)."
    )
    run_async: bool = True


class ComparativeAnalysisRequest(CorpusFilters):
    unit_type: str
    codebook_id: str
    label_ids: list[str]
    group_by: str
    provenance_mode: str = "human_only"
    model_id: str | None = None


class StatisticalModelRequest(BaseModel):
    """User-specified regression on supplied tabular rows (§50)."""

    model: str = Field(default="ols", description="ols | logistic")
    dependent_var: str
    independent_vars: list[str]
    rows: list[dict[str, Any]]
    add_intercept: bool = True


class MeasurementComparisonRequest(BaseModel):
    """Compare two user-aligned measurement series (§51)."""

    source_a: str
    values_a: list[Any]
    source_b: str
    values_b: list[Any]
    ids: list[str] | None = None
    value_kind: str = Field(default="categorical", description="categorical | continuous")
    subgroup: list[str] | None = None


class DictionaryCreate(BaseModel):
    name: str
    description: str | None = None
    version: str = "1.0"
    language: str | None = None
    terms: list[Any] | None = None
    hierarchy: dict[str, Any] | None = None
    exclusions: list[Any] | None = None


class DictionaryUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    version: str | None = None
    language: str | None = None
    terms: list[Any] | None = None
    hierarchy: dict[str, Any] | None = None
    exclusions: list[Any] | None = None


class DictionaryResponse(BaseModel):
    id: str
    project_id: str
    name: str
    version: str
    description: str | None
    language: str | None = None
    terms: list[str]
    hierarchy: dict[str, Any] | None = None
    exclusions: list[dict[str, Any]] = Field(default_factory=list)
    format: str = "flat"
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
