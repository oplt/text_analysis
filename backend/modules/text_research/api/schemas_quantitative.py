"""Request schemas owned by quantitative, statistical, and measurement routes."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

MAX_RESULT_TOP_N = 10_000
MAX_PAIRWISE_TOP_K = 50_000
MAX_COOCCURRENCE_WINDOW = 100
MAX_KWIC_WINDOW = 100
MAX_KWIC_MATCHES = 50_000
MAX_CLUSTERS = 500
MAX_SVD_COMPONENTS = 500
MAX_MINHASH_PERM = 512
MAX_EMBEDDING_ITEMS = 10_000
MAX_EMBEDDING_DIMS = 4_096
MAX_EMBEDDING_TOTAL_FLOATS = 5_000_000
MAX_STATISTICAL_ROWS = 50_000
MAX_STATISTICAL_INDEPENDENT_VARS = 64
MAX_MEASUREMENT_VALUES = 100_000


class CorpusFilters(BaseModel):
    document_ids: list[str] | None = None
    organization: str | None = None
    organization_type: str | None = None
    publication_year: int | None = None
    publication_year_min: int | None = None
    publication_year_max: int | None = None
    country: str | None = None
    region: str | None = None
    cultural_sphere: str | None = None
    language: str | None = None
    publication_type: str | None = None


class AnalysisRequest(CorpusFilters):
    unit_type: str
    preprocessing_profile_id: str | None = None
    run_async: bool = False


class FrequencyRequest(AnalysisRequest):
    top_n: int = Field(default=50, ge=1, le=MAX_RESULT_TOP_N)
    rate_per: float = Field(default=1000, gt=0)
    group_by: str | None = None


class NgramRequest(AnalysisRequest):
    n: int = Field(default=2, ge=1, le=10)
    top_n: int = Field(default=50, ge=1, le=MAX_RESULT_TOP_N)
    rate_per: float = Field(default=1000, gt=0)
    skip: int = Field(default=0, ge=0)


class DfmTrimConfig(BaseModel):
    min_term_frequency: float | None = None
    max_term_frequency: float | None = None
    term_frequency_type: str = "count"
    min_document_frequency: float | None = None
    max_document_frequency: float | None = None
    document_frequency_type: str = "count"
    top_n: int | None = Field(default=None, ge=1, le=MAX_RESULT_TOP_N)


class DfmRequest(AnalysisRequest):
    weighting: str = "count"
    k1: float | None = Field(default=None, gt=0)
    b: float | None = Field(default=None, ge=0, le=1)
    smooth_idf: bool | None = None
    force_sparse_only: bool = False
    trim: DfmTrimConfig | None = None
    run_async: bool = True


class KwicRequest(AnalysisRequest):
    keyword: str
    window_size: int = Field(default=5, ge=0, le=MAX_KWIC_WINDOW)
    case_sensitive: bool = False
    query_mode: str = "auto"
    query_language: str | None = Field(
        default=None, validation_alias=AliasChoices("query_language", "kwic_language")
    )
    token_attribute: str | None = None
    max_matches: int | None = Field(default=None, ge=1, le=MAX_KWIC_MATCHES)

    @model_validator(mode="after")
    def _query_language(self) -> KwicRequest:
        if (self.query_mode == "lemma" or self.token_attribute == "lemma") and not (
            self.query_language or self.language
        ):
            raise ValueError("query_language (or kwic_language) is required for lemma matching")
        if not self.query_language and self.language:
            self.query_language = self.language
        return self


class DictionaryAnalysisRequest(AnalysisRequest):
    dictionary_id: str | None = None
    dictionary_terms: list[str] | None = None
    hierarchy: dict[str, Any] | None = None
    exclusions: list[Any] | None = None
    dictionary_language: str | None = None
    case_sensitive: bool = False
    rate_per: float = Field(default=1000, gt=0)
    group_by: str | None = None
    run_async: bool = False

    @model_validator(mode="after")
    def _source(self) -> DictionaryAnalysisRequest:
        if not self.dictionary_id and not self.dictionary_terms and not self.hierarchy:
            raise ValueError(
                "dictionary_id, dictionary_terms, or hierarchy required (user-defined only)"
            )
        return self


class KeynessRequest(BaseModel):
    unit_type: str
    preprocessing_profile_id: str | None = None
    filters_a: dict[str, Any]
    filters_b: dict[str, Any]
    group_field: str | None = None
    method: str = "log_likelihood"
    correction: str = "bh"
    min_frequency: int = Field(default=1, ge=0)
    top_n: int = Field(default=50, ge=1, le=MAX_RESULT_TOP_N)
    run_async: bool = True


class CooccurrenceRequest(AnalysisRequest):
    window_size: int = Field(default=5, ge=1, le=MAX_COOCCURRENCE_WINDOW)
    top_n: int = Field(default=50, ge=1, le=MAX_RESULT_TOP_N)
    association_method: str = "pmi"
    directional: bool = False
    min_frequency: int = Field(default=1, ge=0)
    min_count: int = Field(default=1, ge=0)
    include_network: bool = True
    run_async: bool = True


class SimilarityRequest(AnalysisRequest):
    method: str = "tfidf_cosine"
    mode: str = "pairwise"
    top_k: int | None = Field(default=20, ge=1, le=MAX_PAIRWISE_TOP_K)
    min_score: float | None = None
    group_by: str | None = None
    centroid_target: str = "between_groups"
    query_text: str | None = None
    query_unit_id: str | None = None
    embeddings: dict[str, list[float]] | None = None
    query_embedding: list[float] | None = None
    embedding_artifact_id: str | None = None
    run_async: bool = False

    @field_validator("embeddings")
    @classmethod
    def _bound_embeddings(
        cls, value: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        if value is None:
            return value
        if len(value) > MAX_EMBEDDING_ITEMS:
            raise ValueError(f"embeddings may contain at most {MAX_EMBEDDING_ITEMS} unit vectors")
        dimensions = {len(vector) for vector in value.values()}
        if len(dimensions) > 1:
            raise ValueError("embedding vectors must have the same dimensionality")
        if dimensions and (next(iter(dimensions)) > MAX_EMBEDDING_DIMS or sum(dimensions) == 0):
            raise ValueError("embedding vector dimensionality is invalid")
        if sum(len(vector) for vector in value.values()) > MAX_EMBEDDING_TOTAL_FLOATS:
            raise ValueError("embedding payload exceeds total-float ceiling")
        return value

    @model_validator(mode="after")
    def _embedding_source(self) -> SimilarityRequest:
        if self.embeddings is not None and self.embedding_artifact_id is not None:
            raise ValueError("Provide either embeddings or embedding_artifact_id, not both")
        if self.method == "embedding_cosine" and not (
            self.embeddings is not None or self.embedding_artifact_id
        ):
            raise ValueError("embedding_cosine requires embeddings or embedding_artifact_id")
        if self.method == "embedding_cosine" and self.mode == "pairwise" and not self.top_k:
            raise ValueError(
                "pairwise embedding_cosine requires top_k (dense all-pairs export is not allowed)"
            )
        return self


class DuplicateDetectionRequest(AnalysisRequest):
    methods: list[str] | None = None
    lexical_threshold: float = Field(default=0.85, ge=0, le=1)
    char_ngram_size: int = Field(default=5, ge=1, le=10)
    use_minhash: bool = False
    minhash_num_perm: int = Field(default=64, ge=1, le=MAX_MINHASH_PERM)
    minhash_shingle_size: int = Field(default=3, ge=1, le=10)
    minhash_threshold: float = Field(default=0.8, ge=0, le=1)
    max_pairs: int | None = Field(default=None, ge=1)
    run_async: bool = True


class ClusteringRequest(AnalysisRequest):
    n_clusters: int = Field(default=5, ge=2, le=MAX_CLUSTERS)
    algorithm: str = "kmeans"
    use_svd: bool = False
    n_svd_components: int = Field(default=50, ge=2, le=MAX_SVD_COMPONENTS)
    top_terms: int = Field(default=10, ge=1, le=MAX_RESULT_TOP_N)
    random_seed: int = 42
    run_async: bool = False

    @model_validator(mode="after")
    def _svd_vs_clusters(self) -> ClusteringRequest:
        if self.use_svd and self.n_svd_components < self.n_clusters:
            raise ValueError("n_svd_components must be >= n_clusters when use_svd is true")
        return self


class DimensionalityReductionRequest(AnalysisRequest):
    method: str = "svd"
    n_components: int = Field(default=2, ge=2, le=3)
    random_seed: int = 42
    run_async: bool = False


class ReadabilityRequest(AnalysisRequest):
    run_async: bool = False


class StatisticalModelRequest(BaseModel):
    model: str = "ols"
    dependent_var: str
    independent_vars: list[str] = Field(min_length=1, max_length=MAX_STATISTICAL_INDEPENDENT_VARS)
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_STATISTICAL_ROWS)
    add_intercept: bool = True

    @model_validator(mode="after")
    def _columns(self) -> StatisticalModelRequest:
        missing = [
            name
            for name in [self.dependent_var, *self.independent_vars]
            if name not in self.rows[0]
        ]
        if missing:
            raise ValueError(f"rows missing required columns: {missing}")
        return self


class MeasurementComparisonRequest(BaseModel):
    source_a: str
    values_a: list[Any] = Field(min_length=1, max_length=MAX_MEASUREMENT_VALUES)
    source_b: str
    values_b: list[Any] = Field(min_length=1, max_length=MAX_MEASUREMENT_VALUES)
    ids: list[str] | None = None
    value_kind: str = "categorical"
    subgroup: list[str] | None = None

    @model_validator(mode="after")
    def _aligned(self) -> MeasurementComparisonRequest:
        if len(self.values_a) != len(self.values_b):
            raise ValueError("values_a and values_b must have the same length")
        if self.ids is not None and len(self.ids) != len(self.values_a):
            raise ValueError("ids length must match values_a / values_b")
        if self.subgroup is not None and len(self.subgroup) != len(self.values_a):
            raise ValueError("subgroup length must match values_a / values_b")
        return self
