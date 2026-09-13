"""Canonical quantitative analysis execution configs (sole default owner).

API request models may mirror these Field defaults for OpenAPI ergonomics, and
the frontend may offer convenience UI defaults, but persisted run parameters and
worker re-entry always canonicalize through these models.

Bump ``QUANTITATIVE_PARAMETER_SCHEMA_VERSION`` when field defaults or meanings
change; :func:`migrate_quantitative_parameters` re-validates older snapshots.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

QUANTITATIVE_PARAMETER_SCHEMA_VERSION = 1


class QuantitativeConfig(BaseModel):
    """Base for execution knobs; ignore non-config keys (filters, identity, …)."""

    model_config = ConfigDict(extra="ignore")


class FrequenciesConfig(QuantitativeConfig):
    top_n: int = 50
    rate_per: float = 1000
    group_by: str | None = None


class NgramsConfig(QuantitativeConfig):
    n: int = 2
    top_n: int = 50
    rate_per: float = 1000
    skip: int = 0


class DfmConfig(QuantitativeConfig):
    weighting: str = "count"
    k1: float | None = None
    b: float | None = None
    smooth_idf: bool | None = None
    force_sparse_only: bool = False
    trim: dict[str, Any] | None = None


class KeynessConfig(QuantitativeConfig):
    filters_a: dict[str, Any] = Field(default_factory=dict)
    filters_b: dict[str, Any] = Field(default_factory=dict)
    group_field: str | None = None
    method: str = "log_likelihood"
    correction: str = "bh"
    min_frequency: int = 1
    top_n: int = 50


class CooccurrenceConfig(QuantitativeConfig):
    window_size: int = 5
    top_n: int = 50
    association_method: str = "pmi"
    directional: bool = False
    min_frequency: int = 1
    min_count: int = 1
    include_network: bool = True


class SimilarityConfig(QuantitativeConfig):
    method: str = "tfidf_cosine"
    mode: str = "pairwise"
    top_k: int = 20
    min_score: float | None = None
    group_by: str | None = None
    centroid_target: str = "between_groups"
    query_text: str | None = None
    query_unit_id: str | None = None
    embedding_artifact_id: str | None = None


class ClusteringConfig(QuantitativeConfig):
    n_clusters: int = 5
    algorithm: str = "kmeans"
    use_svd: bool = False
    n_svd_components: int = 50
    top_terms: int = 10
    random_seed: int = 42


class DimensionalityReductionConfig(QuantitativeConfig):
    method: str = "svd"
    n_components: int = 2
    random_seed: int = 42


class DuplicateDetectionConfig(QuantitativeConfig):
    """Normalized duplicate-detection execution snapshot."""

    methods: list[str] | None = None
    lexical_threshold: float = Field(default=0.85, ge=0, le=1)
    char_ngram_size: int = Field(default=5, ge=1)
    use_minhash: bool = False
    minhash_num_perm: int = Field(default=64, ge=1)
    minhash_shingle_size: int = Field(default=3, ge=1)
    minhash_threshold: float = Field(default=0.8, ge=0, le=1)
    # Cap pairwise lexical/MinHash work; ``None`` means unlimited (explicit).
    max_pairs: int | None = 1000


CONFIG_BY_OPERATION: dict[str, type[QuantitativeConfig]] = {
    "frequencies": FrequenciesConfig,
    "ngrams": NgramsConfig,
    "dfm": DfmConfig,
    "keyness": KeynessConfig,
    "cooccurrence": CooccurrenceConfig,
    "similarity": SimilarityConfig,
    "clustering": ClusteringConfig,
    "dimensionality_reduction": DimensionalityReductionConfig,
    "duplicate_detection": DuplicateDetectionConfig,
}


def config_defaults(operation: str) -> dict[str, Any]:
    """Return the current default dump for ``operation`` (empty if unknown)."""
    config_cls = CONFIG_BY_OPERATION.get(operation)
    if config_cls is None:
        return {}
    return config_cls().model_dump()


def normalize_quantitative_run_parameters(
    operation: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Return versioned, config-validated parameters for persistence / re-entry.

    Missing knobs receive Config defaults. Explicit ``None`` values for optional
    fields are preserved as part of the reproducible snapshot.
    """
    config_cls = CONFIG_BY_OPERATION.get(operation)
    if config_cls is None:
        normalized = dict(parameters)
    else:
        config_dump = config_cls.model_validate(parameters).model_dump()
        extras = {
            key: value for key, value in parameters.items() if key not in config_cls.model_fields
        }
        normalized = {**config_dump, **extras}
    normalized.setdefault("filters", {})
    normalized.setdefault("preprocessing_profile_id", None)
    normalized["parameter_schema_version"] = QUANTITATIVE_PARAMETER_SCHEMA_VERSION
    return normalized


def migrate_quantitative_parameters(operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a persisted snapshot to the current parameter schema.

    Older or incomplete snapshots (missing ``parameter_schema_version`` or an
    earlier version) are re-validated through the current Config so workers and
    rerun adapters never invent ad-hoc fallbacks.
    """
    return normalize_quantitative_run_parameters(operation, parameters)
