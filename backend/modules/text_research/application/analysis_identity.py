"""Canonical scientific-identity parameters for quantitative analyses."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.domain.quantitative_configs import (
    QUANTITATIVE_PARAMETER_SCHEMA_VERSION,
    migrate_quantitative_parameters,
    normalize_quantitative_run_parameters,
)

# Re-export for existing imports / tests.
__all__ = [
    "QUANTITATIVE_PARAMETER_SCHEMA_VERSION",
    "bind_quantitative_execution_kwargs",
    "migrate_quantitative_parameters",
    "normalize_analysis_parameters",
    "normalize_quantitative_run_parameters",
    "quantitative_execution_snapshot",
]


# Service-method kwargs for worker / rerun re-entry. Defaults live only in
# ``domain.quantitative_configs`` — callers must migrate/normalize before binding.
_EXECUTION_KEYS_BY_OPERATION: dict[str, tuple[str, ...]] = {
    "frequencies": ("unit_type", "preprocessing_profile_id", "top_n", "rate_per", "group_by"),
    "ngrams": ("unit_type", "preprocessing_profile_id", "n", "top_n", "rate_per", "skip"),
    "dfm": (
        "unit_type",
        "preprocessing_profile_id",
        "weighting",
        "k1",
        "b",
        "smooth_idf",
        "force_sparse_only",
        "trim",
    ),
    "keyness": (
        "unit_type",
        "preprocessing_profile_id",
        "filters_a",
        "filters_b",
        "group_field",
        "method",
        "correction",
        "min_frequency",
        "top_n",
    ),
    "cooccurrence": (
        "unit_type",
        "preprocessing_profile_id",
        "window_size",
        "top_n",
        "association_method",
        "directional",
        "min_frequency",
        "min_count",
        "include_network",
    ),
    "similarity": (
        "unit_type",
        "preprocessing_profile_id",
        "method",
        "mode",
        "top_k",
        "min_score",
        "group_by",
        "centroid_target",
        "query_text",
        "query_unit_id",
        "embedding_artifact_id",
    ),
    "clustering": (
        "unit_type",
        "preprocessing_profile_id",
        "n_clusters",
        "algorithm",
        "use_svd",
        "n_svd_components",
        "top_terms",
        "random_seed",
    ),
    "dimensionality_reduction": (
        "unit_type",
        "preprocessing_profile_id",
        "method",
        "n_components",
        "random_seed",
    ),
    "duplicate_detection": (
        "unit_type",
        "methods",
        "lexical_threshold",
        "char_ngram_size",
        "use_minhash",
        "minhash_num_perm",
        "minhash_shingle_size",
        "minhash_threshold",
        "max_pairs",
    ),
}

_FILTER_SPREAD_OPERATIONS = frozenset(_EXECUTION_KEYS_BY_OPERATION) - {"keyness"}


def bind_quantitative_execution_kwargs(
    operation: str, parameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind service kwargs from a (possibly incomplete) persisted snapshot.

    Defaults come solely from :func:`migrate_quantitative_parameters` /
    domain Config models. This binder never invents fallback constants.
    """
    keys = _EXECUTION_KEYS_BY_OPERATION.get(operation)
    if keys is None:
        raise ValueError(f"Unsupported quantitative operation {operation!r}")
    if not parameters.get("unit_type"):
        raise ValueError(f"AnalysisRun missing required unit_type for {operation}")
    normalized = migrate_quantitative_parameters(operation, parameters)
    kwargs = {key: normalized[key] for key in keys}
    filters = (
        dict(normalized.get("filters") or {}) if operation in _FILTER_SPREAD_OPERATIONS else {}
    )
    return kwargs, filters


def quantitative_execution_snapshot(operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalized execution payload for adapters / enqueue (includes filters)."""
    kwargs, filters = bind_quantitative_execution_kwargs(operation, parameters)
    snapshot = dict(kwargs)
    if operation in _FILTER_SPREAD_OPERATIONS:
        snapshot["filters"] = filters
    return snapshot


_PARAMETERS_BY_OPERATION: dict[str, tuple[str, ...]] = {
    "corpus_stats": ("group_by",),
    "frequencies": ("top_n", "rate_per", "group_by"),
    "ngrams": ("n", "top_n", "rate_per", "skip"),
    "dfm": ("weighting", "trim", "k1", "b", "smooth_idf", "force_sparse_only"),
    "kwic": (
        "keyword",
        "window_size",
        "case_sensitive",
        "query_mode",
        "query_language",
        "token_attribute",
        "max_matches",
    ),
    "dictionary": (
        "dictionary_id",
        "dictionary_version",
        "dictionary_content_checksum",
        "exclusions",
        "language",
        "case_sensitive",
        "rate_per",
        "group_by",
        "terms",
        "hierarchy",
    ),
    "keyness": (
        "filters_a",
        "filters_b",
        "group_field",
        "method",
        "correction",
        "min_frequency",
        "top_n",
    ),
    "cooccurrence": (
        "window_size",
        "top_n",
        "association_method",
        "directional",
        "min_frequency",
        "min_count",
        "include_network",
    ),
    "similarity": (
        "method",
        "mode",
        "group_by",
        "top_k",
        "min_score",
        "centroid_target",
        "query_text",
        "query_unit_id",
        "query_embedding_checksum",
        "embedding_checksum",
        "embedding_artifact_id",
        "embedding_artifact_checksum",
        "embedding_identity",
        "has_embeddings",
    ),
    "clustering": (
        "n_clusters",
        "algorithm",
        "use_svd",
        "n_svd_components",
        "top_terms",
        "random_seed",
    ),
    "dimensionality_reduction": ("method", "n_components", "random_seed"),
    "duplicate_detection": (
        "methods",
        "lexical_threshold",
        "char_ngram_size",
        "use_minhash",
        "minhash_num_perm",
        "minhash_shingle_size",
        "minhash_threshold",
        "max_pairs",
    ),
    # Selection + preprocessing identity only; no analysis knobs today.
    "readability": (),
    "statistical_model": (
        "model",
        "dependent_var",
        "independent_vars",
        "add_intercept",
        "input_artifact_id",
        "input_artifact_checksum",
    ),
    "measurement_validation": (
        "source_a",
        "source_b",
        "value_kind",
        "input_artifact_id",
        "input_artifact_checksum",
    ),
}


def normalize_analysis_parameters(run_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Select only output-changing parameters for an analysis specification.

    Scheduling controls (for example ``run_async``) intentionally never enter
    this payload, so changing how a deterministic operation is executed does
    not change its scientific identity.
    """
    keys = _PARAMETERS_BY_OPERATION.get(run_type)
    if keys is None:
        return dict(payload)
    return {key: payload.get(key) for key in keys}
