"""Canonical scientific-identity parameters for quantitative analyses."""

from __future__ import annotations

from typing import Any

QUANTITATIVE_PARAMETER_SCHEMA_VERSION = 1


_DEFAULTS_BY_OPERATION: dict[str, dict[str, Any]] = {
    "frequencies": {"top_n": 50, "rate_per": 1000, "group_by": None},
    "ngrams": {"n": 2, "top_n": 50, "rate_per": 1000, "skip": 0},
    "dfm": {
        "weighting": "count",
        "k1": None,
        "b": None,
        "smooth_idf": None,
        "force_sparse_only": False,
        "trim": None,
    },
    "keyness": {
        "filters_a": {},
        "filters_b": {},
        "group_field": None,
        "method": "log_likelihood",
        "correction": "bh",
        "min_frequency": 1,
        "top_n": 50,
    },
    "cooccurrence": {
        "window_size": 5,
        "top_n": 50,
        "association_method": "pmi",
        "directional": False,
        "min_frequency": 1,
        "min_count": 1,
        "include_network": True,
    },
    "similarity": {
        "method": "tfidf_cosine",
        "mode": "pairwise",
        "top_k": 20,
        "min_score": None,
        "group_by": None,
        "centroid_target": "between_groups",
        "query_text": None,
        "query_unit_id": None,
    },
    "clustering": {
        "n_clusters": 5,
        "algorithm": "kmeans",
        "use_svd": False,
        "n_svd_components": 50,
        "top_terms": 10,
        "random_seed": 42,
    },
    "dimensionality_reduction": {"method": "svd", "n_components": 2, "random_seed": 42},
    "duplicate_detection": {
        "methods": None,
        "lexical_threshold": 0.85,
        "char_ngram_size": 5,
        "use_minhash": False,
        "minhash_num_perm": 64,
        "minhash_shingle_size": 3,
        "minhash_threshold": 0.8,
        "max_pairs": None,
    },
}


def normalize_quantitative_run_parameters(
    operation: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Return the versioned, model-dump-shaped persisted quantitative parameters.

    Schema defaults are applied only for absent fields; explicit ``None`` values
    remain part of the reproducible request snapshot.
    """
    normalized = {**_DEFAULTS_BY_OPERATION.get(operation, {}), **parameters}
    normalized.setdefault("filters", {})
    normalized["parameter_schema_version"] = QUANTITATIVE_PARAMETER_SCHEMA_VERSION
    return normalized


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
