"""Canonical pure operators shared by API services and StageRunner/CLI.

Production FastAPI paths continue to use QuantitativeAnalysisService for
corpus selection, persistence, and async scheduling. StageRunner and the
headless CLI call these same operators on an already-prepared corpus so
analysis math does not drift between surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    PreparedCorpusArtifact,
)
from backend.modules.text_research.infrastructure.readability import readability_for_units


def tokenized_from_prepared(prepared: PreparedCorpusArtifact) -> list[list[str]]:
    return [list(seq) for seq in prepared.token_sequences]


def run_corpus_stats_operator(
    prepared: PreparedCorpusArtifact,
    *,
    mattr_window: int = quantitative.DEFAULT_MATTR_WINDOW,
    msttr_window: int = quantitative.DEFAULT_MSTTR_WINDOW,
) -> dict[str, Any]:
    """Compute descriptive statistics from a prepared corpus."""
    document_ids = list(prepared.document_ids)
    return quantitative.corpus_statistics(
        list(prepared.original_units),
        tokenized_from_prepared(prepared),
        document_ids=document_ids if any(document_ids) else None,
        mattr_window=mattr_window,
        msttr_window=msttr_window,
    )


def run_frequencies_operator(
    prepared: PreparedCorpusArtifact,
    *,
    top_n: int = 50,
    rate_per: float = 1000,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    return quantitative.term_frequency_report(
        tokenized_from_prepared(prepared),
        top_n=top_n,
        rate_per=rate_per,
        unit_ids=list(prepared.unit_ids),
        document_ids=list(prepared.document_ids) if prepared.document_ids else None,
        group_keys=group_keys,
    )


def run_ngrams_operator(
    prepared: PreparedCorpusArtifact,
    *,
    n: int = 2,
    top_n: int = 50,
    rate_per: float = 1000,
    skip: int = 0,
) -> dict[str, Any]:
    return quantitative.ngram_frequency_report(
        tokenized_from_prepared(prepared),
        n=n,
        top_n=top_n,
        rate_per=rate_per,
        skip=skip,
        unit_ids=list(prepared.unit_ids),
        document_ids=list(prepared.document_ids) if prepared.document_ids else None,
    )


def run_dfm_operator(
    prepared: PreparedCorpusArtifact,
    *,
    weighting: str = "count",
    **build_kwargs: Any,
) -> dict[str, Any]:
    result = quantitative.build_dfm(
        tokenized_from_prepared(prepared),
        weighting=weighting,
        unit_ids=list(prepared.unit_ids),
        preprocessing_config=prepared.preprocessing_profile,
        **build_kwargs,
    )
    return {"dfm": result, "summary": quantitative.dfm_summary(result)}


def run_kwic_operator(
    prepared: PreparedCorpusArtifact,
    *,
    keyword: str,
    window_size: int = 5,
    case_sensitive: bool = False,
    query_mode: str = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
    unit_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Prefer caller-supplied unit rows (already include text + provenance). Fall
    # back to prepared corpus fields for StageRunner / operator-only callers.
    if unit_metadata is not None:
        payload = [dict(row) for row in unit_metadata]
    else:
        payload = []
        for index, unit_id in enumerate(prepared.unit_ids):
            payload.append(
                {
                    "text": prepared.original_units[index],
                    "text_unit_id": unit_id,
                    "id": unit_id,
                }
            )
    matches = quantitative.kwic_search(
        payload,
        keyword,
        window_size=window_size,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )
    return {"matches": matches, "match_count": len(matches)}


def run_readability_operator(
    prepared: PreparedCorpusArtifact,
) -> dict[str, Any]:
    return readability_for_units(
        list(prepared.original_units),
        list(prepared.unit_ids),
    )


def run_dictionary_operator(
    prepared: PreparedCorpusArtifact,
    *,
    dictionary_spec: Any,
    unit_ids: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
    case_sensitive: bool = False,
    rate_per: float = 1000.0,
    group_keys: list[str] | None = None,
    dictionary_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.dictionary_matcher import match_dictionary

    result = match_dictionary(
        tokenized_from_prepared(prepared),
        dictionary_spec,
        unit_ids=unit_ids if unit_ids is not None else list(prepared.unit_ids),
        metadata=metadata,
        case_sensitive=case_sensitive,
        rate_per=rate_per,
    )
    if dictionary_metadata:
        result["dictionary"] = {
            **result.get("dictionary", {}),
            **dictionary_metadata,
        }
    if group_keys is not None:
        grouped: dict[str, dict[str, float | int]] = {}
        for group, row in zip(group_keys, result["per_unit"], strict=True):
            bucket = grouped.setdefault(group, {"hits": 0, "units": 0})
            bucket["hits"] = int(bucket["hits"]) + int(row["hits"])
            bucket["units"] = int(bucket["units"]) + 1
        result["by_group"] = grouped
    return result


def run_keyness_operator(
    prepared_a: PreparedCorpusArtifact,
    prepared_b: PreparedCorpusArtifact,
    *,
    method: str = "log_likelihood",
    top_n: int = 50,
    min_frequency: int = 1,
    correction: str | None = "bh",
    group_a_label: str | None = None,
    group_b_label: str | None = None,
    group_field: str | None = None,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.keyness import keyness_report

    return keyness_report(
        tokenized_from_prepared(prepared_a),
        tokenized_from_prepared(prepared_b),
        method=method,
        top_n=top_n,
        min_frequency=min_frequency,
        correction=correction,
        group_a_label=group_a_label,
        group_b_label=group_b_label,
        group_field=group_field,
    )


def run_cooccurrence_operator(
    prepared: PreparedCorpusArtifact,
    *,
    window_size: int = 5,
    top_n: int = 50,
    association_method: str = "pmi",
    directional: bool | str | None = False,
    min_frequency: int = 1,
    min_count: int = 1,
    include_network: bool = True,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.collocation import collocation_report

    return collocation_report(
        tokenized_from_prepared(prepared),
        window=window_size,
        top_n=top_n,
        association_method=association_method,
        directional=directional,
        min_frequency=min_frequency,
        min_count=min_count,
        include_network=include_network,
    )


def run_similarity_operator(
    prepared: PreparedCorpusArtifact,
    *,
    method: str = "tfidf_cosine",
    mode: str = "pairwise",
    group_keys: list[str] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    centroid_target: str = "between_groups",
    query_text: str | None = None,
    query_id: str = "query",
    embeddings: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
    exclude_unit_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch similarity modes over a prepared corpus."""
    from backend.modules.text_research.infrastructure import similarity as sim
    from backend.modules.text_research.infrastructure.preprocessing import tokenize

    canonical_method = sim.normalize_similarity_method(method)
    canonical_mode = sim.normalize_similarity_mode(mode)
    ids = list(prepared.unit_ids)
    tokenized = tokenized_from_prepared(prepared)
    if exclude_unit_id is not None:
        keep = [index for index, item_id in enumerate(ids) if item_id != exclude_unit_id]
        ids = [ids[index] for index in keep]
        tokenized = [tokenized[index] for index in keep]
        if group_keys is not None:
            group_keys = [group_keys[index] for index in keep]
    embed_vectors = [embeddings[item_id] for item_id in ids] if embeddings else None

    if canonical_mode == "pairwise":
        return sim.pairwise_similarity(
            ids,
            method=canonical_method,
            tokenized=tokenized,
            embeddings=embed_vectors,
            top_k=top_k,
            min_score=min_score,
        )
    if canonical_mode == "query":
        if canonical_method == "embedding_cosine":
            if query_embedding is None or embed_vectors is None:
                raise ValueError(
                    "query mode with embeddings requires query_embedding and embeddings"
                )
            return sim.query_similarity(
                query_id,
                ids,
                method=canonical_method,
                query_embedding=query_embedding,
                embeddings=embed_vectors,
                top_k=top_k,
                min_score=min_score,
            )
        if query_text is None:
            raise ValueError("query mode requires query_text")
        return sim.query_similarity(
            query_id,
            ids,
            method=canonical_method,
            query_tokens=tokenize(query_text, prepared.preprocessing_profile),
            tokenized=tokenized,
            top_k=top_k,
            min_score=min_score,
        )
    if not group_keys:
        raise ValueError("group_centroid requires group_keys")
    return sim.group_centroid_similarity(
        ids,
        group_keys,
        method=canonical_method,
        tokenized=tokenized,
        embeddings=embed_vectors,
        target=centroid_target,
        top_k=top_k,
        min_score=min_score,
    )


def run_duplicate_detection_operator(
    prepared: PreparedCorpusArtifact,
    **kwargs: Any,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.duplicate_detection import duplicate_report

    return duplicate_report(
        [
            {"id": unit_id, "text": text}
            for unit_id, text in zip(prepared.unit_ids, prepared.original_units, strict=True)
        ],
        **kwargs,
    )


def run_clustering_operator(
    prepared: PreparedCorpusArtifact,
    *,
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    use_svd: bool = False,
    svd_components: int = 50,
    random_seed: int = 42,
    top_n_terms: int = 10,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.clustering import run_clustering

    return run_clustering(
        list(prepared.texts_joined),
        list(prepared.unit_ids),
        n_clusters=n_clusters,
        algorithm=algorithm,
        config=prepared.preprocessing_profile,
        use_svd=use_svd,
        svd_components=svd_components,
        random_seed=random_seed,
        top_n_terms=top_n_terms,
    )


def run_dimensionality_reduction_operator(
    prepared: PreparedCorpusArtifact,
    *,
    method: str = "svd",
    n_components: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.clustering import build_tfidf_matrix
    from backend.modules.text_research.infrastructure.dimensionality import reduce_dimensions

    matrix, _ = build_tfidf_matrix(list(prepared.original_units), prepared.preprocessing_profile)
    return reduce_dimensions(
        matrix,
        list(prepared.unit_ids),
        method=method,
        n_components=n_components,
        random_seed=random_seed,
    )


OPERATOR_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "corpus_stats": run_corpus_stats_operator,
    "frequencies": run_frequencies_operator,
    "ngrams": run_ngrams_operator,
    "dfm": run_dfm_operator,
    "kwic": run_kwic_operator,
    "readability": run_readability_operator,
    "dictionary": run_dictionary_operator,
    "keyness": run_keyness_operator,
    "cooccurrence": run_cooccurrence_operator,
    "similarity": run_similarity_operator,
    "duplicate_detection": run_duplicate_detection_operator,
    "clustering": run_clustering_operator,
    "dimensionality_reduction": run_dimensionality_reduction_operator,
}

# Ticket / StageRunner alias — prefer this name at call sites.
OPERATORS = OPERATOR_REGISTRY


def list_registered_operators() -> list[str]:
    return sorted(OPERATOR_REGISTRY)


def build_operator_kwargs(
    analysis_type: str,
    params: dict[str, Any] | None = None,
    *,
    group_keys: list[str] | None = None,
    unit_metadata: list[dict[str, Any]] | None = None,
    dictionary_spec: Any = None,
    random_seed: int = 42,
    feature_extraction_type: str | None = None,
) -> dict[str, Any]:
    """Map normalized analysis parameters onto operator keyword arguments.

    Shared by QuantitativeAnalysisService and StageRunner so both surfaces
    construct the same call into :data:`OPERATORS`.

    Quantitative ops are migrated through domain Config models first so this
    builder does not own business defaults.
    """
    from backend.modules.text_research.domain.quantitative_configs import (
        CONFIG_BY_OPERATION,
        migrate_quantitative_parameters,
    )

    params = dict(params or {})
    if analysis_type in CONFIG_BY_OPERATION:
        params = migrate_quantitative_parameters(analysis_type, params)
    if analysis_type == "corpus_stats":
        kwargs: dict[str, Any] = {}
        if "mattr_window" in params:
            kwargs["mattr_window"] = int(params["mattr_window"])
        if "msttr_window" in params:
            kwargs["msttr_window"] = int(params["msttr_window"])
        return kwargs
    if analysis_type == "frequencies":
        return {
            "top_n": int(params["top_n"]),
            "rate_per": float(params["rate_per"]),
            "group_keys": group_keys,
        }
    if analysis_type == "ngrams":
        return {
            "n": int(params["n"]),
            "top_n": int(params["top_n"]),
            "rate_per": float(params["rate_per"]),
            "skip": int(params["skip"]),
        }
    if analysis_type == "dfm":
        build_kwargs: dict[str, Any] = {
            "weighting": str(params.get("weighting") or feature_extraction_type or "count"),
        }
        for key in ("k1", "b", "smooth_idf", "force_sparse_only", "trim"):
            if key in params and params[key] is not None:
                build_kwargs[key] = params[key]
        return build_kwargs
    if analysis_type == "kwic":
        return {
            "keyword": str(params.get("keyword", "")),
            "window_size": int(params.get("window_size", 5)),
            "case_sensitive": bool(params.get("case_sensitive", False)),
            "query_mode": str(params.get("query_mode", "auto")),
            "language": params.get("query_language") or params.get("language"),
            "token_attribute": params.get("token_attribute"),
            "max_matches": params.get("max_matches"),
            "unit_metadata": unit_metadata,
        }
    if analysis_type == "readability":
        return {}
    if analysis_type == "dictionary":
        if dictionary_spec is None:
            raise ValueError("dictionary requires dictionary_spec")
        return {
            "dictionary_spec": dictionary_spec,
            "case_sensitive": bool(params.get("case_sensitive", False)),
            "rate_per": float(params.get("rate_per", 1000.0)),
            "group_keys": group_keys,
            "dictionary_metadata": params.get("dictionary_metadata"),
            "unit_ids": params.get("unit_ids"),
            "metadata": params.get("metadata"),
        }
    if analysis_type == "keyness":
        return {
            "method": str(params["method"]),
            "top_n": int(params["top_n"]),
            "min_frequency": int(params["min_frequency"]),
            "correction": params["correction"],
            "group_a_label": params.get("group_a_label"),
            "group_b_label": params.get("group_b_label"),
            "group_field": params.get("group_field"),
        }
    if analysis_type == "cooccurrence":
        return {
            "window_size": int(params["window_size"]),
            "top_n": int(params["top_n"]),
            "association_method": str(params["association_method"]),
            "directional": bool(params["directional"]),
            "min_frequency": int(params["min_frequency"]),
            "min_count": int(params["min_count"]),
            "include_network": bool(params["include_network"]),
        }
    if analysis_type == "similarity":
        return {
            "method": str(params["method"]),
            "mode": str(params["mode"]),
            "group_keys": group_keys,
            "top_k": params.get("top_k"),
            "min_score": params.get("min_score"),
            "centroid_target": str(params["centroid_target"]),
            "query_text": params.get("query_text"),
            "query_id": params.get("query_id") or params.get("query_unit_id") or "query",
            "embeddings": params.get("embeddings"),
            "query_embedding": params.get("query_embedding"),
            "exclude_unit_id": params.get("exclude_unit_id"),
        }
    if analysis_type == "clustering":
        return {
            "n_clusters": int(params["n_clusters"]),
            "algorithm": str(params["algorithm"]),
            "use_svd": bool(params["use_svd"]),
            "svd_components": int(params["n_svd_components"]),
            "random_seed": int(params["random_seed"]),
            "top_n_terms": int(params["top_terms"]),
        }
    if analysis_type == "dimensionality_reduction":
        return {
            "method": str(params["method"]),
            "n_components": int(params["n_components"]),
            "random_seed": int(params["random_seed"]),
        }
    if analysis_type == "duplicate_detection":
        return {
            "methods": params.get("methods"),
            "lexical_threshold": float(params["lexical_threshold"]),
            "char_ngram_size": int(params["char_ngram_size"]),
            "minhash_num_perm": int(params["minhash_num_perm"]),
            "minhash_shingle_size": int(params["minhash_shingle_size"]),
            "minhash_threshold": float(params["minhash_threshold"]),
            "max_pairs": params.get("max_pairs"),
        }
    raise KeyError(f"No operator kwargs builder for {analysis_type!r}")


def invoke_operator(
    analysis_type: str,
    prepared: PreparedCorpusArtifact,
    params: dict[str, Any] | None = None,
    *,
    prepared_b: PreparedCorpusArtifact | None = None,
    group_keys: list[str] | None = None,
    unit_metadata: list[dict[str, Any]] | None = None,
    dictionary_spec: Any = None,
    random_seed: int = 42,
    feature_extraction_type: str | None = None,
) -> dict[str, Any]:
    """Run ``OPERATORS[analysis_type](prepared, **normalized_kwargs)``.

    This is the single deterministic analysis entry used by both the FastAPI
    quantitative service and headless StageRunner.
    """
    operator = OPERATORS.get(analysis_type)
    if operator is None:
        raise KeyError(f"No operator registered for {analysis_type!r}")

    kwargs = build_operator_kwargs(
        analysis_type,
        params,
        group_keys=group_keys,
        unit_metadata=unit_metadata,
        dictionary_spec=dictionary_spec,
        random_seed=random_seed,
        feature_extraction_type=feature_extraction_type,
    )
    if analysis_type == "keyness":
        if prepared_b is None:
            raise ValueError("keyness requires prepared_b")
        return operator(prepared, prepared_b, **kwargs)

    result = operator(prepared, **kwargs)
    if analysis_type == "clustering":
        return {key: value for key, value in result.items() if key != "tfidf_matrix"}
    return result
