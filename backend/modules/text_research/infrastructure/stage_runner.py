"""Execute compiled pipeline stages against in-memory or delegated backends."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure import artifact_registry, stage_cache
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ExecutionPlan,
    computation_identity,
)
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig

StageHandler = Callable[[dict[str, Any], ExecutionPlan], None]

DELEGATED_ANALYSES: frozenset[str] = frozenset(
    {"classification", "topic_model", "measurement_validation"}
)


def _resolve_group_value(doc: Any, group_by: str) -> str:
    if doc is None:
        return "unspecified"
    if hasattr(doc, "get_field_value"):
        value = doc.get_field_value(group_by)
    else:
        value = getattr(doc, group_by, None)
    if value is None or value == "":
        return "unspecified"
    return str(value)


def _resolve_group_keys(
    prepared: PreparedCorpusArtifact,
    *,
    group_by: str | list[str] | None,
    documents_by_id: dict[str, Any] | None,
) -> list[str] | None:
    if not group_by:
        return None
    field = group_by[0] if isinstance(group_by, list) else group_by
    if not field:
        return None
    doc_lookup = documents_by_id or {}
    keys: list[str] = []
    for unit_id, document_id in zip(prepared.unit_ids, prepared.document_ids, strict=True):
        doc = doc_lookup.get(document_id) if document_id else None
        if doc is None and unit_id in prepared.metadata_by_unit:
            meta = prepared.metadata_by_unit[unit_id]
            value = meta.get(field)
            keys.append(str(value) if value not in (None, "") else "unspecified")
        else:
            keys.append(_resolve_group_value(doc, field))
    return keys


def _preprocessing_config(context: dict[str, Any]) -> dict[str, Any]:
    config = context.get("config")
    if config is None:
        return PreprocessingConfig().to_dict()
    if isinstance(config, PreprocessingConfig):
        return config.to_dict()
    return dict(config)


def _stage_validate_spec(context: dict[str, Any], plan: ExecutionPlan) -> None:
    spec = context.get("spec")
    if spec is None:
        raise ValueError("context['spec'] is required for validate_spec")
    if isinstance(spec, dict):
        spec = AnalysisSpecification.model_validate(spec)
    normalized = spec.normalize()
    normalized.validate()
    context["spec"] = normalized
    context["analysis_spec_hash"] = plan.spec_hash


def _stage_resolve_corpus(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    texts = context.get("texts")
    if not texts:
        raise ValueError("context['texts'] must be a non-empty list for resolve_corpus")
    originals = list(texts)
    n = len(originals)
    unit_ids = context.get("unit_ids")
    if unit_ids is None:
        unit_ids = [f"unit-{index}" for index in range(n)]
    if len(unit_ids) != n:
        raise ValueError("unit_ids must align 1:1 with texts")
    context["texts"] = originals
    context["unit_ids"] = list(unit_ids)
    if context.get("document_ids") is None:
        context["document_ids"] = [None] * n


def _stage_prepare_corpus(context: dict[str, Any], plan: ExecutionPlan) -> None:
    spec: AnalysisSpecification = context["spec"]
    cfg = _preprocessing_config(context)
    prepared = prepare_texts(
        context["texts"],
        cfg,
        unit_ids=context["unit_ids"],
        document_ids=context.get("document_ids"),
        metadata_by_unit=context.get("metadata_by_unit"),
        language_mode=spec.corpus.language_mode,
        language_override=spec.corpus.filters.get("language"),
        force_in_memory=context.get("force_in_memory", False),
    )
    context["prepared"] = prepared
    context["checksums"] = {
        "corpus_checksum": prepared.corpus_checksum,
        "pipeline_checksum": prepared.pipeline_checksum,
        "analysis_spec_hash": plan.spec_hash,
        "engine_version": plan.engine_version,
    }

    artifact_id = artifact_registry.register(
        "prepared_corpus",
        prepared.corpus_checksum,
        {
            "unit_count": len(prepared.unit_ids),
            "pipeline_checksum": prepared.pipeline_checksum,
            "vocabulary_size": len(prepared.vocabulary),
        },
        prepared.pipeline_checksum,
    )
    context["prepared_artifact_id"] = artifact_id

    if context.get("use_stage_cache", True):
        cache_key = stage_cache.stage_cache_key(
            engine_version=plan.engine_version,
            stage_name="prepare_corpus",
            input_checksum=prepared.corpus_checksum,
            spec_hash=plan.spec_hash,
            params={"pipeline_checksum": prepared.pipeline_checksum},
        )
        if not stage_cache.has_stage(cache_key):
            stage_cache.put_stage(
                cache_key,
                meta={
                    "stage_name": "prepare_corpus",
                    "artifact_id": artifact_id,
                    "corpus_checksum": prepared.corpus_checksum,
                    "pipeline_checksum": prepared.pipeline_checksum,
                    "unit_count": len(prepared.unit_ids),
                },
                payload={
                    "unit_ids": list(prepared.unit_ids),
                    "corpus_checksum": prepared.corpus_checksum,
                    "pipeline_checksum": prepared.pipeline_checksum,
                },
                payload_format="json",
            )
        snapshot_hash = prepared.corpus_checksum
        context["computation_identity"] = computation_identity(
            plan.spec_hash,
            snapshot_hash,
            plan.engine_version,
        )
        if context.get("remember_computation", False):
            stage_cache.remember_computation(
                spec_hash=plan.spec_hash,
                corpus_snapshot_hash=snapshot_hash,
                engine_version=plan.engine_version,
                meta={
                    "prepared_artifact_id": artifact_id,
                    "pipeline_checksum": prepared.pipeline_checksum,
                },
                payload={"unit_count": len(prepared.unit_ids)},
                payload_format="json",
            )


def _run_frequencies(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_frequencies_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    spec: AnalysisSpecification = context["spec"]
    params = spec.analysis.parameters
    group_keys = _resolve_group_keys(
        prepared,
        group_by=params.get("group_by"),
        documents_by_id=context.get("documents_by_id"),
    )
    context["results"] = run_frequencies_operator(
        prepared,
        top_n=int(params.get("top_n", 50)),
        rate_per=float(params.get("rate_per", 1000)),
        group_keys=group_keys,
    )


def _run_corpus_stats(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_corpus_stats_operator,
    )

    context["results"] = run_corpus_stats_operator(context["prepared"])


def _run_ngrams(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_ngrams_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    context["results"] = run_ngrams_operator(
        prepared,
        n=int(params.get("n", 2)),
        top_n=int(params.get("top_n", 50)),
        rate_per=float(params.get("rate_per", 1000)),
        skip=int(params.get("skip", 0)),
    )


def _run_dfm(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_dfm_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    spec: AnalysisSpecification = context["spec"]
    params = spec.analysis.parameters
    fe = spec.feature_extraction
    build_kwargs: dict[str, Any] = {}
    for key in ("k1", "b", "smooth_idf", "force_sparse_only", "trim"):
        if key in params and params[key] is not None:
            build_kwargs[key] = params[key]
    context["results"] = run_dfm_operator(
        prepared,
        weighting=str(params.get("weighting") or fe.type or "count"),
        **build_kwargs,
    )


def _run_kwic(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_kwic_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    context["results"] = run_kwic_operator(
        prepared,
        keyword=str(params.get("keyword", "")),
        window_size=int(params.get("window_size", 5)),
        case_sensitive=bool(params.get("case_sensitive", False)),
        query_mode=str(params.get("query_mode", "auto")),
        language=params.get("query_language") or params.get("language"),
        token_attribute=params.get("token_attribute"),
        max_matches=params.get("max_matches"),
    )


def _run_dictionary(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_dictionary_operator,
    )
    from backend.modules.text_research.infrastructure.dictionary_matcher import (
        parse_dictionary_payload,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    spec_payload = params.get("hierarchy") or params.get("dictionary_terms") or params.get("terms")
    if isinstance(spec_payload, dict):
        dictionary_spec = parse_dictionary_payload(spec_payload)
    else:
        dictionary_spec = parse_dictionary_payload(
            {"terms": spec_payload or [], "source": "inline"}
        )
    group_keys = _resolve_group_keys(
        prepared,
        group_by=params.get("group_by"),
        documents_by_id=context.get("documents_by_id"),
    )
    context["results"] = run_dictionary_operator(
        prepared,
        dictionary_spec=dictionary_spec,
        case_sensitive=bool(params.get("case_sensitive", False)),
        rate_per=float(params.get("rate_per", 1000.0)),
        group_keys=group_keys,
    )


def _run_keyness(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import run_keyness_operator

    prepared_a: PreparedCorpusArtifact = context.get("prepared_a") or context["prepared"]
    prepared_b: PreparedCorpusArtifact | None = context.get("prepared_b")
    if prepared_b is None:
        texts_b = context.get("texts_b")
        if not texts_b:
            raise ValueError("keyness requires context['texts_b'] or context['prepared_b']")
        prepared_b = prepare_texts(
            texts_b,
            prepared_a.preprocessing_profile,
            unit_ids=context.get("unit_ids_b"),
            force_in_memory=context.get("force_in_memory", False),
        )
        context["prepared_b"] = prepared_b

    params = context["spec"].analysis.parameters
    context["results"] = run_keyness_operator(
        prepared_a,
        prepared_b,
        method=str(params.get("method", "log_likelihood")),
        top_n=int(params.get("top_n", 50)),
        min_frequency=int(params.get("min_frequency", 1)),
        correction=params.get("correction", "bh"),
        group_a_label=params.get("group_a_label"),
        group_b_label=params.get("group_b_label"),
        group_field=params.get("group_field"),
    )


def _run_cooccurrence(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_cooccurrence_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    context["results"] = run_cooccurrence_operator(
        prepared,
        window_size=int(params.get("window_size", 5)),
        top_n=int(params.get("top_n", 50)),
        association_method=str(params.get("association_method", "pmi")),
        directional=bool(params.get("directional", False)),
        min_frequency=int(params.get("min_frequency", 1)),
        min_count=int(params.get("min_count", 1)),
        include_network=bool(params.get("include_network", True)),
    )


def _run_similarity(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import run_similarity_operator

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    group_keys = _resolve_group_keys(
        prepared,
        group_by=params.get("group_by"),
        documents_by_id=context.get("documents_by_id"),
    )

    context["results"] = run_similarity_operator(
        prepared,
        method=str(params.get("method", "tfidf_cosine")),
        mode=str(params.get("mode", "pairwise")),
        group_keys=group_keys,
        top_k=params.get("top_k"),
        min_score=params.get("min_score"),
        centroid_target=str(params.get("centroid_target", "between_groups")),
        query_text=params.get("query_text"),
        query_id=params.get("query_id") or "query",
        embeddings=params.get("embeddings"),
        query_embedding=params.get("query_embedding"),
    )


def _run_clustering(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import run_clustering_operator

    prepared: PreparedCorpusArtifact = context["prepared"]
    params = context["spec"].analysis.parameters
    result = run_clustering_operator(
        prepared,
        n_clusters=int(params.get("n_clusters", 5)),
        algorithm=str(params.get("algorithm", "kmeans")),
        use_svd=bool(params.get("use_svd", False)),
        svd_components=int(params.get("n_svd_components", params.get("svd_components", 50))),
        random_seed=int(params.get("random_seed", context["spec"].random_seed)),
        top_n_terms=int(params.get("top_terms", 10)),
    )
    context["results"] = {k: v for k, v in result.items() if k != "tfidf_matrix"}


def _run_dimensionality_reduction(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_dimensionality_reduction_operator,
    )

    params = context["spec"].analysis.parameters
    context["results"] = run_dimensionality_reduction_operator(
        context["prepared"],
        method=str(params.get("method", "svd")),
        n_components=int(params.get("n_components", 2)),
        random_seed=int(params.get("random_seed", context["spec"].random_seed)),
    )


def _run_duplicate_detection(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_duplicate_detection_operator,
    )

    params = context["spec"].analysis.parameters
    context["results"] = run_duplicate_detection_operator(
        context["prepared"],
        methods=params.get("methods"),
        lexical_threshold=float(params.get("lexical_threshold", 0.85)),
        char_ngram_size=int(params.get("char_ngram_size", 5)),
        minhash_num_perm=int(params.get("minhash_num_perm", 64)),
        minhash_shingle_size=int(params.get("minhash_shingle_size", 3)),
        minhash_threshold=float(params.get("minhash_threshold", 0.8)),
        max_pairs=params.get("max_pairs", 1000),
    )


def _run_readability(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.application.analysis_operators import (
        run_readability_operator,
    )

    prepared: PreparedCorpusArtifact = context["prepared"]
    context["results"] = run_readability_operator(prepared)


def _run_statistical_model(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    from backend.modules.text_research.infrastructure.statistical_modeling import (
        fit_statistical_model,
    )

    params = context["spec"].analysis.parameters
    rows = params.get("rows") or context.get("rows")
    if not rows:
        raise ValueError("statistical_model requires analysis.parameters.rows or context['rows']")
    context["results"] = fit_statistical_model(
        rows,
        model=str(params.get("model", "ols")),
        dependent_var=str(params["dependent_var"]),
        independent_vars=list(params["independent_vars"]),
        add_intercept=bool(params.get("add_intercept", True)),
    )


def _run_delegated(context: dict[str, Any], plan: ExecutionPlan) -> None:
    callback = context.get("delegate_callback")
    if callback is not None:
        context["results"] = callback(context, plan)
    context["delegated"] = True


def _stage_persist_run(context: dict[str, Any], plan: ExecutionPlan) -> None:
    checksums = dict(context.get("checksums") or {})
    checksums.setdefault("analysis_spec_hash", plan.spec_hash)
    checksums.setdefault("engine_version", plan.engine_version)
    context["checksums"] = checksums
    context.setdefault("run_record", {}).update(
        {
            "analysis_type": context["spec"].analysis.type,
            "analysis_spec_hash": plan.spec_hash,
            "corpus_checksum": checksums.get("corpus_checksum"),
            "pipeline_checksum": checksums.get("pipeline_checksum"),
            "delegated": bool(context.get("delegated")),
        }
    )


def _stage_build_manifest(context: dict[str, Any], plan: ExecutionPlan) -> None:
    from backend.modules.text_research.infrastructure.provenance import build_run_provenance

    prepared: PreparedCorpusArtifact | None = context.get("prepared")
    checksums = dict(context.get("checksums") or {})
    spec = context.get("spec")
    parent_ids: list[str] = []
    if prepared is not None:
        parent_ids = [
            prepared.corpus_checksum,
            prepared.pipeline_checksum,
        ]
    for key in ("parent_artifact_checksums", "input_artifact_ids"):
        extra_parents = context.get(key)
        if isinstance(extra_parents, list):
            parent_ids.extend(str(item) for item in extra_parents)

    provenance = build_run_provenance(
        spec=spec,
        corpus_checksum=checksums.get("corpus_checksum")
        or (prepared.corpus_checksum if prepared else None),
        pipeline_checksum=checksums.get("pipeline_checksum")
        or (prepared.pipeline_checksum if prepared else None),
        parent_artifact_checksums=parent_ids,
        preprocessing_config=(
            prepared.preprocessing_profile
            if prepared is not None and isinstance(prepared.preprocessing_profile, dict)
            else None
        ),
        random_seed=getattr(spec, "random_seed", None) if spec is not None else None,
        implementation_version=plan.engine_version,
        extra={"stage_timings": dict(context.get("stage_timings") or {})},
    )
    manifest: dict[str, Any] = {
        "engine_version": plan.engine_version,
        "analysis_spec_hash": plan.spec_hash,
        "stages": list(plan.stages),
        "checksums": checksums,
        "stage_timings": dict(context.get("stage_timings") or {}),
        "delegated": bool(context.get("delegated")),
        "provenance": provenance,
    }
    if prepared is not None:
        manifest["prepared"] = {
            "unit_count": len(prepared.unit_ids),
            "corpus_checksum": prepared.corpus_checksum,
            "pipeline_checksum": prepared.pipeline_checksum,
            "vocabulary_size": len(prepared.vocabulary),
            "provenance": prepared.provenance,
        }
    context["manifest"] = manifest
    context.setdefault("run_record", {})["provenance"] = provenance


ANALYSIS_HANDLERS: dict[str, StageHandler] = {
    "frequencies": _run_frequencies,
    "corpus_stats": _run_corpus_stats,
    "ngrams": _run_ngrams,
    "dfm": _run_dfm,
    "kwic": _run_kwic,
    "dictionary": _run_dictionary,
    "keyness": _run_keyness,
    "cooccurrence": _run_cooccurrence,
    "similarity": _run_similarity,
    "clustering": _run_clustering,
    "dimensionality_reduction": _run_dimensionality_reduction,
    "duplicate_detection": _run_duplicate_detection,
    "readability": _run_readability,
    "statistical_model": _run_statistical_model,
}

BASE_STAGE_HANDLERS: dict[str, StageHandler] = {
    "validate_spec": _stage_validate_spec,
    "resolve_corpus": _stage_resolve_corpus,
    "prepare_corpus": _stage_prepare_corpus,
    "persist_run": _stage_persist_run,
    "build_manifest": _stage_build_manifest,
}


class StageRunner:
    """Run an :class:`ExecutionPlan` against a mutable context dict."""

    def __init__(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any],
        *,
        delegate_callback: Callable[[dict[str, Any], ExecutionPlan], Any] | None = None,
    ) -> None:
        self.plan = plan
        self.context = context
        if delegate_callback is not None:
            self.context["delegate_callback"] = delegate_callback

    def run(self) -> dict[str, Any]:
        from backend.modules.text_research.application.research_observability import (
            observe_stage_duration,
        )

        timings: dict[str, float] = {}
        analysis_type = getattr(getattr(self.context.get("spec"), "analysis", None), "type", None)
        run_id = self.context.get("run_id")
        for stage in self.plan.stages:
            started = time.perf_counter()
            if stage in BASE_STAGE_HANDLERS:
                BASE_STAGE_HANDLERS[stage](self.context, self.plan)
            elif stage in DELEGATED_ANALYSES:
                _run_delegated(self.context, self.plan)
            elif stage in ANALYSIS_HANDLERS:
                ANALYSIS_HANDLERS[stage](self.context, self.plan)
            else:
                raise ValueError(f"Unknown pipeline stage {stage!r}")
            elapsed = time.perf_counter() - started
            timings[stage] = elapsed
            observe_stage_duration(
                stage=stage,
                analysis_type=str(analysis_type or "unknown"),
                seconds=elapsed,
                run_id=str(run_id) if run_id else None,
            )
        self.context["stage_timings"] = timings
        return self.context
