"""Quantitative text-analysis workflows: corpus stats, frequencies, n-grams,
DFM, KWIC, dictionary scoring, keyness, and co-occurrence.

Every method here computes real statistics from persisted `TextUnit` text via
`infrastructure.quantitative`, and persists an `AnalysisRun` recording the
parameters (including the preprocessing profile) and results for
reproducibility. Nothing is fabricated.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
    estimate_workload,
    execute_or_enqueue,
    run_cpu_bound,
    run_prepared_analysis,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.engine_comparison import (
    COMPARABLE_ANALYSIS_TYPES,
    COMPARISON_AWAITING_STAGE,
    COMPARISON_PARENT_RUN_ID_KEY,
    COMPARISON_ROLE_KEY,
    TERMINAL_COMPARISON_CHILD_STATUSES,
    child_failure_detail,
    compare_engine_results,
    extract_runtime,
)
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.domain.analysis_result import (
    build_analysis_identity,
    build_scientific_inputs,
    scientific_inputs_as_dicts,
)
from backend.modules.text_research.domain.analysis_specification import EngineSpec
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.execution_defaults import (
    ENGINE_VERSION,
    computation_identity,
)
from backend.modules.text_research.domain.models import (
    AnalysisRun,
    CorpusDocument,
    TextUnit,
    dumps,
    loads,
)
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.feature_cache import build_cache_key
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts_cached_async,
)
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
)

QUANT_OP_KEY = "_quantitative_operation"


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
    units: list[TextUnit],
    documents: list[CorpusDocument],
    group_by: str | list[str] | None,
) -> list[str] | None:
    if not group_by:
        return None
    field = group_by[0] if isinstance(group_by, list) else group_by
    if not field:
        return None
    doc_lookup = {d.id: d for d in documents}
    return [_resolve_group_value(doc_lookup.get(unit.corpus_document_id), field) for unit in units]


def _tokenized_from_prepared(prepared: PreparedCorpusArtifact) -> list[list[str]]:
    return [list(seq) for seq in prepared.token_sequences]


async def _prepare_with_identity(
    *,
    corpus_id: str,
    analysis_type: str,
    texts: list[str],
    config: PreprocessingConfig,
    units: list[TextUnit],
    unit_type: str,
    filters: dict[str, Any] | None = None,
    cleaning_profile_hash: str | None = None,
    analysis_parameters: dict[str, Any] | None = None,
) -> tuple[PreparedCorpusArtifact, dict[str, Any]]:
    """Reuse the prepared-corpus stage cache for deterministic preprocessing.

    Analysis-specific knobs (top_n, DFM weighting, …) stay out of the prep
    cache key so frequencies → DFM with identical scientific inputs share the
    tokenized artifact. Fitted IDF / classifiers are never stored here.
    """
    prepared = await prepare_texts_cached_async(
        texts,
        config.to_dict(),
        corpus_id=corpus_id,
        unit_type=unit_type,
        unit_ids=[u.id for u in units],
        document_ids=[u.corpus_document_id for u in units],
        filters=filters or {},
        cleaning_profile_hash=cleaning_profile_hash,
        operation_config={},
    )
    spec = build_spec_from_request(
        analysis_type,
        corpus_id,
        analysis_parameters=analysis_parameters or {},
    )
    identity = attach_run_identity(
        {
            "corpus_checksum": prepared.corpus_checksum,
            "pipeline_checksum": prepared.pipeline_checksum,
        },
        spec,
    )
    return prepared, identity


def _filter_kwargs(filters: dict[str, Any]) -> dict[str, Any]:
    """Map analysis filter names onto repository document-select kwargs."""
    mapping = {
        "organization": "organization",
        "organization_type": "organization_type",
        "region": "region",
        "cultural_sphere": "cultural_sphere",
        "language": "language",
        "publication_type": "publication_type",
        "country": "country",
        "publication_year": "publication_year",
        "publication_year_min": "publication_year_min",
        "publication_year_max": "publication_year_max",
        "year_min": "publication_year_min",
        "year_max": "publication_year_max",
    }
    return {
        mapping[key]: value
        for key, value in filters.items()
        if key in mapping and value is not None
    }


def _apply_document_filters(
    documents: list[CorpusDocument], filters: dict[str, Any]
) -> list[CorpusDocument]:
    """Legacy in-memory filter kept for callers that already loaded documents."""
    result = documents
    kwargs = _filter_kwargs(filters)
    if kwargs.get("organization"):
        result = [d for d in result if d.organization == kwargs["organization"]]
    if kwargs.get("organization_type"):
        result = [d for d in result if d.organization_type == kwargs["organization_type"]]
    if kwargs.get("region"):
        result = [d for d in result if d.region == kwargs["region"]]
    if kwargs.get("cultural_sphere"):
        result = [d for d in result if d.cultural_sphere == kwargs["cultural_sphere"]]
    if kwargs.get("language"):
        result = [d for d in result if d.language == kwargs["language"]]
    if kwargs.get("publication_type"):
        result = [d for d in result if d.publication_type == kwargs["publication_type"]]
    if kwargs.get("country"):
        result = [d for d in result if d.country == kwargs["country"]]
    if kwargs.get("publication_year") is not None:
        result = [d for d in result if d.publication_year == kwargs["publication_year"]]
    if kwargs.get("publication_year_min") is not None:
        result = [
            d
            for d in result
            if d.publication_year is not None
            and d.publication_year >= kwargs["publication_year_min"]
        ]
    if kwargs.get("publication_year_max") is not None:
        result = [
            d
            for d in result
            if d.publication_year is not None
            and d.publication_year <= kwargs["publication_year_max"]
        ]
    if filters.get("document_ids"):
        wanted = set(filters["document_ids"])
        result = [d for d in result if d.id in wanted]
    return result


class QuantitativeAnalysisService(ResearchAccessMixin):
    async def _ensure_r_execution_capability(self, *, inline: bool) -> None:
        """Reject R work before creating a run that cannot be executed."""
        from backend.modules.text_research.infrastructure.r_runtime.capabilities import (
            get_live_r_worker_capabilities,
            r_feature_enabled,
        )

        if not r_feature_enabled():
            raise HTTPException(status_code=503, detail="R / quanteda analysis is unavailable")
        if inline:
            if not RAnalysisEngine.runtime_ready():
                raise HTTPException(
                    status_code=503,
                    detail="R / quanteda local runtime is not ready",
                )
            return
        if not await get_live_r_worker_capabilities():
            raise HTTPException(
                status_code=503,
                detail="No live R / quanteda worker is ready",
            )

    async def _resolve_config(
        self, preprocessing_profile_id: str | None, *, user_id: str
    ) -> tuple[PreprocessingConfig, dict[str, Any]]:
        if preprocessing_profile_id is None:
            config = PreprocessingConfig()
            return config, {
                "preprocessing_profile_id": None,
                "preprocessing_config": config.to_dict(),
                "preprocessing_implementation": describe_implementation(config),
            }
        profile = await self.get_preprocessing_profile_or_404(
            preprocessing_profile_id, user_id=user_id
        )
        config = PreprocessingProfileService.resolve_config(profile)
        return config, {
            "preprocessing_profile_id": profile.id,
            "preprocessing_config": config.to_dict(),
            "preprocessing_implementation": describe_implementation(config),
        }

    async def _select(
        self, corpus_id: str, *, user_id: str, unit_type: str, filters: dict[str, Any] | None = None
    ) -> tuple[Any, list[TextUnit], list[CorpusDocument]]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        filter_kwargs = _filter_kwargs(filters or {})
        documents = await self.repo.list_documents(corpus_id, **filter_kwargs)
        if filters and filters.get("document_ids"):
            wanted = set(filters["document_ids"])
            documents = [d for d in documents if d.id in wanted]
        doc_ids = [d.id for d in documents]
        units = await self.repo.list_text_units_for_corpus(
            corpus_id,
            unit_type=unit_type,
            document_ids=doc_ids
            if (filters and filter_kwargs) or (filters and filters.get("document_ids"))
            else None,
        )
        if not units:
            raise HTTPException(
                status_code=422,
                detail="No text units match the requested corpus/unit_type/filters. "
                "Segment the corpus first or relax filters.",
            )
        return corpus, units, documents

    async def _persist_run(
        self,
        corpus,
        run_type: AnalysisRunType,
        *,
        user_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        results: dict[str, Any],
        existing_run: AnalysisRun | None = None,
    ) -> AnalysisRun:
        comparison_parent_id: str | None = None
        if existing_run is not None:
            prior = loads(existing_run.parameters_json, {}) or {}
            raw_parent = prior.get(COMPARISON_PARENT_RUN_ID_KEY)
            if isinstance(raw_parent, str) and raw_parent:
                comparison_parent_id = raw_parent
                parameters = {
                    **parameters,
                    COMPARISON_PARENT_RUN_ID_KEY: comparison_parent_id,
                    COMPARISON_ROLE_KEY: prior.get(COMPARISON_ROLE_KEY),
                }
            from backend.modules.text_research.application.run_lifecycle import (
                ensure_not_cancelled,
            )

            existing_run = await ensure_not_cancelled(self.repo, existing_run)
            updated = await self.repo.update_run(
                existing_run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(results),
                completed_at=_utcnow(),
                error_message=None,
            )
            await self.repo.upsert_run_artifacts(updated.id, list(results.get("artifacts") or []))
            await self.db.commit()
            if comparison_parent_id:
                self._schedule_comparison_finalize(
                    comparison_parent_id, user_id=user_id or existing_run.created_by
                )
            return updated

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=run_type.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(results),
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.repo.upsert_run_artifacts(run.id, list(results.get("artifacts") or []))
        await self.db.commit()
        return run

    def _schedule_comparison_finalize(self, parent_run_id: str, *, user_id: str) -> None:
        """Enqueue idempotent parent finalization after a child terminal transition."""
        from backend.modules.text_research.workers import queue_engine_comparison_finalize

        queue_engine_comparison_finalize(parent_run_id=parent_run_id, user_id=user_id)

    async def _enqueue_quantitative(
        self,
        corpus,
        run_type: AnalysisRunType,
        *,
        user_id: str,
        operation: str,
        parameters: dict[str, Any],
        estimate: Any,
        execution_operation: str = "quantitative",
    ) -> AnalysisRun:
        from backend.modules.text_research.application.execution_service import ExecutionService

        params = {
            **parameters,
            QUANT_OP_KEY: operation,
            "workload_estimate": estimate.to_dict()
            if hasattr(estimate, "to_dict")
            else dict(estimate),
        }
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=run_type.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(params),
                created_by=user_id,
            )
        )
        await self.db.commit()
        await ExecutionService.submit(
            db=self.db, run=run, operation=execution_operation, user_id=user_id
        )
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def _resolve_existing_run(self, existing_run_id: str | None) -> AnalysisRun | None:
        if not existing_run_id:
            return None
        run = await self.repo.get_run(existing_run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {existing_run_id} not found")
        from backend.modules.text_research.application.run_lifecycle import ensure_not_cancelled

        run = await ensure_not_cancelled(self.repo, run)
        if run.status == AnalysisRunStatus.QUEUED.value:
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.RUNNING.value,
                progress_stage="running",
                started_at=_utcnow(),
            )
            await self.db.commit()
        return run

    async def execute_quantitative(self, run_id: str) -> AnalysisRun:
        """Worker entry: re-enter the matching quantitative method inline."""
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {
            AnalysisRunStatus.COMPLETED.value,
            AnalysisRunStatus.CANCELLED.value,
        }:
            return run

        params = loads(run.parameters_json, {}) or {}
        operation = params.get(QUANT_OP_KEY)
        filters = dict(params.get("filters") or {})
        user_id = run.created_by
        corpus_id = run.corpus_id
        if not corpus_id or not operation:
            raise ValueError(f"AnalysisRun {run_id} missing corpus_id or quantitative operation")

        try:
            if operation == "frequencies":
                return await self.frequencies(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    top_n=params.get("top_n", 50),
                    rate_per=params.get("rate_per", 1000),
                    group_by=params.get("group_by"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "r_analysis":
                return await self.r_analysis(
                    corpus_id,
                    user_id=user_id,
                    analysis_type=str(params["analysis_type"]),
                    unit_type=params["unit_type"],
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    analysis_parameters=dict(params.get("analysis_parameters") or {}),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "r_keyness":
                return await self.r_keyness(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    filters_a=dict(params["filters_a"]),
                    filters_b=dict(params["filters_b"]),
                    group_field=params.get("group_field"),
                    method=str(params.get("method", "log_likelihood")),
                    correction=str(params.get("correction", "bh")),
                    min_frequency=int(params.get("min_frequency", 1)),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    top_n=int(params.get("top_n", 50)),
                    force_inline=True,
                    existing_run_id=run_id,
                )
            if operation == "engine_comparison":
                return await self.engine_comparison(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    analysis_type=str(params["analysis_type"]),
                    analysis_parameters=dict(params.get("analysis_parameters") or {}),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "ngrams":
                return await self.ngrams(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    n=params.get("n", 2),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    top_n=params.get("top_n", 50),
                    rate_per=params.get("rate_per", 1000),
                    skip=params.get("skip", 0),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "dfm":
                return await self.dfm(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    weighting=params.get("weighting", "count"),
                    k1=params.get("k1"),
                    b=params.get("b"),
                    smooth_idf=params.get("smooth_idf"),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_sparse_only=params.get("force_sparse_only", False),
                    trim=params.get("trim"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "kwic":
                return await self.kwic(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    keyword=str(params["keyword"]),
                    window_size=int(params.get("window_size", 5)),
                    case_sensitive=bool(params.get("case_sensitive", False)),
                    query_mode=str(params.get("query_mode", "auto")),
                    language=params.get("language"),
                    token_attribute=params.get("token_attribute"),
                    max_matches=params.get("max_matches"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "dictionary":
                return await self.dictionary(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    dictionary_terms=params.get("dictionary_terms"),
                    dictionary_id=params.get("dictionary_id"),
                    hierarchy=params.get("hierarchy"),
                    exclusions=params.get("exclusions"),
                    dictionary_language=params.get("dictionary_language"),
                    case_sensitive=bool(params.get("case_sensitive", False)),
                    rate_per=float(params.get("rate_per", 1000.0)),
                    group_by=params.get("group_by"),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "keyness":
                return await self.keyness(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    filters_a=params.get("filters_a") or {},
                    filters_b=params.get("filters_b") or {},
                    group_field=params.get("group_field"),
                    method=params.get("method", "log_likelihood"),
                    correction=params.get("correction", "bh"),
                    min_frequency=params.get("min_frequency", 1),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    top_n=params.get("top_n", 50),
                    force_inline=True,
                    existing_run_id=run_id,
                )
            if operation == "cooccurrence":
                return await self.cooccurrence(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    window_size=params.get("window_size", 5),
                    top_n=params.get("top_n", 50),
                    association_method=str(params.get("association_method", "pmi")),
                    directional=bool(params.get("directional", False)),
                    min_frequency=int(params.get("min_frequency", 1)),
                    min_count=int(params.get("min_count", 1)),
                    include_network=bool(params.get("include_network", True)),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "similarity":
                return await self.similarity(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    method=params.get("method", "tfidf_cosine"),
                    mode=params.get("mode", "pairwise"),
                    top_k=params.get("top_k", 20),
                    min_score=params.get("min_score"),
                    group_by=params.get("group_by"),
                    centroid_target=params.get("centroid_target", "between_groups"),
                    query_text=params.get("query_text"),
                    query_unit_id=params.get("query_unit_id"),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "clustering":
                return await self.clustering(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    n_clusters=params.get("n_clusters", 5),
                    algorithm=params.get("algorithm", "kmeans"),
                    use_svd=params.get("use_svd", False),
                    n_svd_components=params.get("n_svd_components", 50),
                    top_terms=params.get("top_terms", 10),
                    random_seed=params.get("random_seed", 42),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "dimensionality_reduction":
                return await self.dimensionality_reduction(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    method=params.get("method", "svd"),
                    n_components=params.get("n_components", 2),
                    random_seed=params.get("random_seed", 42),
                    preprocessing_profile_id=params.get("preprocessing_profile_id"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            if operation == "duplicate_detection":
                return await self.duplicate_detection(
                    corpus_id,
                    user_id=user_id,
                    unit_type=params["unit_type"],
                    methods=params.get("methods"),
                    lexical_threshold=params.get("lexical_threshold", 0.9),
                    char_ngram_size=params.get("char_ngram_size", 3),
                    use_minhash=params.get("use_minhash", False),
                    minhash_num_perm=params.get("minhash_num_perm", 64),
                    minhash_shingle_size=params.get("minhash_shingle_size", 5),
                    minhash_threshold=params.get("minhash_threshold", 0.8),
                    max_pairs=params.get("max_pairs"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            raise ValueError(f"Unsupported quantitative operation {operation!r}")
        except Exception as exc:
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                error_message=str(exc),
                completed_at=_utcnow(),
            )
            await self.db.commit()
            parent_id = params.get(COMPARISON_PARENT_RUN_ID_KEY)
            if isinstance(parent_id, str) and parent_id:
                self._schedule_comparison_finalize(parent_id, user_id=user_id)
            raise

    async def _enqueue_comparison_child(
        self,
        corpus,
        *,
        user_id: str,
        parent_run_id: str,
        role: str,
        analysis_type: str,
        unit_type: str,
        analysis_parameters: dict[str, Any],
        preprocessing_profile_id: str | None,
        filters: dict[str, Any],
        estimate: Any,
    ) -> AnalysisRun:
        """Always queue a comparison child; never run it on the parent worker."""
        shared = {
            COMPARISON_PARENT_RUN_ID_KEY: parent_run_id,
            COMPARISON_ROLE_KEY: role,
            "unit_type": unit_type,
            "preprocessing_profile_id": preprocessing_profile_id,
            "filters": filters,
        }
        run_types = {
            "frequencies": AnalysisRunType.FREQUENCY_ANALYSIS,
            "dfm": AnalysisRunType.DFM,
            "kwic": AnalysisRunType.KWIC,
            "dictionary": AnalysisRunType.DICTIONARY_ANALYSIS,
            "keyness": AnalysisRunType.KEYNESS,
            "cooccurrence": AnalysisRunType.COOCCURRENCE,
        }
        if analysis_type not in run_types:
            raise HTTPException(
                status_code=422,
                detail=f"Engine comparison does not support {analysis_type!r}",
            )
        if role == "python":
            if analysis_type == "frequencies":
                parameters = {
                    **shared,
                    "top_n": int(analysis_parameters.get("top_n", 50)),
                    "rate_per": float(analysis_parameters.get("rate_per", 1000)),
                    "group_by": analysis_parameters.get("group_by"),
                }
            elif analysis_type == "dfm":
                parameters = {
                    **shared,
                    "weighting": str(analysis_parameters.get("weighting", "count")),
                    "k1": analysis_parameters.get("k1"),
                    "b": analysis_parameters.get("b"),
                    "smooth_idf": analysis_parameters.get("smooth_idf"),
                    "force_sparse_only": bool(analysis_parameters.get("force_sparse_only", False)),
                    "trim": analysis_parameters.get("trim"),
                }
            elif analysis_type == "kwic":
                parameters = {
                    **shared,
                    "keyword": str(analysis_parameters["keyword"]),
                    "window_size": int(analysis_parameters.get("window_size", 5)),
                    "case_sensitive": bool(analysis_parameters.get("case_sensitive", False)),
                    "query_mode": str(analysis_parameters.get("query_mode", "word")),
                    "language": analysis_parameters.get("language"),
                    "token_attribute": analysis_parameters.get("token_attribute"),
                    "max_matches": analysis_parameters.get("max_matches"),
                }
            elif analysis_type == "dictionary":
                parameters = {
                    **shared,
                    "dictionary_id": analysis_parameters.get("dictionary_id"),
                    "dictionary_terms": analysis_parameters.get("dictionary_terms"),
                    "hierarchy": analysis_parameters.get("hierarchy"),
                    "exclusions": analysis_parameters.get("exclusions"),
                    "dictionary_language": analysis_parameters.get("dictionary_language"),
                    "case_sensitive": bool(analysis_parameters.get("case_sensitive", False)),
                    "rate_per": float(analysis_parameters.get("rate_per", 1000.0)),
                    "group_by": analysis_parameters.get("group_by"),
                }
            elif analysis_type == "keyness":
                parameters = {
                    **shared,
                    "filters_a": dict(analysis_parameters.get("filters_a") or {}),
                    "filters_b": dict(analysis_parameters.get("filters_b") or {}),
                    "group_field": analysis_parameters.get("group_field"),
                    "method": str(analysis_parameters.get("method", "log_likelihood")),
                    "correction": str(analysis_parameters.get("correction", "bh")),
                    "min_frequency": int(analysis_parameters.get("min_frequency", 1)),
                    "top_n": int(analysis_parameters.get("top_n", 50)),
                    "filters": {},
                }
            else:  # cooccurrence
                parameters = {
                    **shared,
                    "window_size": int(analysis_parameters.get("window_size", 5)),
                    "top_n": int(analysis_parameters.get("top_n", 50)),
                    "association_method": str(analysis_parameters.get("association_method", "pmi")),
                    "directional": bool(analysis_parameters.get("directional", False)),
                    "min_frequency": int(analysis_parameters.get("min_frequency", 1)),
                    "min_count": int(analysis_parameters.get("min_count", 1)),
                    "include_network": bool(analysis_parameters.get("include_network", False)),
                }
            return await self._enqueue_quantitative(
                corpus,
                run_types[analysis_type],
                user_id=user_id,
                operation=analysis_type,
                parameters=parameters,
                estimate=estimate,
                execution_operation="quantitative",
            )

        if analysis_type == "keyness":
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.KEYNESS,
                user_id=user_id,
                operation="r_keyness",
                parameters={
                    **shared,
                    "filters_a": dict(analysis_parameters.get("filters_a") or {}),
                    "filters_b": dict(analysis_parameters.get("filters_b") or {}),
                    "group_field": analysis_parameters.get("group_field"),
                    "method": str(analysis_parameters.get("method", "log_likelihood")),
                    "correction": str(analysis_parameters.get("correction", "bh")),
                    "min_frequency": int(analysis_parameters.get("min_frequency", 1)),
                    "top_n": int(analysis_parameters.get("top_n", 50)),
                    "filters": {},
                    "engine": EngineSpec(runtime="r").model_dump(mode="json"),
                },
                estimate=estimate,
                execution_operation="r_quantitative",
            )
        r_params = dict(analysis_parameters)
        if analysis_type == "cooccurrence":
            r_params.setdefault("include_network", False)
        return await self._enqueue_quantitative(
            corpus,
            run_types[analysis_type],
            user_id=user_id,
            operation="r_analysis",
            parameters={
                **shared,
                "analysis_type": analysis_type,
                "analysis_parameters": r_params,
                "engine": EngineSpec(runtime="r").model_dump(mode="json"),
            },
            estimate=estimate,
            execution_operation="r_quantitative",
        )

    async def finalize_engine_comparison(self, parent_run_id: str) -> AnalysisRun | None:
        """Idempotent finalizer: compare children once both are terminal.

        Opens whatever session the caller provides; intended to run in a fresh
        worker session so it never shares a long-lived poll loop with children.
        """
        parent = await self.repo.get_run(parent_run_id)
        if parent is None:
            return None
        if parent.status in TERMINAL_COMPARISON_CHILD_STATUSES:
            return parent

        params = loads(parent.parameters_json, {}) or {}
        python_run_id = params.get("python_run_id")
        r_run_id = params.get("r_run_id")
        if not isinstance(python_run_id, str) or not isinstance(r_run_id, str):
            return parent

        python_run = await self.repo.get_run(python_run_id)
        r_run = await self.repo.get_run(r_run_id)
        if python_run is None or r_run is None:
            return parent
        if (
            python_run.status not in TERMINAL_COMPARISON_CHILD_STATUSES
            or r_run.status not in TERMINAL_COMPARISON_CHILD_STATUSES
        ):
            return parent

        # Re-check parent before writing — duplicate finalizers / worker restart.
        parent = await self.repo.get_run(parent_run_id)
        if parent is None or parent.status in TERMINAL_COMPARISON_CHILD_STATUSES:
            return parent

        analysis_type = str(params.get("analysis_type") or "")
        request_params = {
            "unit_type": params.get("unit_type"),
            "analysis_type": analysis_type,
            "analysis_parameters": dict(params.get("analysis_parameters") or {}),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": dict(params.get("filters") or {}),
            "python_run_id": python_run.id,
            "r_run_id": r_run.id,
            QUANT_OP_KEY: "engine_comparison",
        }
        corpus = await self.repo.get_corpus(parent.corpus_id) if parent.corpus_id else None
        if corpus is None:
            await self.repo.update_run(
                parent,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                error_message="Engine comparison parent is missing corpus",
                completed_at=_utcnow(),
            )
            await self.db.commit()
            return parent

        if (
            python_run.status != AnalysisRunStatus.COMPLETED.value
            or r_run.status != AnalysisRunStatus.COMPLETED.value
        ):
            detail = child_failure_detail(python_run=python_run, r_run=r_run)
            await self.repo.update_run(
                parent,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                parameters_json=dumps(request_params),
                metrics_json=dumps({"python_run_id": python_run.id, "r_run_id": r_run.id}),
                results_json=dumps(
                    {
                        "python_run_id": python_run.id,
                        "r_run_id": r_run.id,
                        "python_status": python_run.status,
                        "r_status": r_run.status,
                        "diagnostic": detail,
                    }
                ),
                error_message=f"Engine comparison child run(s) did not complete: {detail}",
                completed_at=_utcnow(),
            )
            await self.db.commit()
            return parent

        python_results = loads(python_run.results_json, {}) or {}
        r_results = loads(r_run.results_json, {}) or {}
        comparison = compare_engine_results(analysis_type, python_results, r_results)
        return await self._persist_run(
            corpus,
            AnalysisRunType.ENGINE_COMPARISON,
            user_id=parent.created_by,
            parameters=request_params,
            metrics={"python_run_id": python_run.id, "r_run_id": r_run.id},
            results={
                "comparison": comparison,
                "python_run_id": python_run.id,
                "r_run_id": r_run.id,
                "python_runtime": extract_runtime(python_results),
                "r_runtime": extract_runtime(r_results),
            },
            existing_run=parent,
        )

    async def engine_comparison(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        analysis_type: str,
        analysis_parameters: dict[str, Any],
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Create independent Python/R child runs, then exit for event-driven finalize.

        Parent orchestration never polls children. Each child terminal transition
        schedules :meth:`finalize_engine_comparison` on a fresh worker session.
        """
        if analysis_type not in COMPARABLE_ANALYSIS_TYPES:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Engine comparison supports frequencies, DFM, KWIC, "
                    "dictionary, keyness, and co-occurrence"
                ),
            )
        self._validate_engine_comparison_parameters(analysis_type, analysis_parameters)
        initial_submission = not force_inline and not existing_run_id
        if analysis_type == "keyness":
            # Keyness selects two groups from analysis_parameters, not corpus filters.
            filters = {}
            corpus, units_a, _ = await self._select(
                corpus_id,
                user_id=user_id,
                unit_type=unit_type,
                filters=dict(analysis_parameters.get("filters_a") or {}),
            )
            _, units_b, _ = await self._select(
                corpus_id,
                user_id=user_id,
                unit_type=unit_type,
                filters=dict(analysis_parameters.get("filters_b") or {}),
            )
            units = list(units_a) + list(units_b)
        else:
            corpus, units, _documents = await self._select(
                corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
            )
        if initial_submission:
            await self._ensure_r_execution_capability(inline=False)
        analysis_parameters = await self._materialize_comparison_analysis_parameters(
            analysis_type,
            analysis_parameters,
            user_id=user_id,
            project_id=corpus.project_id,
        )
        request_params = {
            "unit_type": unit_type,
            "analysis_type": analysis_type,
            "analysis_parameters": analysis_parameters,
            "preprocessing_profile_id": preprocessing_profile_id,
            "filters": filters,
        }
        estimate = estimate_workload(
            analysis_type=analysis_type,
            n_units=len(units),
            texts=[unit.text for unit in units],
        )
        if initial_submission:
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.ENGINE_COMPARISON,
                user_id=user_id,
                operation="engine_comparison",
                parameters=request_params,
                estimate=estimate,
                execution_operation="engine_comparison",
            )

        parent = await self._resolve_existing_run(existing_run_id)
        if parent is None:
            raise ValueError("engine comparison orchestration requires an existing parent run")
        if parent.status in TERMINAL_COMPARISON_CHILD_STATUSES:
            return parent

        await self._ensure_r_execution_capability(inline=False)
        python_run = await self._enqueue_comparison_child(
            corpus,
            user_id=user_id,
            parent_run_id=parent.id,
            role="python",
            analysis_type=analysis_type,
            unit_type=unit_type,
            analysis_parameters=analysis_parameters,
            preprocessing_profile_id=preprocessing_profile_id,
            filters=filters,
            estimate=estimate,
        )
        r_run = await self._enqueue_comparison_child(
            corpus,
            user_id=user_id,
            parent_run_id=parent.id,
            role="r",
            analysis_type=analysis_type,
            unit_type=unit_type,
            analysis_parameters=analysis_parameters,
            preprocessing_profile_id=preprocessing_profile_id,
            filters=filters,
            estimate=estimate,
        )
        stamped = {
            **request_params,
            "python_run_id": python_run.id,
            "r_run_id": r_run.id,
            QUANT_OP_KEY: "engine_comparison",
        }
        parent = await self.repo.update_run(
            parent,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage=COMPARISON_AWAITING_STAGE,
            parameters_json=dumps(stamped),
            metrics_json=dumps({"python_run_id": python_run.id, "r_run_id": r_run.id}),
            results_json=dumps(
                {
                    "python_run_id": python_run.id,
                    "r_run_id": r_run.id,
                    "status": COMPARISON_AWAITING_STAGE,
                }
            ),
            error_message=None,
        )
        await self.db.commit()
        # Parent worker exits here — children trigger finalize when terminal.
        return parent

    def _validate_engine_comparison_parameters(
        self, analysis_type: str, analysis_parameters: dict[str, Any]
    ) -> None:
        """Reject parameterizations that are not scientifically comparable across engines."""
        if analysis_type == "frequencies" and analysis_parameters.get("group_by"):
            raise HTTPException(
                status_code=422,
                detail="Engine comparison for frequencies does not support group_by (R limitation)",
            )
        if analysis_type == "dfm":
            if analysis_parameters.get("weighting", "count") != "count":
                raise HTTPException(
                    status_code=422,
                    detail="Engine comparison for DFM requires weighting='count'",
                )
            if analysis_parameters.get("trim") or analysis_parameters.get("force_sparse_only"):
                raise HTTPException(
                    status_code=422,
                    detail="Engine comparison for DFM does not support trim/sparse-only options",
                )
        if analysis_type == "kwic":
            keyword = str(analysis_parameters.get("keyword") or "").strip()
            if "keyword" not in analysis_parameters or not keyword:
                raise HTTPException(status_code=422, detail="KWIC comparison requires keyword")
            if analysis_parameters.get("query_mode", "word") != "word":
                raise HTTPException(
                    status_code=422,
                    detail="Engine comparison for KWIC requires query_mode='word'",
                )
        if analysis_type == "dictionary":
            has_terms = bool(
                analysis_parameters.get("dictionary_terms") or analysis_parameters.get("terms")
            )
            has_hierarchy = analysis_parameters.get("hierarchy") is not None
            has_id = bool(analysis_parameters.get("dictionary_id"))
            if not (has_terms or has_hierarchy or has_id):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Dictionary comparison requires dictionary_id, hierarchy, "
                        "or dictionary_terms"
                    ),
                )
            if analysis_parameters.get("group_by"):
                raise HTTPException(
                    status_code=422,
                    detail="Engine comparison for dictionary does not support group_by",
                )
        if analysis_type == "keyness":
            filters_a = analysis_parameters.get("filters_a") or {}
            filters_b = analysis_parameters.get("filters_b") or {}
            if not filters_a or not filters_b:
                raise HTTPException(
                    status_code=422,
                    detail="Keyness comparison requires filters_a and filters_b",
                )
        if analysis_type == "cooccurrence":
            from backend.modules.text_research.infrastructure.collocation import (
                normalize_association_method,
            )

            try:
                normalize_association_method(analysis_parameters.get("association_method", "pmi"))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def _materialize_comparison_analysis_parameters(
        self,
        analysis_type: str,
        analysis_parameters: dict[str, Any],
        *,
        user_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Expand dictionary_id into explicit terms/hierarchy for identical Python/R inputs."""
        params = dict(analysis_parameters)
        if analysis_type != "dictionary":
            return params
        dictionary_id = params.get("dictionary_id")
        if not dictionary_id:
            return params
        if params.get("hierarchy") is not None or params.get("dictionary_terms"):
            return params
        from backend.modules.text_research.application.dictionary_service import DictionaryService

        dictionary = await DictionaryService(self.db).get_dictionary(
            str(dictionary_id), user_id=user_id
        )
        if dictionary.project_id != project_id:
            raise HTTPException(
                status_code=400,
                detail="Dictionary does not belong to this corpus project",
            )
        hierarchy = DictionaryService.get_hierarchy(dictionary)
        if hierarchy is not None:
            params["hierarchy"] = hierarchy
        else:
            params["dictionary_terms"] = DictionaryService.get_spec(dictionary).flattened_terms()
        spec = DictionaryService.get_spec(dictionary)
        if spec.exclusions:
            params["exclusions"] = [item.to_dict() for item in spec.exclusions]
        if spec.language and not params.get("dictionary_language"):
            params["dictionary_language"] = spec.language
        return params

    def _document_lookup(
        self, units: list[TextUnit], documents: list[CorpusDocument]
    ) -> dict[str, CorpusDocument]:
        return {d.id: d for d in documents}

    def _cache_key(
        self,
        *,
        corpus_id: str,
        unit_type: str,
        units: list[TextUnit],
        config: PreprocessingConfig,
        filters: dict[str, Any] | None,
        mode: str,
    ) -> str:
        return build_cache_key(
            corpus_id=corpus_id,
            unit_type=unit_type,
            unit_ids=[u.id for u in units],
            unit_hashes=[u.text_hash for u in units],
            config=config,
            mode=mode,
            filters=filters,
        )

    async def r_analysis(
        self,
        corpus_id: str,
        *,
        user_id: str,
        analysis_type: str,
        unit_type: str,
        preprocessing_profile_id: str | None,
        analysis_parameters: dict[str, Any],
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Persist and execute a supported R analysis through the normal run lifecycle."""
        engine = RAnalysisEngine()
        if not engine.supports(analysis_type):
            raise HTTPException(status_code=422, detail=f"R does not support {analysis_type!r}")
        await self._ensure_r_execution_capability(
            inline=force_inline or bool(existing_run_id),
        )
        if analysis_type == "frequencies" and analysis_parameters.get("group_by"):
            raise HTTPException(status_code=422, detail="R frequencies do not yet support group_by")
        if analysis_type == "dfm" and analysis_parameters.get("weighting", "count") != "count":
            raise HTTPException(
                status_code=422,
                detail="R DFM currently supports count weighting only",
            )
        if analysis_type == "dfm" and (
            analysis_parameters.get("trim")
            or analysis_parameters.get("force_sparse_only")
            or analysis_parameters.get("smooth_idf") is not None
            or analysis_parameters.get("k1") is not None
            or analysis_parameters.get("b") is not None
        ):
            raise HTTPException(
                status_code=422,
                detail="R DFM does not support trim, sparse-only, or TF-IDF/BM25 options yet",
            )
        if analysis_type == "kwic" and analysis_parameters.get("query_mode", "auto") != "word":
            raise HTTPException(
                status_code=422,
                detail="R KWIC requires query_mode='word' (literal token match)",
            )
        if analysis_type == "cooccurrence":
            from backend.modules.text_research.infrastructure.collocation import (
                normalize_association_method,
            )

            try:
                normalize_association_method(analysis_parameters.get("association_method", "pmi"))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        corpus, units, _documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        texts = [unit.text for unit in units]
        estimate = estimate_workload(analysis_type=analysis_type, n_units=len(units), texts=texts)
        request_params = {
            "analysis_type": analysis_type,
            "unit_type": unit_type,
            "preprocessing_profile_id": preprocessing_profile_id,
            "analysis_parameters": analysis_parameters,
            "filters": filters,
            "engine": EngineSpec(runtime="r", implementation="quanteda").model_dump(mode="json"),
        }
        run_types = {
            "frequencies": AnalysisRunType.FREQUENCY_ANALYSIS,
            "dfm": AnalysisRunType.DFM,
            "kwic": AnalysisRunType.KWIC,
            "dictionary": AnalysisRunType.DICTIONARY_ANALYSIS,
            "cooccurrence": AnalysisRunType.COOCCURRENCE,
        }
        if not force_inline and not existing_run_id:
            return await self._enqueue_quantitative(
                corpus,
                run_types[analysis_type],
                user_id=user_id,
                operation="r_analysis",
                parameters=request_params,
                estimate=estimate,
                execution_operation="r_quantitative",
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
        )
        spec = build_spec_from_request(
            analysis_type,
            corpus.id,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters=analysis_parameters,
            engine={"runtime": "r", "implementation": "quanteda"},
        )
        unit_ids = [str(unit.id) for unit in units]
        document_ids = [unit.corpus_document_id for unit in units]
        runner_context = await run_cpu_bound(
            run_prepared_analysis,
            spec,
            texts,
            unit_ids=unit_ids,
            config=config,
            document_ids=document_ids,
            run_id=existing_run.id if existing_run is not None else None,
        )
        canonical = runner_context["analysis_result"]
        prepared = runner_context["prepared"]
        if analysis_type == "cooccurrence" and analysis_parameters.get("include_network", True):
            from backend.modules.text_research.infrastructure.association_network import (
                build_association_network,
            )

            pairs = canonical.results.get("pairs") or []
            canonical.results["network"] = build_association_network(
                pairs,
                directed=bool(canonical.results.get("directional")),
                weight_field=(
                    "count"
                    if canonical.results.get("association_method") == "count"
                    else "association_score"
                ),
            )
        persisted_results = {
            **canonical.results,
            "analysis_result": canonical.model_dump(mode="json"),
            "runtime": canonical.runtime.model_dump(mode="json"),
            "identity": canonical.identity.model_dump(mode="json"),
            "warnings": canonical.warnings,
            "diagnostics": canonical.diagnostics,
            "artifacts": canonical.artifacts,
            "timing": canonical.timing.model_dump(mode="json"),
            "corpus_checksum": prepared.corpus_checksum,
            "pipeline_checksum": prepared.pipeline_checksum,
        }
        if analysis_type == "cooccurrence":
            persisted_results.update(
                {
                    "cooccurrence": canonical.results.get("pairs") or [],
                    "report": canonical.results,
                    "network": canonical.results.get("network"),
                }
            )
        return await self._persist_run(
            corpus,
            run_types[analysis_type],
            user_id=user_id,
            parameters={
                **request_params,
                **attach_run_identity(
                    {},
                    spec,
                    implementation_version=engine.implementation_version,
                ),
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "engine": "r",
                "analysis_type": analysis_type,
            },
            results=persisted_results,
            existing_run=existing_run,
        )

    async def r_keyness(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        filters_a: dict[str, Any],
        filters_b: dict[str, Any],
        group_field: str | None = None,
        method: str = "log_likelihood",
        correction: str = "bh",
        min_frequency: int = 1,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        force_inline: bool = False,
        existing_run_id: str | None = None,
    ) -> AnalysisRun:
        """Execute keyness in R over the same two canonical prepared corpora."""
        engine = RAnalysisEngine()
        await self._ensure_r_execution_capability(
            inline=force_inline or bool(existing_run_id),
        )
        if not filters_a or not filters_b:
            raise HTTPException(status_code=400, detail="filters_a and filters_b are required")
        from backend.modules.text_research.infrastructure.keyness import (
            normalize_correction,
            normalize_keyness_method,
        )

        try:
            normalize_keyness_method(method)
            normalize_correction(correction)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        corpus, units_a, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_a
        )
        _, units_b, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_b
        )
        if not units_a or not units_b:
            raise HTTPException(
                status_code=400,
                detail="Both comparison groups must contain at least one text unit",
            )
        estimate = estimate_workload(
            analysis_type="keyness",
            n_units=len(units_a) + len(units_b),
            texts=[unit.text for unit in units_a] + [unit.text for unit in units_b],
            requested_top_k=top_n,
        )
        request_params = {
            "unit_type": unit_type,
            "preprocessing_profile_id": preprocessing_profile_id,
            "filters_a": filters_a,
            "filters_b": filters_b,
            "group_field": group_field,
            "method": method,
            "correction": correction,
            "min_frequency": min_frequency,
            "top_n": top_n,
            "engine": EngineSpec(runtime="r", implementation="quanteda").model_dump(mode="json"),
        }
        if not force_inline and not existing_run_id:
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.KEYNESS,
                user_id=user_id,
                operation="r_keyness",
                parameters={**request_params, "filters": {}},
                estimate=estimate,
                execution_operation="r_quantitative",
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
        )
        inferred_field = group_field
        shared_fields = set(filters_a) & set(filters_b)
        if inferred_field is None and len(shared_fields) == 1:
            inferred_field = next(iter(shared_fields))
        params = {
            "method": method,
            "correction": correction,
            "min_frequency": min_frequency,
            "top_n": top_n,
            "group_field": inferred_field,
            "group_a_label": ", ".join(
                f"{key}={value}" for key, value in sorted(filters_a.items())
            ),
            "group_b_label": ", ".join(
                f"{key}={value}" for key, value in sorted(filters_b.items())
            ),
        }
        spec = build_spec_from_request(
            "keyness",
            corpus.id,
            unit_type=unit_type,
            filters={"a": filters_a, "b": filters_b},
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters=params,
            engine={"runtime": "r", "implementation": "quanteda"},
        )
        prepared_b = await prepare_texts_cached_async(
            [unit.text for unit in units_b],
            config.to_dict(),
            corpus_id=corpus.id,
            unit_type=unit_type,
            unit_ids=[unit.id for unit in units_b],
            document_ids=[unit.corpus_document_id for unit in units_b],
            filters=filters_b,
            operation_config={},
        )
        runner_context = await run_cpu_bound(
            run_prepared_analysis,
            spec,
            [unit.text for unit in units_a],
            unit_ids=[str(unit.id) for unit in units_a],
            document_ids=[unit.corpus_document_id for unit in units_a],
            config=config,
            prepared_b=prepared_b,
            run_id=existing_run.id if existing_run is not None else None,
        )
        canonical = runner_context["analysis_result"]
        prepared_a = runner_context["prepared"]
        report = canonical.results
        scientific_inputs = scientific_inputs_as_dicts(canonical.identity.inputs)
        return await self._persist_run(
            corpus,
            AnalysisRunType.KEYNESS,
            user_id=user_id,
            parameters={
                **request_params,
                **attach_run_identity(
                    {
                        "corpus_checksum": prepared_a.corpus_checksum,
                        "pipeline_checksum": prepared_a.pipeline_checksum,
                        "scientific_inputs": scientific_inputs,
                        "computation_identity": runner_context.get("computation_identity"),
                    },
                    spec,
                    implementation_version=engine.implementation_version,
                    scientific_inputs=scientific_inputs,
                    parent_artifact_checksums=[
                        prepared_a.corpus_checksum,
                        prepared_a.pipeline_checksum,
                        *(
                            [prepared_b.corpus_checksum, prepared_b.pipeline_checksum]
                            if prepared_b is not None
                            else []
                        ),
                    ],
                ),
                **config_params,
            },
            metrics={
                "unit_count_a": len(units_a),
                "unit_count_b": len(units_b),
                "features_returned": len(report.get("features") or []),
                "method": report.get("method"),
                "correction": report.get("correction"),
                "engine": "r",
            },
            results={
                "keyness": report.get("features") or [],
                "report": report,
                "analysis_result": canonical.model_dump(mode="json"),
                "identity": canonical.identity.model_dump(mode="json"),
                "scientific_inputs": scientific_inputs,
                "corpus_checksum": prepared_a.corpus_checksum,
                "pipeline_checksum": prepared_a.pipeline_checksum,
                "reference_corpus_checksum": prepared_b.corpus_checksum,
                "reference_pipeline_checksum": prepared_b.pipeline_checksum,
                "computation_identity": runner_context.get("computation_identity"),
            },
            existing_run=existing_run,
        )

    # ------------------------------------------------------------------
    async def corpus_stats(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        group_by: list[str] | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        texts = [u.text for u in units]
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        stats = await asyncio.to_thread(
            quantitative.corpus_stats,
            texts,
            config,
            cache_key=cache_key,
            document_ids=[u.corpus_document_id for u in units],
        )

        doc_lookup = self._document_lookup(units, documents)

        def group_counts(field: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for unit in units:
                doc = doc_lookup.get(unit.corpus_document_id)
                if doc is None:
                    key = "unspecified"
                elif hasattr(doc, "get_field_value"):
                    value = doc.get_field_value(field)
                    key = str(value) if value is not None and value != "" else "unspecified"
                else:
                    key = str(getattr(doc, field, None) or "unspecified")
                counts[key] = counts.get(key, 0) + 1
            return counts

        # Generic metadata slicing only — no hardcoded research dimensions.
        fields = [f for f in (group_by or []) if f]
        breakdowns = {field: group_counts(field) for field in fields}
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="frequencies",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.CORPUS_STATS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "filters": filters,
                "group_by": fields,
                "pipeline_workflow": list(quantitative.PIPELINE_STAGES),
                **identity,
                **config_params,
            },
            metrics=stats,
            results={
                "breakdowns": breakdowns,
                "lexical_diversity": stats.get("lexical_diversity"),
                "pipeline": stats.get("pipeline"),
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
        )

    async def frequencies(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        rate_per: float = 1000,
        group_by: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        texts = [u.text for u in units]
        estimate = estimate_workload(
            analysis_type="frequencies",
            n_units=len(units),
            n_documents=len(documents),
            texts=texts,
            requested_top_k=top_n,
        )
        request_params = {
            "unit_type": unit_type,
            "preprocessing_profile_id": preprocessing_profile_id,
            "top_n": top_n,
            "rate_per": rate_per,
            "group_by": group_by,
            "filters": filters,
        }

        async def _enqueue():
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.FREQUENCY_ANALYSIS,
                user_id=user_id,
                operation="frequencies",
                parameters=request_params,
                estimate=estimate,
            )

        async def _inline():
            existing_run = await self._resolve_existing_run(existing_run_id)
            config, config_params = await self._resolve_config(
                preprocessing_profile_id, user_id=user_id
            )
            prepared, identity = await _prepare_with_identity(
                corpus_id=corpus.id,
                analysis_type="frequencies",
                texts=texts,
                config=config,
                units=units,
                unit_type=unit_type,
                filters=filters,
                analysis_parameters={"top_n": top_n, "rate_per": rate_per, "group_by": group_by},
            )
            group_keys = _resolve_group_keys(units, documents, group_by)

            report = await run_cpu_bound(
                quantitative.term_frequency_report,
                _tokenized_from_prepared(prepared),
                top_n=top_n,
                rate_per=rate_per,
                unit_ids=[u.id for u in units],
                document_ids=[u.corpus_document_id for u in units],
                group_keys=group_keys,
            )
            rows = report["frequencies"]
            metadata = report["metadata"]
            return await self._persist_run(
                corpus,
                AnalysisRunType.FREQUENCY_ANALYSIS,
                user_id=user_id,
                parameters={
                    **request_params,
                    **identity,
                    **config_params,
                },
                metrics={
                    "unit_count": metadata["unit_count"],
                    "token_count": metadata["token_count"],
                    "vocabulary_size": metadata["vocabulary_size"],
                    "unique_terms_returned": metadata["terms_returned"],
                    "rate_per": metadata["rate_per"],
                },
                results={
                    "frequencies": rows,
                    "metadata": metadata,
                    "corpus_checksum": prepared.corpus_checksum,
                    "pipeline_checksum": prepared.pipeline_checksum,
                },
                existing_run=existing_run,
            )

        return await execute_or_enqueue(
            estimate=estimate,
            inline=_inline,
            enqueue=_enqueue,
            force_inline=force_inline or bool(existing_run_id),
        )

    async def ngrams(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        n: int = 2,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        rate_per: float = 1000,
        skip: int = 0,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        try:
            quantitative.validate_ngram_order(n, skip=skip)
            quantitative.resolve_rate_per(rate_per)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        texts = [u.text for u in units]

        estimate = estimate_workload(
            analysis_type="ngrams",
            n_units=len(units),
            texts=texts,
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.NGRAM_ANALYSIS,
                user_id=user_id,
                operation="ngrams",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "n": n,
                    "top_n": top_n,
                    "rate_per": rate_per,
                    "skip": skip,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="frequencies",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={"n": n, "top_n": top_n, "rate_per": rate_per, "skip": skip},
        )
        report = await asyncio.to_thread(
            quantitative.ngram_frequency_report,
            _tokenized_from_prepared(prepared),
            n=n,
            top_n=top_n,
            rate_per=rate_per,
            skip=skip,
            unit_ids=[u.id for u in units],
            document_ids=[u.corpus_document_id for u in units],
        )
        rows = report["ngrams"]
        metadata = report["metadata"]
        return await self._persist_run(
            corpus,
            AnalysisRunType.NGRAM_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "n": n,
                "top_n": top_n,
                "rate_per": rate_per,
                "skip": skip,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": metadata["unit_count"],
                "ngram_token_count": metadata["ngram_token_count"],
                "vocabulary_size": metadata["vocabulary_size"],
                "ngrams_returned": metadata["ngrams_returned"],
                "n": metadata["n"],
                "n_label": metadata["n_label"],
            },
            results={
                "ngrams": rows,
                "metadata": metadata,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def dfm(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        weighting: str = "count",
        k1: float | None = None,
        b: float | None = None,
        smooth_idf: bool | None = None,
        preprocessing_profile_id: str | None = None,
        force_sparse_only: bool = False,
        trim: dict[str, Any] | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        try:
            weighting = quantitative.normalize_dfm_weighting(weighting)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        texts = [u.text for u in units]

        estimate = estimate_workload(
            analysis_type="dfm",
            n_units=len(units),
            texts=texts,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DFM,
                user_id=user_id,
                operation="dfm",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "weighting": weighting,
                    "k1": k1,
                    "b": b,
                    "smooth_idf": smooth_idf,
                    "force_sparse_only": force_sparse_only,
                    "trim": trim,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dfm",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={"weighting": weighting, "trim": trim},
        )
        build_kwargs: dict[str, Any] = {
            "weighting": weighting,
            "force_sparse_only": force_sparse_only,
            "unit_ids": [u.id for u in units],
            "preprocessing_config": prepared.preprocessing_profile,
        }
        if k1 is not None:
            build_kwargs["k1"] = k1
        if b is not None:
            build_kwargs["b"] = b
        if smooth_idf is not None:
            build_kwargs["smooth_idf"] = smooth_idf
        if trim:
            build_kwargs["trim"] = {k: v for k, v in trim.items() if v is not None}

        try:
            result = await asyncio.to_thread(
                quantitative.build_dfm,
                _tokenized_from_prepared(prepared),
                **build_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = quantitative.dfm_summary(result)
        # Persist sparse DFM + provenance; omit dense_matrix from DB payload when present
        # to keep storage lean (dense remains available via preview / recompute).
        persisted = {
            "dimensions": result["dimensions"],
            "density": result["density"],
            "nnz": result["nnz"],
            "feature_names": result["feature_names"],
            "unit_ids": result["unit_ids"],
            "preprocessing_config": result.get("preprocessing_config"),
            "mode": result["mode"],
            "weighting": result["weighting"],
            "weighting_scheme": result.get("weighting_scheme"),
            "sublinear_tf": result.get("sublinear_tf", False),
            "sparse": result["sparse"],
            "storage": result["storage"],
            "preview": result.get("preview"),
            "trim": result.get("trim"),
        }
        return await self._persist_run(
            corpus,
            AnalysisRunType.DFM,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "weighting": weighting,
                "k1": k1,
                "b": b,
                "smooth_idf": smooth_idf,
                "force_sparse_only": force_sparse_only,
                "trim": trim,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": summary["unit_count"],
                "feature_count": summary["feature_count"],
                "density": summary["density"],
                "sparsity": summary.get("sparsity"),
                "nnz": summary["nnz"],
                "storage": summary["storage"],
                "mode": summary["mode"],
                "trim": summary.get("trim"),
                "estimated_memory_bytes": summary.get("estimated_memory_bytes"),
                "scientific_warnings": summary.get("scientific_warnings"),
            },
            results={
                "summary": summary,
                "dfm": persisted,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def kwic(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        keyword: str,
        window_size: int = 5,
        case_sensitive: bool = False,
        query_mode: str = "auto",
        language: str | None = None,
        token_attribute: str | None = None,
        max_matches: int | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        estimate = estimate_workload(
            analysis_type="kwic",
            n_units=len(units),
            texts=[u.text for u in units],
        )
        request_params = {
            "unit_type": unit_type,
            "keyword": keyword,
            "window_size": window_size,
            "case_sensitive": case_sensitive,
            "query_mode": query_mode,
            "language": language,
            "token_attribute": token_attribute,
            "max_matches": max_matches,
            "filters": filters,
        }
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.KWIC,
                user_id=user_id,
                operation="kwic",
                parameters=request_params,
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config = PreprocessingConfig()
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="kwic",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={"keyword": keyword},
        )
        doc_lookup = self._document_lookup(units, documents)
        payload = []
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            payload.append(
                {
                    "text": unit.text,
                    "text_unit_id": unit.id,
                    "id": unit.id,
                    "corpus_document_id": unit.corpus_document_id,
                    "unit_type": unit.unit_type,
                    "position": unit.position,
                    "page_number": unit.page_number,
                    "paragraph_number": unit.paragraph_number,
                    "sentence_number": unit.sentence_number,
                    "unit_char_start": unit.char_start,
                    "unit_char_end": unit.char_end,
                    "section_heading": unit.section_heading,
                    "document_title": doc.title if doc else None,
                    "organization": doc.organization if doc else None,
                    "publication_year": doc.publication_year if doc else None,
                    "source_url": doc.source_url if doc else None,
                }
            )
        try:
            matches = await asyncio.to_thread(
                quantitative.kwic_search,
                payload,
                keyword,
                window_size=window_size,
                case_sensitive=case_sensitive,
                query_mode=query_mode,
                language=language,
                token_attribute=token_attribute,
                max_matches=max_matches,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await self._persist_run(
            corpus,
            AnalysisRunType.KWIC,
            user_id=user_id,
            parameters={
                **request_params,
                **identity,
            },
            metrics={
                "unit_count": len(units),
                "match_count": len(matches),
            },
            results={
                "matches": matches,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def dictionary(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        dictionary_terms: list[str] | None = None,
        dictionary_id: str | None = None,
        hierarchy: dict[str, Any] | None = None,
        exclusions: list[Any] | None = None,
        dictionary_language: str | None = None,
        case_sensitive: bool = False,
        rate_per: float = 1000.0,
        group_by: str | None = None,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.dictionary_service import DictionaryService
        from backend.modules.text_research.infrastructure.dictionary_matcher import (
            parse_dictionary_payload,
        )

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        estimate = estimate_workload(
            analysis_type="dictionary",
            n_units=len(units),
            texts=[u.text for u in units],
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DICTIONARY_ANALYSIS,
                user_id=user_id,
                operation="dictionary",
                parameters={
                    "unit_type": unit_type,
                    "dictionary_id": dictionary_id,
                    "dictionary_terms": dictionary_terms,
                    "hierarchy": hierarchy,
                    "exclusions": exclusions,
                    "dictionary_language": dictionary_language,
                    "case_sensitive": case_sensitive,
                    "rate_per": rate_per,
                    "group_by": group_by,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        texts = [u.text for u in units]
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dictionary",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={"group_by": group_by, "rate_per": rate_per},
        )
        doc_lookup = self._document_lookup(units, documents)

        dictionary_meta: dict[str, Any] = {"source": "user"}
        if dictionary_id:
            dictionary = await DictionaryService(self.db).get_dictionary(
                dictionary_id, user_id=user_id
            )
            if dictionary.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary does not belong to this corpus project",
                )
            spec = DictionaryService.get_spec(dictionary)
            if dictionary_language:
                spec.language = dictionary_language
            dictionary_meta.update(
                {
                    "dictionary_id": dictionary.id,
                    "name": dictionary.name,
                    "version": dictionary.version,
                    "description": dictionary.description,
                }
            )
        elif hierarchy is not None:
            spec = parse_dictionary_payload(
                {
                    "hierarchy": hierarchy,
                    "exclusions": exclusions or [],
                    "language": dictionary_language,
                    "source": "user",
                },
                language=dictionary_language,
            )
        else:
            terms = dictionary_terms or []
            if not terms:
                raise HTTPException(
                    status_code=400,
                    detail="dictionary_terms or hierarchy required when dictionary_id is omitted",
                )
            spec = parse_dictionary_payload(
                {
                    "terms": terms,
                    "exclusions": exclusions or [],
                    "language": dictionary_language,
                    "source": "user",
                },
                language=dictionary_language,
            )

        group_keys = _resolve_group_keys(units, documents, group_by)

        unit_metadata = []
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            unit_metadata.append(
                {
                    "text_unit_id": unit.id,
                    "corpus_document_id": unit.corpus_document_id,
                    "page_number": unit.page_number,
                    "section_heading": unit.section_heading,
                    "unit_char_start": unit.char_start,
                    "unit_char_end": unit.char_end,
                    "document_title": doc.title if doc else None,
                    "organization": doc.organization if doc else None,
                    "publication_year": doc.publication_year if doc else None,
                }
            )

        from backend.modules.text_research.infrastructure.dictionary_matcher import match_dictionary

        def _run_dictionary() -> dict[str, Any]:
            result = match_dictionary(
                _tokenized_from_prepared(prepared),
                spec,
                unit_ids=[u.id for u in units],
                document_ids=[u.corpus_document_id for u in units],
                metadata=unit_metadata,
                case_sensitive=case_sensitive,
                rate_per=rate_per,
            )
            if dictionary_meta:
                result["dictionary"] = {**result.get("dictionary", {}), **dictionary_meta}
            if group_keys is not None:
                grouped: dict[str, dict[str, float | int]] = {}
                for group, row in zip(group_keys, result["per_unit"], strict=True):
                    bucket = grouped.setdefault(group, {"hits": 0, "units": 0})
                    bucket["hits"] = int(bucket["hits"]) + int(row["hits"])
                    bucket["units"] = int(bucket["units"]) + 1
                result["by_group"] = grouped
            return result

        try:
            result = await asyncio.to_thread(_run_dictionary)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DICTIONARY_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "dictionary_id": dictionary_id,
                "dictionary_terms": dictionary_terms,
                "hierarchy": hierarchy,
                "exclusions": exclusions,
                "dictionary_language": dictionary_language or spec.language,
                "case_sensitive": case_sensitive,
                "rate_per": rate_per,
                "group_by": group_by,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "total_hits": result["total_hits"],
                "normalized_hits": result["normalized_hits"],
                "hits_per_1000_tokens": result["hits_per_1000_tokens"],
                "document_prevalence": result["document_prevalence"],
                "match_count": len(result.get("matches") or []),
            },
            results={
                **result,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def keyness(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        filters_a: dict[str, Any],
        filters_b: dict[str, Any],
        group_field: str | None = None,
        method: str = "log_likelihood",
        correction: str = "bh",
        min_frequency: int = 1,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        force_inline: bool = False,
        existing_run_id: str | None = None,
    ) -> AnalysisRun:
        if not filters_a or not filters_b:
            raise HTTPException(
                status_code=400,
                detail=(
                    "filters_a and filters_b are required — choose comparison "
                    "groups from corpus metadata"
                ),
            )
        corpus, units_a, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_a
        )
        _, units_b, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_b
        )
        if not units_a or not units_b:
            raise HTTPException(
                status_code=400,
                detail="Both comparison groups must contain at least one text unit",
            )
        estimate = estimate_workload(
            analysis_type="keyness",
            n_units=len(units_a) + len(units_b),
            texts=[u.text for u in units_a] + [u.text for u in units_b],
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.KEYNESS,
                user_id=user_id,
                operation="keyness",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "filters_a": filters_a,
                    "filters_b": filters_b,
                    "group_field": group_field,
                    "method": method,
                    "correction": correction,
                    "min_frequency": min_frequency,
                    "top_n": top_n,
                    "filters": {},
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        prepared_a, _identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="keyness",
            texts=[u.text for u in units_a],
            config=config,
            units=units_a,
            unit_type=unit_type,
            filters=filters_a,
            analysis_parameters={
                "method": method,
                "top_n": top_n,
                "min_frequency": min_frequency,
                "correction": correction,
            },
        )
        prepared_b = await prepare_texts_cached_async(
            [u.text for u in units_b],
            config.to_dict(),
            corpus_id=corpus.id,
            unit_type=unit_type,
            unit_ids=[u.id for u in units_b],
            document_ids=[u.corpus_document_id for u in units_b],
            filters=filters_b,
            operation_config={},
        )

        inferred_field = group_field
        if inferred_field is None:
            shared = set(filters_a) & set(filters_b)
            if len(shared) == 1:
                inferred_field = next(iter(shared))

        label_a = ", ".join(f"{k}={v}" for k, v in sorted(filters_a.items()))
        label_b = ", ".join(f"{k}={v}" for k, v in sorted(filters_b.items()))

        from backend.modules.text_research.infrastructure.keyness import keyness_report

        try:
            report = await asyncio.to_thread(
                keyness_report,
                _tokenized_from_prepared(prepared_a),
                _tokenized_from_prepared(prepared_b),
                top_n=top_n,
                method=method,
                min_frequency=min_frequency,
                correction=correction,
                group_a_label=label_a,
                group_b_label=label_b,
                group_field=inferred_field,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rows = report["features"]
        keyness_spec = build_spec_from_request(
            "keyness",
            corpus.id,
            unit_type=unit_type,
            filters={"a": filters_a, "b": filters_b},
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "method": method,
                "top_n": top_n,
                "min_frequency": min_frequency,
                "correction": correction,
                "group_field": inferred_field,
            },
        )
        scientific_inputs = build_scientific_inputs(
            target_corpus_checksum=prepared_a.corpus_checksum,
            target_pipeline_checksum=prepared_a.pipeline_checksum,
            reference_corpus_checksum=prepared_b.corpus_checksum,
            reference_pipeline_checksum=prepared_b.pipeline_checksum,
        )
        inputs_payload = scientific_inputs_as_dicts(scientific_inputs)
        analysis_identity = build_analysis_identity(
            spec_hash=keyness_spec.spec_hash(),
            engine_name="python",
            engine_version=ENGINE_VERSION,
            inputs=scientific_inputs,
        )
        keyness_computation_identity = computation_identity(
            analysis_identity.spec_hash,
            prepared_a.corpus_checksum,
            engine_version=ENGINE_VERSION,
            engine_name="python",
            pipeline_checksum=prepared_a.pipeline_checksum,
            scientific_inputs=scientific_inputs,
        )
        run_identity = attach_run_identity(
            {
                "corpus_checksum": prepared_a.corpus_checksum,
                "pipeline_checksum": prepared_a.pipeline_checksum,
                "scientific_inputs": inputs_payload,
                "computation_identity": keyness_computation_identity,
            },
            keyness_spec,
            scientific_inputs=inputs_payload,
            parent_artifact_checksums=[
                prepared_a.corpus_checksum,
                prepared_a.pipeline_checksum,
                prepared_b.corpus_checksum,
                prepared_b.pipeline_checksum,
            ],
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.KEYNESS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "filters_a": filters_a,
                "filters_b": filters_b,
                "group_field": inferred_field,
                "method": report["method"],
                "correction": report["correction"],
                "min_frequency": min_frequency,
                "top_n": top_n,
                **run_identity,
                **config_params,
            },
            metrics={
                "unit_count_a": len(units_a),
                "unit_count_b": len(units_b),
                "features_tested": report["features_tested"],
                "features_returned": len(rows),
                "method": report["method"],
                "correction": report["correction"],
            },
            results={
                "keyness": rows,
                "report": report,
                "identity": analysis_identity.model_dump(mode="json"),
                "scientific_inputs": inputs_payload,
                "corpus_checksum": prepared_a.corpus_checksum,
                "pipeline_checksum": prepared_a.pipeline_checksum,
                "reference_corpus_checksum": prepared_b.corpus_checksum,
                "reference_pipeline_checksum": prepared_b.pipeline_checksum,
                "computation_identity": keyness_computation_identity,
            },
            existing_run=existing_run,
        )

    async def cooccurrence(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        window_size: int = 5,
        top_n: int = 50,
        association_method: str = "pmi",
        directional: bool = False,
        min_frequency: int = 1,
        min_count: int = 1,
        include_network: bool = True,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )

        estimate = estimate_workload(
            analysis_type="cooccurrence",
            n_units=len(units),
            texts=[u.text for u in units],
            estimated_pairs=len(units) * max(window_size, 1),
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.COOCCURRENCE,
                user_id=user_id,
                operation="cooccurrence",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "window_size": window_size,
                    "top_n": top_n,
                    "association_method": association_method,
                    "directional": directional,
                    "min_frequency": min_frequency,
                    "min_count": min_count,
                    "include_network": include_network,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="cooccurrence",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={
                "window_size": window_size,
                "top_n": top_n,
                "association_method": association_method,
            },
        )
        from backend.modules.text_research.infrastructure.collocation import collocation_report

        try:
            report = await asyncio.to_thread(
                collocation_report,
                _tokenized_from_prepared(prepared),
                window=window_size,
                top_n=top_n,
                association_method=association_method,
                directional=directional,
                min_frequency=min_frequency,
                min_count=min_count,
                include_network=include_network,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rows = report["pairs"]
        network = report.get("network")
        return await self._persist_run(
            corpus,
            AnalysisRunType.COOCCURRENCE,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "window_size": window_size,
                "top_n": top_n,
                "association_method": report["association_method"],
                "directional": report["directional"],
                "min_frequency": min_frequency,
                "min_count": min_count,
                "include_network": include_network,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "pairs_returned": len(rows),
                "pairs_tested": report["pairs_tested"],
                "association_method": report["association_method"],
                "window": report["window"],
                "direction": report["direction"],
                "node_count": (network or {}).get("node_count"),
                "edge_count": (network or {}).get("edge_count"),
            },
            results={
                "cooccurrence": rows,
                "report": report,
                "network": network,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def similarity(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        method: str = "tfidf_cosine",
        mode: str = "pairwise",
        top_k: int | None = 20,
        min_score: float | None = None,
        group_by: str | None = None,
        centroid_target: str = "between_groups",
        query_text: str | None = None,
        query_unit_id: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
        query_embedding: list[float] | None = None,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Document-to-document / unit-to-unit / query-to-document / group-centroid
        similarity over text units selected from the corpus.

        ``method='embedding_cosine'`` requires an explicit ``embeddings`` mapping
        supplied on *this* call — embeddings are never computed or cached by this
        service, so rerunning an embedding-based similarity run requires passing
        ``embeddings`` again (they are intentionally not persisted in run parameters).
        """
        from backend.modules.text_research.infrastructure import similarity as sim_mod

        try:
            canonical_method = sim_mod.normalize_similarity_method(method)
            canonical_mode = sim_mod.normalize_similarity_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )

        ids = [u.id for u in units]
        texts = [u.text for u in units]

        estimate = estimate_workload(
            analysis_type="similarity",
            n_units=len(units),
            texts=texts,
            pair_mode=canonical_mode,
            requested_top_k=top_k or 0,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.SIMILARITY,
                user_id=user_id,
                operation="similarity",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "method": method,
                    "mode": mode,
                    "top_k": top_k,
                    "min_score": min_score,
                    "group_by": group_by,
                    "centroid_target": centroid_target,
                    "query_text": query_text,
                    "query_unit_id": query_unit_id,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="similarity",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={
                "method": canonical_method,
                "mode": canonical_mode,
                "group_by": group_by,
            },
        )
        tokenized = _tokenized_from_prepared(prepared)

        group_keys: list[str] | None = None
        if canonical_mode == "group_centroid":
            if not group_by:
                raise HTTPException(
                    status_code=422,
                    detail="mode='group_centroid' requires 'group_by' (a document metadata field)",
                )
            group_keys = _resolve_group_keys(units, documents, group_by)

        resolved_query_text = query_text
        query_id = "query"
        if canonical_mode == "query" and canonical_method != "embedding_cosine":
            if query_unit_id:
                query_unit = next((u for u in units if u.id == query_unit_id), None)
                if query_unit is None:
                    raise HTTPException(
                        status_code=422,
                        detail="query_unit_id must reference one of the selected text units",
                    )
                resolved_query_text = query_unit.text
                query_id = query_unit.id
                keep = [i for i, u in enumerate(units) if u.id != query_unit_id]
                ids = [ids[i] for i in keep]
                tokenized = [tokenized[i] for i in keep]
            elif not query_text:
                raise HTTPException(
                    status_code=422,
                    detail="mode='query' requires 'query_text' or 'query_unit_id'",
                )

        from backend.modules.text_research.infrastructure.preprocessing import tokenize

        def _run_similarity() -> dict[str, Any]:
            if canonical_method == "embedding_cosine":
                if not embeddings:
                    raise ValueError(
                        "method='embedding_cosine' requires an explicit 'embeddings' mapping"
                    )
                embed_vectors = [embeddings[item_id] for item_id in ids]
                if canonical_mode == "pairwise":
                    return sim_mod.pairwise_similarity(
                        ids,
                        method=canonical_method,
                        embeddings=embed_vectors,
                        top_k=top_k,
                        min_score=min_score,
                    )
                if canonical_mode == "group_centroid":
                    return sim_mod.group_centroid_similarity(
                        ids,
                        group_keys or [],
                        method=canonical_method,
                        embeddings=embed_vectors,
                        target=centroid_target,
                        top_k=top_k,
                        min_score=min_score,
                    )
                raise ValueError("query mode with embeddings requires query_embedding")
            if canonical_mode == "pairwise":
                return sim_mod.pairwise_similarity(
                    ids,
                    method=canonical_method,
                    tokenized=tokenized,
                    top_k=top_k,
                    min_score=min_score,
                )
            if canonical_mode == "query":
                query_tokens = tokenize(resolved_query_text or "", config.to_dict())
                return sim_mod.query_similarity(
                    query_id,
                    ids,
                    method=canonical_method,
                    query_tokens=query_tokens,
                    tokenized=tokenized,
                    top_k=top_k,
                    min_score=min_score,
                )
            return sim_mod.group_centroid_similarity(
                ids,
                group_keys or [],
                method=canonical_method,
                tokenized=tokenized,
                target=centroid_target,
                top_k=top_k,
                min_score=min_score,
            )

        try:
            report = await asyncio.to_thread(_run_similarity)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result_rows = report.get("pairs") or report.get("items") or []
        return await self._persist_run(
            corpus,
            AnalysisRunType.SIMILARITY,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "method": canonical_method,
                "mode": canonical_mode,
                "top_k": top_k,
                "min_score": min_score,
                "group_by": group_by,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "has_embeddings": bool(embeddings),
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "method": canonical_method,
                "mode": canonical_mode,
                "item_count": report.get("item_count", len(ids)),
                "rows_returned": len(result_rows),
            },
            results={
                **report,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def duplicate_detection(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        methods: list[str] | None = None,
        lexical_threshold: float = 0.85,
        char_ngram_size: int = 5,
        use_minhash: bool = False,
        minhash_num_perm: int = 64,
        minhash_shingle_size: int = 3,
        minhash_threshold: float = 0.8,
        max_pairs: int | None = 1000,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Exact / normalized checksum, lexical near-dup, and optional MinHash
        duplicate detection over text units selected from the corpus.

        This is the same engine ingestion QA uses for its near-duplicate check
        (:mod:`infrastructure.duplicate_detection`), exposed here as an explicit,
        rerunnable analysis over any unit granularity / metadata filter.
        """
        from backend.modules.text_research.infrastructure.duplicate_detection import (
            duplicate_report,
            normalize_duplicate_methods,
        )

        try:
            resolved_methods = normalize_duplicate_methods(methods)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if use_minhash and "minhash" not in resolved_methods:
            resolved_methods.append("minhash")

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        estimate = estimate_workload(
            analysis_type="duplicate_detection",
            n_units=len(units),
            texts=[u.text for u in units],
            pair_mode="duplicate",
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DUPLICATE_DETECTION,
                user_id=user_id,
                operation="duplicate_detection",
                parameters={
                    "unit_type": unit_type,
                    "methods": methods,
                    "lexical_threshold": lexical_threshold,
                    "char_ngram_size": char_ngram_size,
                    "use_minhash": use_minhash,
                    "minhash_num_perm": minhash_num_perm,
                    "minhash_shingle_size": minhash_shingle_size,
                    "minhash_threshold": minhash_threshold,
                    "max_pairs": max_pairs,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        items = [{"id": u.id, "text": u.text} for u in units]

        try:
            report = await asyncio.to_thread(
                duplicate_report,
                items,
                methods=resolved_methods,
                lexical_threshold=lexical_threshold,
                char_ngram_size=char_ngram_size,
                minhash_num_perm=minhash_num_perm,
                minhash_shingle_size=minhash_shingle_size,
                minhash_threshold=minhash_threshold,
                max_pairs=max_pairs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DUPLICATE_DETECTION,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "methods": resolved_methods,
                "lexical_threshold": lexical_threshold,
                "char_ngram_size": char_ngram_size,
                "use_minhash": use_minhash,
                "minhash_num_perm": minhash_num_perm,
                "minhash_shingle_size": minhash_shingle_size,
                "minhash_threshold": minhash_threshold,
                "max_pairs": max_pairs,
                "filters": filters,
            },
            metrics={"unit_count": len(units), **report["summary"]},
            results=report,
            existing_run=existing_run,
        )

    async def clustering(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        n_clusters: int = 5,
        algorithm: str = "kmeans",
        use_svd: bool = False,
        n_svd_components: int = 50,
        top_terms: int = 10,
        random_seed: int = 42,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.clustering import run_clustering

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )

        estimate = estimate_workload(
            analysis_type="clustering",
            n_units=len(units),
            texts=[u.text for u in units],
            n_clusters=n_clusters,
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.CLUSTERING,
                user_id=user_id,
                operation="clustering",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "n_clusters": n_clusters,
                    "algorithm": algorithm,
                    "use_svd": use_svd,
                    "n_svd_components": n_svd_components,
                    "top_terms": top_terms,
                    "random_seed": random_seed,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="clustering",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                "use_svd": use_svd,
            },
        )
        try:
            result = await asyncio.to_thread(
                run_clustering,
                list(prepared.texts_joined),
                [u.id for u in units],
                n_clusters=n_clusters,
                algorithm=algorithm,
                config=prepared.preprocessing_profile,
                use_svd=use_svd,
                svd_components=n_svd_components,
                random_seed=random_seed,
                top_n_terms=top_terms,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Drop non-JSON-serializable matrix handles before persistence.
        persisted = {k: v for k, v in result.items() if k not in {"tfidf_matrix"}}

        return await self._persist_run(
            corpus,
            AnalysisRunType.CLUSTERING,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                "use_svd": use_svd,
                "n_svd_components": n_svd_components,
                "top_terms": top_terms,
                "random_seed": random_seed,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "n_clusters": persisted.get("n_clusters"),
                "silhouette_score": persisted.get("silhouette_score"),
            },
            results={
                **persisted,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def dimensionality_reduction(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        method: str = "svd",
        n_components: int = 2,
        random_seed: int = 42,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.clustering import build_tfidf_matrix
        from backend.modules.text_research.infrastructure.dimensionality import reduce_dimensions

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )

        estimate = estimate_workload(
            analysis_type="dimensionality_reduction",
            n_units=len(units),
            texts=[u.text for u in units],
        )
        if should_enqueue_cpu_job(estimate, force_inline=force_inline or bool(existing_run_id)):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DIMENSIONALITY_REDUCTION,
                user_id=user_id,
                operation="dimensionality_reduction",
                parameters={
                    "unit_type": unit_type,
                    "preprocessing_profile_id": preprocessing_profile_id,
                    "method": method,
                    "n_components": n_components,
                    "random_seed": random_seed,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        cfg = config if isinstance(config, dict) else config.to_dict()

        def _run() -> dict[str, Any]:
            matrix, _ = build_tfidf_matrix([u.text for u in units], cfg)
            return reduce_dimensions(
                matrix,
                [u.id for u in units],
                method=method,
                n_components=n_components,
                random_seed=random_seed,
            )

        try:
            result = await asyncio.to_thread(_run)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DIMENSIONALITY_REDUCTION,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "method": method,
                "n_components": n_components,
                "random_seed": random_seed,
                "filters": filters,
                **config_params,
            },
            metrics={"unit_count": len(units), "n_components": n_components},
            results=result,
            existing_run=existing_run,
        )

    async def readability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.readability import readability_for_units

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config = PreprocessingConfig()
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="readability",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
        )
        result = await asyncio.to_thread(
            readability_for_units,
            list(prepared.original_units),
            [u.id for u in units],
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.READABILITY,
            user_id=user_id,
            parameters={"unit_type": unit_type, "filters": filters, **identity},
            metrics={
                "unit_count": len(units),
                "flesch_reading_ease": result["corpus"]["flesch_reading_ease"],
                "flesch_kincaid_grade": result["corpus"]["flesch_kincaid_grade"],
            },
            results={
                **result,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
        )
