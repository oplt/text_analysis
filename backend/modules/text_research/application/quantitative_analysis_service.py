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
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import (
    AnalysisRun,
    CorpusDocument,
    TextUnit,
    dumps,
    loads,
)
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure import quantitative
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
        if existing_run is not None:
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
            await self.db.commit()
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
        await self.db.commit()
        return run

    async def _enqueue_quantitative(
        self,
        corpus,
        run_type: AnalysisRunType,
        *,
        user_id: str,
        operation: str,
        parameters: dict[str, Any],
        estimate: Any,
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
            db=self.db, run=run, operation="quantitative", user_id=user_id
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
            raise

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
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
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
                "unit_type": unit_type,
                "keyword": keyword,
                "window_size": window_size,
                "case_sensitive": case_sensitive,
                "query_mode": query_mode,
                "language": language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
                "filters": filters,
                **identity,
            },
            metrics={"unit_count": len(units), "match_count": len(matches)},
            results={
                "matches": matches,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
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
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.dictionary_service import DictionaryService
        from backend.modules.text_research.infrastructure.dictionary_matcher import (
            parse_dictionary_payload,
        )

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
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
        prepared_a, identity = await _prepare_with_identity(
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
                **identity,
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
                "corpus_checksum": prepared_a.corpus_checksum,
                "pipeline_checksum": prepared_a.pipeline_checksum,
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
