"""Quantitative text-analysis workflows: corpus stats, frequencies, n-grams,
DFM, KWIC, dictionary scoring, keyness, and co-occurrence.

Every method here computes real statistics from persisted `TextUnit` text via
`infrastructure.quantitative`, and persists an `AnalysisRun` recording the
parameters (including the preprocessing profile) and results for
reproducibility. Nothing is fabricated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_quantitative_spec,
    estimate_workload,
    execute_or_enqueue,
    run_cpu_bound,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.analysis_identity import (
    QUANTITATIVE_PARAMETER_SCHEMA_VERSION,
    normalize_analysis_parameters,
    normalize_quantitative_run_parameters,
)
from backend.modules.text_research.application.analysis_operators import (
    run_clustering_operator,
    run_cooccurrence_operator,
    run_corpus_stats_operator,
    run_dfm_operator,
    run_dictionary_operator,
    run_dimensionality_reduction_operator,
    run_duplicate_detection_operator,
    run_frequencies_operator,
    run_keyness_operator,
    run_kwic_operator,
    run_ngrams_operator,
    run_similarity_operator,
)
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.domain.analysis_specification import normalize_corpus_filters
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


def _content_checksum(value: Any) -> str:
    """Fingerprint an inline scientific input without persisting its full content."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    cleaning_profile_id: str | None = None,
    preprocessing_profile_id: str | None = None,
    analysis_parameters: dict[str, Any] | None = None,
    random_seed: int = 42,
) -> tuple[PreparedCorpusArtifact, dict[str, Any]]:
    """Reuse the prepared-corpus stage cache for deterministic preprocessing.

    Analysis-specific knobs (top_n, DFM weighting, …) stay out of the prep
    cache key so frequencies → DFM with identical scientific inputs share the
    tokenized artifact. Fitted IDF / classifiers are never stored here.

    Scientific identity (``analysis_spec_hash``) includes unit type, normalized
    filters (incl. language when present), resolved preprocessing fingerprint,
    analysis parameters, and seed.
    """
    prep_config = config.to_dict()
    prepared = await prepare_texts_cached_async(
        texts,
        prep_config,
        corpus_id=corpus_id,
        unit_type=unit_type,
        unit_ids=[u.id for u in units],
        document_ids=[u.corpus_document_id for u in units],
        filters=filters or {},
        cleaning_profile_hash=cleaning_profile_hash,
        operation_config={},
    )
    spec = build_quantitative_spec(
        analysis_type,
        corpus_id,
        unit_type=unit_type,
        filters=filters,
        preprocessing_profile_id=preprocessing_profile_id,
        cleaning_profile_id=cleaning_profile_id,
        preprocessing_config=prep_config,
        analysis_parameters=normalize_analysis_parameters(analysis_type, analysis_parameters or {}),
        random_seed=random_seed,
    )
    cleaning_snapshot = None
    if cleaning_profile_id or cleaning_profile_hash:
        cleaning_snapshot = {
            "id": cleaning_profile_id,
            "config_hash": cleaning_profile_hash,
        }
    identity = attach_run_identity(
        {
            "corpus_checksum": prepared.corpus_checksum,
            "pipeline_checksum": prepared.pipeline_checksum,
        },
        spec,
        preprocessing_profile={
            "id": preprocessing_profile_id,
            "config": prep_config,
        },
        preprocessing_config=prep_config,
        cleaning_profile=cleaning_snapshot,
    )
    from backend.modules.text_research.application.run_dedup import attach_computation_identity

    return prepared, attach_computation_identity(identity)


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

    async def _reuse_deterministic_run(
        self,
        corpus,
        *,
        user_id: str,
        identity: dict[str, Any],
        existing_run_id: str | None = None,
    ) -> AnalysisRun | None:
        """Return an in-flight or completed twin when identity is trustworthy."""
        if existing_run_id:
            return None
        from backend.modules.text_research.application.run_dedup import (
            attach_computation_identity,
            find_reusable_run,
            materialize_reuse_run,
        )

        identity = attach_computation_identity(identity)
        computation_id = identity.get("computation_identity")
        if not computation_id or not corpus.id:
            return None
        found = await find_reusable_run(
            self.repo,
            project_id=corpus.project_id,
            corpus_id=corpus.id,
            computation_identity_value=str(computation_id),
        )
        if found is None:
            return None
        reused = await materialize_reuse_run(
            self.repo, source=found, user_id=user_id, parameters=identity
        )
        await self.db.commit()
        return reused

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
        from backend.modules.text_research.application.result_artifacts import (
            maybe_artifactize_results,
        )
        from backend.modules.text_research.application.run_dedup import attach_computation_identity

        parameters.setdefault("parameter_schema_version", QUANTITATIVE_PARAMETER_SCHEMA_VERSION)
        parameters = attach_computation_identity(parameters)
        inline_results, artifact_ref = maybe_artifactize_results(
            results,
            producing_run_id=existing_run.id if existing_run is not None else None,
        )
        if artifact_ref:
            metrics = {
                **metrics,
                "results_artifactized": True,
                "results_artifact_id": artifact_ref,
            }

        if existing_run is not None:
            from backend.modules.text_research.application.run_lifecycle import (
                complete_if_active,
                ensure_not_cancelled,
            )

            existing_run = await ensure_not_cancelled(self.repo, existing_run)
            updated = await complete_if_active(
                self.repo,
                existing_run,
                progress_stage="completed",
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(inline_results),
                artifact_path=artifact_ref or existing_run.artifact_path,
                completed_at=_utcnow(),
                error_message=None,
            )
            await self.db.commit()
            if updated is None:
                refreshed = await self.repo.get_run(existing_run.id)
                assert refreshed is not None
                return refreshed
            return updated

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=run_type.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(inline_results),
                artifact_path=artifact_ref,
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
        from backend.modules.text_research.application.run_dedup import (
            attach_computation_identity,
            find_reusable_run,
            materialize_reuse_run,
        )

        params = attach_computation_identity(
            normalize_quantitative_run_parameters(
                operation,
                {
                    **parameters,
                    QUANT_OP_KEY: operation,
                    "workload_estimate": estimate.to_dict()
                    if hasattr(estimate, "to_dict")
                    else dict(estimate),
                },
            )
        )
        computation_id = params.get("computation_identity")
        if computation_id and corpus.id:
            found = await find_reusable_run(
                self.repo,
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                computation_identity_value=str(computation_id),
            )
            if found is not None:
                reused = await materialize_reuse_run(
                    self.repo, source=found, user_id=user_id, parameters=params
                )
                await self.db.commit()
                return reused

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
        from backend.modules.text_research.application.run_lifecycle import (
            TERMINAL_RUN_STATUSES,
        )

        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in TERMINAL_RUN_STATUSES:
            return run

        persisted_params = loads(run.parameters_json, {}) or {}
        operation = persisted_params.get(QUANT_OP_KEY)
        params = normalize_quantitative_run_parameters(operation or "", persisted_params)
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
                    association_method=params.get("association_method", "pmi"),
                    directional=params.get("directional", False),
                    min_frequency=params.get("min_frequency", 1),
                    min_count=params.get("min_count", 1),
                    include_network=params.get("include_network", True),
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
                    embedding_artifact_id=params.get("embedding_artifact_id"),
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
                    lexical_threshold=params.get("lexical_threshold", 0.85),
                    char_ngram_size=params.get("char_ngram_size", 5),
                    use_minhash=params.get("use_minhash", False),
                    minhash_num_perm=params.get("minhash_num_perm", 64),
                    minhash_shingle_size=params.get("minhash_shingle_size", 3),
                    minhash_threshold=params.get("minhash_threshold", 0.8),
                    max_pairs=params.get("max_pairs"),
                    force_inline=True,
                    existing_run_id=run_id,
                    **filters,
                )
            raise ValueError(f"Unsupported quantitative operation {operation!r}")
        except Exception as exc:
            from backend.modules.text_research.application.run_lifecycle import (
                RunCancelledError,
                fail_if_active,
            )

            if isinstance(exc, RunCancelledError):
                await self.db.commit()
                refreshed = await self.repo.get_run(run_id)
                assert refreshed is not None
                return refreshed
            await fail_if_active(
                self.repo,
                run,
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
            analysis_type="corpus_stats",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={"group_by": fields},
        )
        stats = await asyncio.to_thread(run_corpus_stats_operator, prepared)
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
        run_async: bool = False,
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
                preprocessing_profile_id=preprocessing_profile_id,
                analysis_parameters={"top_n": top_n, "rate_per": rate_per, "group_by": group_by},
            )
            reused = await self._reuse_deterministic_run(
                corpus,
                user_id=user_id,
                identity={**request_params, **identity, **config_params},
                existing_run_id=existing_run_id,
            )
            if reused is not None:
                return reused
            group_keys = _resolve_group_keys(units, documents, group_by)

            report = await run_cpu_bound(
                run_frequencies_operator,
                prepared,
                top_n=top_n,
                rate_per=rate_per,
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
            force_async=run_async,
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
        run_async: bool = False,
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
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
            analysis_type="ngrams",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={"n": n, "top_n": top_n, "rate_per": rate_per, "skip": skip},
        )
        report = await asyncio.to_thread(
            run_ngrams_operator,
            prepared,
            n=n,
            top_n=top_n,
            rate_per=rate_per,
            skip=skip,
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
        run_async: bool = False,
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
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "weighting": weighting,
                "trim": trim,
                "k1": k1,
                "b": b,
                "smooth_idf": smooth_idf,
                "force_sparse_only": force_sparse_only,
            },
        )
        build_kwargs: dict[str, Any] = {
            "weighting": weighting,
            "force_sparse_only": force_sparse_only,
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
            operator_result = await asyncio.to_thread(run_dfm_operator, prepared, **build_kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = operator_result["dfm"]
        summary = operator_result["summary"]
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
        query_language: str | None = None,
        token_attribute: str | None = None,
        max_matches: int | None = None,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="kwic",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "keyword": keyword,
                "window_size": window_size,
                "case_sensitive": case_sensitive,
                "query_mode": query_mode,
                "query_language": query_language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
            },
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
            operator_result = await asyncio.to_thread(
                run_kwic_operator,
                prepared,
                keyword=keyword,
                window_size=window_size,
                case_sensitive=case_sensitive,
                query_mode=query_mode,
                language=query_language,
                token_attribute=token_attribute,
                max_matches=max_matches,
                unit_metadata=payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        matches = operator_result["matches"]
        from backend.modules.text_research.application.analysis_policy import (
            validate_analysis_policy,
        )

        prep_raw = getattr(prepared, "preprocessing_profile", None)
        prep_dict = prep_raw if isinstance(prep_raw, dict) else None
        policy_warnings = validate_analysis_policy(
            analysis_type="kwic",
            case_sensitive=case_sensitive,
            preprocessing=prep_dict,
            language=query_language,
        )
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
                "query_language": query_language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={"unit_count": len(units), "match_count": len(matches)},
            results={
                "matches": matches,
                "scientific_warnings": policy_warnings,
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
        run_async: bool = False,  # accepted for API parity; dictionary stays inline today
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
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={"group_by": group_by, "rate_per": rate_per},
        )
        doc_lookup = self._document_lookup(units, documents)

        dictionary_meta: dict[str, Any] = {"source": "user"}
        # Precedence (TASK-016 / FE contract): when custom ``dictionary_terms`` are
        # supplied together with ``dictionary_id``, the custom terms override the
        # stored dictionary leaf expressions while retaining dictionary identity
        # metadata. Explicit ``hierarchy`` always wins over flat terms.
        if dictionary_id:
            dictionary = await DictionaryService(self.db).get_dictionary(
                dictionary_id, user_id=user_id
            )
            if dictionary.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary does not belong to this corpus project",
                )
            base_spec = DictionaryService.get_spec(dictionary)
            resolved_language = dictionary_language or base_spec.language
            if hierarchy is not None:
                spec = parse_dictionary_payload(
                    {
                        "hierarchy": hierarchy,
                        "exclusions": exclusions if exclusions is not None else [],
                        "language": resolved_language,
                        "source": "user",
                    },
                    language=resolved_language,
                )
                dictionary_meta["terms_source"] = "request_hierarchy_override"
            elif dictionary_terms:
                spec = parse_dictionary_payload(
                    {
                        "terms": dictionary_terms,
                        "exclusions": exclusions
                        if exclusions is not None
                        else [e.to_dict() for e in base_spec.exclusions],
                        "language": resolved_language,
                        "source": "user",
                    },
                    language=resolved_language,
                )
                dictionary_meta["terms_source"] = "request_terms_override"
            else:
                spec = base_spec
                if dictionary_language:
                    spec.language = dictionary_language
                dictionary_meta["terms_source"] = "dictionary_id"
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

        # Rebuild the analysis identity after resolving stored dictionary content
        # and request overrides. The prep artifact is cache-backed, so this does
        # not repeat tokenization.
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dictionary",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "dictionary_id": dictionary_id,
                "dictionary_version": dictionary_meta.get("version"),
                "dictionary_content_checksum": _content_checksum(spec.to_dict()),
                "exclusions": [entry.to_dict() for entry in spec.exclusions],
                "language": spec.language,
                "case_sensitive": case_sensitive,
                "rate_per": rate_per,
                "group_by": group_by,
                "terms": spec.to_dict(),
            },
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

        try:
            result = await asyncio.to_thread(
                run_dictionary_operator,
                prepared,
                dictionary_spec=spec,
                unit_ids=[u.id for u in units],
                metadata=unit_metadata,
                case_sensitive=case_sensitive,
                rate_per=rate_per,
                group_keys=group_keys,
                dictionary_metadata=dictionary_meta,
            )
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
        run_async: bool = False,
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
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "method": method,
                "top_n": top_n,
                "min_frequency": min_frequency,
                "correction": correction,
                "group_field": group_field,
                "filters_b": normalize_corpus_filters(filters_b),
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

        try:
            report = await asyncio.to_thread(
                run_keyness_operator,
                prepared_a,
                prepared_b,
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
        run_async: bool = False,
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
            pair_mode="cooccurrence",
            window_size=window_size,
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "window_size": window_size,
                "top_n": top_n,
                "association_method": association_method,
                "directional": directional,
                "min_frequency": min_frequency,
                "min_count": min_count,
                "include_network": include_network,
            },
        )
        try:
            report = await asyncio.to_thread(
                run_cooccurrence_operator,
                prepared,
                window_size=window_size,
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
        embedding_artifact_id: str | None = None,
        preprocessing_profile_id: str | None = None,
        force_inline: bool = False,
        run_async: bool = False,
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

        n_groups = 0
        if canonical_mode == "group_centroid" and group_by:
            # Cheap cardinality hint for workload (full keys resolved later).
            n_groups = len({_resolve_group_value(d, group_by) for d in documents})
        estimate = estimate_workload(
            analysis_type="similarity",
            n_units=len(units),
            texts=texts,
            pair_mode=canonical_mode,
            requested_top_k=top_k or 0,
            n_groups=n_groups,
        )
        raw_embeddings = canonical_method == "embedding_cosine" and bool(embeddings)
        would_enqueue = should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        )
        if raw_embeddings and would_enqueue:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Raw embedding_cosine vectors cannot be queued because vectors are "
                    "not persisted. Submit a small inline request or use managed embeddings."
                ),
            )
        if canonical_method == "embedding_cosine" and not (embeddings or embedding_artifact_id):
            raise HTTPException(
                status_code=422,
                detail=(
                    "embedding_cosine requires explicit embeddings or a managed embedding artifact"
                ),
            )
        if would_enqueue:
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
                    "embedding_artifact_id": embedding_artifact_id,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        artifact_metadata: dict[str, Any] | None = None
        if embedding_artifact_id:
            from backend.modules.text_research.infrastructure.embeddings import (
                load_managed_embedding_artifact,
            )

            try:
                embeddings, artifact_metadata = load_managed_embedding_artifact(
                    embedding_artifact_id,
                    project_id=corpus.project_id,
                    corpus_id=corpus.id,
                    unit_ids=ids,
                    expected_dim=len(query_embedding) if query_embedding else None,
                )
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (KeyError, ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "method": canonical_method,
                "mode": canonical_mode,
                "group_by": group_by,
                "top_k": top_k,
                "min_score": min_score,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "query_embedding_checksum": _content_checksum(query_embedding)
                if query_embedding
                else None,
                "embedding_checksum": _content_checksum(embeddings) if embeddings else None,
                "embedding_artifact_id": embedding_artifact_id,
                "embedding_artifact_checksum": (
                    artifact_metadata.get("content_checksum") if artifact_metadata else None
                ),
                "has_embeddings": bool(embeddings),
            },
        )
        reused = await self._reuse_deterministic_run(
            corpus,
            user_id=user_id,
            identity={
                "unit_type": unit_type,
                "method": canonical_method,
                "mode": canonical_mode,
                "top_k": top_k,
                "filters": filters,
                **identity,
                **config_params,
            },
            existing_run_id=existing_run_id,
        )
        if reused is not None:
            return reused
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
            elif not query_text:
                raise HTTPException(
                    status_code=422,
                    detail="mode='query' requires 'query_text' or 'query_unit_id'",
                )

        try:
            report = await asyncio.to_thread(
                run_similarity_operator,
                prepared,
                method=canonical_method,
                mode=canonical_mode,
                group_keys=group_keys,
                top_k=top_k,
                min_score=min_score,
                centroid_target=centroid_target,
                query_text=resolved_query_text,
                query_id=query_id,
                embeddings=embeddings,
                query_embedding=query_embedding,
                exclude_unit_id=(
                    query_unit_id if canonical_mode == "query" and query_unit_id else None
                ),
            )
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
                "embedding_artifact_id": embedding_artifact_id,
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
        run_async: bool = False,
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
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="duplicate_detection",
            texts=[u.text for u in units],
            config=PreprocessingConfig(),
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={
                "methods": resolved_methods,
                "lexical_threshold": lexical_threshold,
                "char_ngram_size": char_ngram_size,
                "use_minhash": use_minhash,
                "minhash_num_perm": minhash_num_perm,
                "minhash_shingle_size": minhash_shingle_size,
                "minhash_threshold": minhash_threshold,
                "max_pairs": max_pairs,
            },
        )
        try:
            report = await asyncio.to_thread(
                run_duplicate_detection_operator,
                prepared,
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
                **identity,
            },
            metrics={"unit_count": len(units), **report["summary"]},
            results={
                **report,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
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
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        if n_clusters >= max(len(units), 1):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"n_clusters ({n_clusters}) must be less than the number of "
                    f"selected observations ({len(units)})"
                ),
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
        if should_enqueue_cpu_job(
            estimate, force_inline=force_inline or bool(existing_run_id), force_async=run_async
        ):
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
            preprocessing_profile_id=preprocessing_profile_id,
            random_seed=random_seed,
            analysis_parameters={
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                "use_svd": use_svd,
                "n_svd_components": n_svd_components,
                "top_terms": top_terms,
                "random_seed": random_seed,
            },
        )
        try:
            result = await asyncio.to_thread(
                run_clustering_operator,
                prepared,
                n_clusters=n_clusters,
                algorithm=algorithm,
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
        run_async: bool = False,
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
            analysis_type="dimensionality_reduction",
            n_units=len(units),
            texts=[u.text for u in units],
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
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
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dimensionality_reduction",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            random_seed=random_seed,
            analysis_parameters={
                "method": method,
                "n_components": n_components,
                "random_seed": random_seed,
            },
        )
        try:
            result = await asyncio.to_thread(
                run_dimensionality_reduction_operator,
                prepared,
                method=method,
                n_components=n_components,
                random_seed=random_seed,
            )
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
                **identity,
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
        from backend.modules.text_research.application.analysis_operators import (
            run_readability_operator,
        )
        from backend.modules.text_research.application.analysis_policy import (
            validate_analysis_policy,
        )

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config = PreprocessingConfig()
        language = filters.get("language") or getattr(config, "language", None)
        policy_warnings = validate_analysis_policy(
            analysis_type="readability",
            language=language if isinstance(language, str) else None,
            preprocessing=config.to_dict(),
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="readability",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=None,
        )
        result = await asyncio.to_thread(run_readability_operator, prepared)
        if policy_warnings:
            result = {
                **result,
                "scientific_warnings": [
                    *list(result.get("scientific_warnings") or []),
                    *policy_warnings,
                ],
            }
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
