"""Shared selection, persistence, and enqueue helpers for quantitative analyses."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_quantitative_spec,
)
from backend.modules.text_research.application.analysis_identity import (
    QUANTITATIVE_PARAMETER_SCHEMA_VERSION,
    bind_quantitative_execution_kwargs,
    migrate_quantitative_parameters,
    normalize_analysis_parameters,
    normalize_quantitative_run_parameters,
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
from backend.modules.text_research.domain.quantitative_configs import CONFIG_BY_OPERATION
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


def _with_frozen_preprocessing(
    parameters: dict[str, Any],
    preprocessing_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach frozen preprocessing for exact-reproduce async re-entry."""
    if preprocessing_config is None:
        return parameters
    return {**parameters, "preprocessing_config": preprocessing_config}


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


class QuantitativeSupportMixin:
    """Access, prepare, persist, and worker re-entry for quantitative runs."""

    async def _resolve_config(
        self,
        preprocessing_profile_id: str | None,
        *,
        user_id: str,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> tuple[PreprocessingConfig, dict[str, Any]]:
        """Resolve preprocessing for analysis.

        When ``preprocessing_config`` is provided (exact reproduce), use that
        frozen snapshot and ignore the live profile behind ``preprocessing_profile_id``.
        Replay omits the override so the current profile is loaded.
        """
        if preprocessing_config is not None:
            config = PreprocessingConfig.from_dict(preprocessing_config)
            return config, {
                "preprocessing_profile_id": preprocessing_profile_id,
                "preprocessing_config": config.to_dict(),
                "preprocessing_implementation": describe_implementation(config),
            }
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
            completed_result_reuse_allowed,
            find_reusable_run,
            materialize_reuse_run,
            resolve_operation,
        )

        identity = attach_computation_identity(identity)
        computation_id = identity.get("computation_identity")
        if not computation_id or not corpus.id:
            return None
        operation = resolve_operation(identity)
        if not completed_result_reuse_allowed(operation, identity):
            return None
        found = await find_reusable_run(
            self.repo,
            project_id=corpus.project_id,
            corpus_id=corpus.id,
            computation_identity_value=str(computation_id),
            request_params=identity,
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

        operation = parameters.get(QUANT_OP_KEY) or run_type.value
        if operation in CONFIG_BY_OPERATION:
            parameters = migrate_quantitative_parameters(operation, parameters)
        else:
            parameters.setdefault("parameter_schema_version", QUANTITATIVE_PARAMETER_SCHEMA_VERSION)
        parameters = attach_computation_identity(parameters)
        # Pre-allocate so artifactized results can record producing_run_id even
        # for synchronous creates (LATEST-010).
        producing_run_id = existing_run.id if existing_run is not None else str(uuid4())
        inline_results, artifact_ref = maybe_artifactize_results(
            results,
            producing_run_id=producing_run_id,
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
                # Keep artifact_path for model/vectorizer paths only — result
                # bulk payloads use results_artifact_id in results/metrics.
                artifact_path=existing_run.artifact_path,
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
                id=producing_run_id,
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=run_type.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(inline_results),
                artifact_path=None,
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

        # LATEST-005: do not reuse at queue time. Prepared corpus / spec hashes are
        # not available yet, so any computation_identity here would be incomplete.
        # Post-prepare reuse happens via ``_reuse_deterministic_run`` on inline paths;
        # workers re-enter with ``existing_run_id`` and skip reuse intentionally.
        params = normalize_quantitative_run_parameters(
            operation,
            {
                **parameters,
                QUANT_OP_KEY: operation,
                "workload_estimate": estimate.to_dict()
                if hasattr(estimate, "to_dict")
                else dict(estimate),
            },
        )
        # Strip any stale computation_identity from callers — enqueue must not
        # look like a scientifically complete identity snapshot.
        params.pop("computation_identity", None)

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
        user_id = run.created_by
        corpus_id = run.corpus_id
        if not corpus_id or not operation:
            raise ValueError(f"AnalysisRun {run_id} missing corpus_id or quantitative operation")

        try:
            kwargs, filters = bind_quantitative_execution_kwargs(operation, persisted_params)
            frozen_prep = persisted_params.get("preprocessing_config")
            if isinstance(frozen_prep, dict):
                kwargs["preprocessing_config"] = frozen_prep
            frozen_dictionary = persisted_params.get("frozen_dictionary_spec")
            if isinstance(frozen_dictionary, dict):
                kwargs["frozen_dictionary_spec"] = frozen_dictionary
            # LATEST-006: legacy queued raw embedding_cosine runs cannot be executed —
            # vectors were never persisted on the AnalysisRun.
            if operation == "similarity":
                method_name = str(kwargs.get("method") or "")
                if method_name == "embedding_cosine" and not kwargs.get("embedding_artifact_id"):
                    raise ValueError(
                        "Raw embedding vectors cannot be queued; persist/use a managed "
                        "embedding artifact."
                    )
            method = getattr(self, operation, None)
            if method is None or not callable(method):
                raise ValueError(f"Unsupported quantitative operation {operation!r}")
            return await method(
                corpus_id,
                user_id=user_id,
                force_inline=True,
                existing_run_id=run_id,
                **kwargs,
                **filters,
            )
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
