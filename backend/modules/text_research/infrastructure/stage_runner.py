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


def _run_registered_operator(context: dict[str, Any], _plan: ExecutionPlan) -> None:
    """Dispatch StageRunner analysis stages through canonical OPERATORS."""
    from backend.modules.text_research.application.analysis_operators import invoke_operator
    from backend.modules.text_research.infrastructure.dictionary_matcher import (
        parse_dictionary_payload,
    )

    prepared: PreparedCorpusArtifact = context.get("prepared_a") or context["prepared"]
    spec: AnalysisSpecification = context["spec"]
    analysis_type = spec.analysis.type
    params = dict(spec.analysis.parameters or {})

    group_keys = None
    if analysis_type in {"frequencies", "dictionary", "similarity"}:
        group_keys = _resolve_group_keys(
            prepared,
            group_by=params.get("group_by"),
            documents_by_id=context.get("documents_by_id"),
        )

    dictionary_spec = None
    if analysis_type == "dictionary":
        spec_payload = (
            params.get("hierarchy") or params.get("dictionary_terms") or params.get("terms")
        )
        if isinstance(spec_payload, dict):
            dictionary_spec = parse_dictionary_payload(spec_payload)
        else:
            dictionary_spec = parse_dictionary_payload(
                {"terms": spec_payload or [], "source": "inline"}
            )

    prepared_b = None
    if analysis_type == "keyness":
        prepared_b = context.get("prepared_b")
        if prepared_b is None:
            texts_b = context.get("texts_b")
            if not texts_b:
                raise ValueError("keyness requires context['texts_b'] or context['prepared_b']")
            prepared_b = prepare_texts(
                texts_b,
                prepared.preprocessing_profile,
                unit_ids=context.get("unit_ids_b"),
                force_in_memory=context.get("force_in_memory", False),
            )
            context["prepared_b"] = prepared_b

    feature_type = None
    if analysis_type == "dfm":
        feature_type = getattr(getattr(spec, "feature_extraction", None), "type", None)

    context["results"] = invoke_operator(
        analysis_type,
        prepared,
        params,
        prepared_b=prepared_b,
        group_keys=group_keys,
        dictionary_spec=dictionary_spec,
        random_seed=int(getattr(spec, "random_seed", 42) or 42),
        feature_extraction_type=feature_type,
    )


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
    "frequencies": _run_registered_operator,
    "corpus_stats": _run_registered_operator,
    "ngrams": _run_registered_operator,
    "dfm": _run_registered_operator,
    "kwic": _run_registered_operator,
    "dictionary": _run_registered_operator,
    "keyness": _run_registered_operator,
    "cooccurrence": _run_registered_operator,
    "similarity": _run_registered_operator,
    "clustering": _run_registered_operator,
    "dimensionality_reduction": _run_registered_operator,
    "duplicate_detection": _run_registered_operator,
    "readability": _run_registered_operator,
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
