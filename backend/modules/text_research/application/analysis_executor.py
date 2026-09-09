"""Thin helpers for building specs and attaching run identity metadata."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.analysis_task import AnalysisTask, resource_class_for
from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.execution_policy import (
    is_idempotent_hit,
    retry_policy_for,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ExecutionPlan,
    compile_plan,
    computation_identity,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.stage_runner import StageRunner


def build_spec_from_request(
    analysis_type: str,
    corpus_id: str,
    **kwargs: Any,
) -> AnalysisSpecification:
    """Construct a v2 specification from flat request-style arguments."""
    return AnalysisSpecification.from_flat(
        corpus_id=corpus_id,
        analysis_type=analysis_type,
        **kwargs,
    )


def attach_run_identity(
    parameters: dict[str, Any],
    spec: AnalysisSpecification,
    *,
    parent_artifact_checksums: list[str] | None = None,
    cleaning_profile: dict[str, Any] | None = None,
    preprocessing_profile: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    nlp_model: dict[str, Any] | None = None,
    implementation_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Augment run parameters with stable spec identity and full provenance (§23)."""
    from backend.modules.text_research.infrastructure.provenance import attach_provenance

    return attach_provenance(
        parameters,
        spec=spec,
        parent_artifact_checksums=parent_artifact_checksums,
        cleaning_profile=cleaning_profile,
        preprocessing_profile=preprocessing_profile,
        preprocessing_config=preprocessing_config,
        nlp_model=nlp_model,
        implementation_version=implementation_version,
        extra=extra,
    )


def plan_and_task(
    spec: AnalysisSpecification,
    *,
    corpus_snapshot_hash: str | None = None,
    input_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compile an execution plan, AnalysisTask, and optional stage-cache hit."""
    normalized = spec.normalize()
    plan = compile_plan(normalized)
    task = AnalysisTask.create(
        analysis_type=normalized.analysis.type,
        spec_hash=plan.spec_hash,
        input_artifact_ids=input_artifact_ids or [],
    )
    identity = None
    cache_hit = None
    if corpus_snapshot_hash:
        identity = computation_identity(
            plan.spec_hash,
            corpus_snapshot_hash,
            plan.engine_version,
        )
        cache_hit = is_idempotent_hit(
            spec_hash=plan.spec_hash,
            corpus_snapshot_hash=corpus_snapshot_hash,
            engine_version=plan.engine_version,
            stage_cache_get=stage_cache.get_stage,
        )
    return {
        "plan": plan,
        "task": task,
        "computation_identity": identity,
        "stage_cache_hit": cache_hit,
        "resource_class": resource_class_for(normalized.analysis.type),
        "retry_policy": retry_policy_for(task.resource_class),
    }


def run_prepared_analysis(
    spec: AnalysisSpecification | dict[str, Any],
    texts: list[str],
    *,
    unit_ids: list[str] | None = None,
    config: PreprocessingConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Normalize a spec, compile a plan, and execute it via :class:`StageRunner`.

    Intended for CLI entry points and unit tests that run quantitative analyses
    without a database session.
    """
    spec_obj = AnalysisSpecification.model_validate(spec) if isinstance(spec, dict) else spec
    normalized = spec_obj.normalize()
    normalized.validate()
    plan = compile_plan(normalized)

    context: dict[str, Any] = {
        "spec": normalized,
        "texts": list(texts),
        "unit_ids": unit_ids,
        "config": config or PreprocessingConfig(),
    }
    context.update(kwargs)

    if normalized.output.include_manifest is False:
        # compile_plan always appends build_manifest when include_manifest is True;
        # honor explicit False by stripping it post-compile for headless runs.
        plan = ExecutionPlan(
            stages=[stage for stage in plan.stages if stage != "build_manifest"],
            spec_hash=plan.spec_hash,
            engine_version=plan.engine_version,
        )

    return StageRunner(plan, context, delegate_callback=kwargs.get("delegate_callback")).run()
