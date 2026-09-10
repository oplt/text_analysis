"""Thin helpers for building specs, attaching provenance, and workload-aware dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
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
from backend.modules.text_research.infrastructure.workload_estimator import (
    WorkloadEstimate,
    estimate_workload,
    exceeds_async_threshold,
)

logger = logging.getLogger(__name__)


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
            plan.engine_name,
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
            engine_name=plan.engine_name,
            engine_version=plan.engine_version,
        )

    return StageRunner(plan, context, delegate_callback=kwargs.get("delegate_callback")).run()


def should_enqueue_cpu_job(
    estimate: WorkloadEstimate,
    *,
    force_inline: bool = False,
    force_async: bool = False,
) -> bool:
    """Decide whether a CPU-heavy analysis should leave the FastAPI worker.

    * ``force_inline`` — always run in-process (worker re-entry, tests).
    * ``force_async`` — always enqueue (explicit client preference).
    """
    if force_inline:
        return False
    if force_async:
        return True
    return exceeds_async_threshold(estimate)


async def run_cpu_bound[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run moderate CPU work off the event loop via a worker thread."""
    return await asyncio.to_thread(fn, *args, **kwargs)


async def execute_or_enqueue[T](
    *,
    estimate: WorkloadEstimate,
    inline: Callable[[], Awaitable[T]],
    enqueue: Callable[[], Awaitable[T]],
    force_inline: bool = False,
    force_async: bool = False,
) -> T:
    """Run ``inline`` for small jobs; otherwise call ``enqueue`` (Celery path)."""
    if should_enqueue_cpu_job(estimate, force_inline=force_inline, force_async=force_async):
        logger.info(
            "research analysis enqueue analysis_type=%s n_units=%s tokens=%s pairs=%s",
            estimate.analysis_type,
            estimate.n_units,
            estimate.estimated_tokens,
            estimate.estimated_pairs,
        )
        return await enqueue()
    logger.debug(
        "research analysis inline analysis_type=%s n_units=%s tokens=%s pairs=%s",
        estimate.analysis_type,
        estimate.n_units,
        estimate.estimated_tokens,
        estimate.estimated_pairs,
    )
    return await inline()


# Re-export estimator helpers for call sites that import from analysis_executor.
__all__ = [
    "WorkloadEstimate",
    "attach_run_identity",
    "build_spec_from_request",
    "estimate_workload",
    "execute_or_enqueue",
    "plan_and_task",
    "run_cpu_bound",
    "run_prepared_analysis",
    "should_enqueue_cpu_job",
]
