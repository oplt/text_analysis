"""Safe deterministic active-job dedup and completed-result reuse (TASK-015).

Only reuses runs whose ``computation_identity`` is derived from trustworthy
``analysis_spec_hash + corpus_checksum + pipeline_checksum + engine_version``.
Failed and cancelled runs are never reused.
"""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.execution_defaults import (
    ENGINE_VERSION,
    computation_identity,
)
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads

ACTIVE_STATUSES = frozenset(
    {
        AnalysisRunStatus.QUEUED.value,
        AnalysisRunStatus.RUNNING.value,
        "pending",
    }
)
REUSABLE_COMPLETED = frozenset({AnalysisRunStatus.COMPLETED.value})
NEVER_REUSE = frozenset(
    {
        AnalysisRunStatus.FAILED.value,
        AnalysisRunStatus.CANCELLED.value,
    }
)
IDENTITY_COMPLETE_OPS = frozenset(
    {
        "corpus_stats",
        "frequencies",
        "ngrams",
        "dfm",
        "kwic",
        "dictionary",
        "keyness",
        "cooccurrence",
        "clustering",
        "dimensionality_reduction",
        "duplicate_detection",
        # Only lexical similarity is reusable; raw vectors remain transient.
        "similarity",
    }
)


def _is_reusable_source(run: AnalysisRun) -> bool:
    """Require a complete, allowlisted identity before reusing a source run."""
    params = loads(getattr(run, "parameters_json", None), {}) or {}
    spec = params.get("analysis_specification")
    analysis = spec.get("analysis") if isinstance(spec, dict) else None
    operation = analysis.get("type") if isinstance(analysis, dict) else None
    if operation not in IDENTITY_COMPLETE_OPS:
        return False
    if operation == "similarity" and (
        params.get("has_embeddings")
        or (isinstance(analysis, dict) and analysis.get("parameters", {}).get("has_embeddings"))
        or params.get("method") == "embedding_cosine"
    ):
        return False
    provenance = params.get("provenance") if isinstance(params.get("provenance"), dict) else {}
    return bool(
        (params.get("analysis_spec_hash") or provenance.get("analysis_spec_hash"))
        and (params.get("corpus_checksum") or provenance.get("corpus_checksum"))
        and (params.get("pipeline_checksum") or provenance.get("pipeline_checksum"))
        and (params.get("engine_version") or provenance.get("engine_version"))
    )


def build_computation_identity(
    *,
    analysis_spec_hash: str | None,
    corpus_checksum: str | None,
    pipeline_checksum: str | None = None,
    engine_version: str | None = None,
) -> str | None:
    """Return a deterministic identity or ``None`` when inputs are incomplete."""
    if not analysis_spec_hash or not corpus_checksum:
        return None
    # Prefer corpus checksum as the snapshot hash; fold pipeline into engine salt
    # when present so prep-pipeline drift cannot collide.
    snapshot = corpus_checksum
    if pipeline_checksum:
        snapshot = f"{corpus_checksum}:{pipeline_checksum}"
    return computation_identity(
        analysis_spec_hash,
        snapshot,
        engine_version or ENGINE_VERSION,
    )


def identity_from_parameters(parameters: dict[str, Any] | None) -> str | None:
    params = parameters or {}
    existing = params.get("computation_identity")
    if isinstance(existing, str) and existing:
        return existing
    provenance = params.get("provenance") if isinstance(params.get("provenance"), dict) else {}
    return build_computation_identity(
        analysis_spec_hash=params.get("analysis_spec_hash") or provenance.get("analysis_spec_hash"),
        corpus_checksum=params.get("corpus_checksum") or provenance.get("corpus_checksum"),
        pipeline_checksum=params.get("pipeline_checksum") or provenance.get("pipeline_checksum"),
        engine_version=params.get("engine_version") or provenance.get("engine_version"),
    )


def attach_computation_identity(parameters: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``computation_identity`` is present when hashes are trustworthy."""
    payload = dict(parameters)
    identity = identity_from_parameters(payload)
    if identity:
        payload["computation_identity"] = identity
    return payload


async def find_reusable_run(
    repo: Any,
    *,
    project_id: str,
    corpus_id: str,
    computation_identity_value: str,
    exclude_run_id: str | None = None,
) -> AnalysisRun | None:
    """Prefer an in-flight twin, else a completed immutable result."""
    active = await repo.find_run_by_computation_identity(
        project_id=project_id,
        corpus_id=corpus_id,
        computation_identity=computation_identity_value,
        statuses=sorted(ACTIVE_STATUSES),
        exclude_run_id=exclude_run_id,
    )
    if active is not None and active.status not in NEVER_REUSE and _is_reusable_source(active):
        return active
    completed = await repo.find_run_by_computation_identity(
        project_id=project_id,
        corpus_id=corpus_id,
        computation_identity=computation_identity_value,
        statuses=sorted(REUSABLE_COMPLETED),
        exclude_run_id=exclude_run_id,
    )
    if (
        completed is not None
        and completed.status not in NEVER_REUSE
        and _is_reusable_source(completed)
    ):
        return completed
    return None


async def materialize_reuse_run(
    repo: Any,
    *,
    source: AnalysisRun,
    user_id: str,
    parameters: dict[str, Any] | None = None,
) -> AnalysisRun:
    """Create an audit run that reuses immutable results from ``source``.

    Active (queued/running) sources are returned as-is so callers join the
    in-flight job. Completed sources get a new COMPLETED pointer run.
    """
    if source.status in ACTIVE_STATUSES:
        return source

    params = dict(parameters or loads(source.parameters_json, {}) or {})
    params = attach_computation_identity(params)
    params["reused_from_run_id"] = source.id
    params["result_reuse"] = True
    run = AnalysisRun(
        project_id=source.project_id,
        corpus_id=source.corpus_id,
        run_type=source.run_type,
        status=AnalysisRunStatus.COMPLETED.value,
        progress_stage="reused",
        parameters_json=dumps(params),
        metrics_json=source.metrics_json,
        results_json=source.results_json,
        artifact_path=source.artifact_path,
        random_seed=source.random_seed,
        evidence_revision_hash=source.evidence_revision_hash,
        created_by=user_id,
        started_at=source.started_at,
        completed_at=source.completed_at,
    )
    created = await repo.create_run(run)
    return created
