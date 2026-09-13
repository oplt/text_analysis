"""Safe deterministic active-job dedup and completed-result reuse (LATEST-005).

Completed-result reuse is allowed only when:
1. the operation is in :data:`IDENTITY_COMPLETE_OPS` (exhaustive identity tests),
2. the request/source carries trustworthy
   ``analysis_spec_hash + corpus_checksum + pipeline_checksum + engine_version``,
3. similarity is not raw-vector ``embedding_cosine`` (vectors are not persisted).

Queue-time reuse is disabled: enqueue happens before prepared-corpus hashes exist,
so a computation identity attached there is not scientifically trustworthy.
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

# Operations whose scientific identity is covered by LATEST-003 mutation tests.
# Completed-result reuse stays disabled for anything outside this set.
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
        # Lexical / managed-artifact similarity only — see is_raw_embedding_similarity.
        "similarity",
    }
)

_QUANT_OP_KEY = "_quantitative_operation"


def _analysis_parameters(params: dict[str, Any]) -> dict[str, Any]:
    spec = params.get("analysis_specification")
    analysis = spec.get("analysis") if isinstance(spec, dict) else None
    raw = analysis.get("parameters") if isinstance(analysis, dict) else None
    return raw if isinstance(raw, dict) else {}


def resolve_operation(params: dict[str, Any] | None) -> str | None:
    """Best-effort operation name from persisted or in-flight parameters."""
    payload = params or {}
    quant_op = payload.get(_QUANT_OP_KEY)
    if isinstance(quant_op, str) and quant_op:
        return quant_op
    spec = payload.get("analysis_specification")
    analysis = spec.get("analysis") if isinstance(spec, dict) else None
    analysis_type = analysis.get("type") if isinstance(analysis, dict) else None
    if isinstance(analysis_type, str) and analysis_type:
        return analysis_type
    return None


def has_trustworthy_computation_hashes(params: dict[str, Any] | None) -> bool:
    """True when identity inputs needed for safe reuse are all present."""
    payload = params or {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    return bool(
        (payload.get("analysis_spec_hash") or provenance.get("analysis_spec_hash"))
        and (payload.get("corpus_checksum") or provenance.get("corpus_checksum"))
        and (payload.get("pipeline_checksum") or provenance.get("pipeline_checksum"))
        and (payload.get("engine_version") or provenance.get("engine_version"))
    )


def is_raw_embedding_similarity(params: dict[str, Any] | None) -> bool:
    """Raw embedding vectors are transient and must never back completed reuse."""
    payload = params or {}
    analysis_params = _analysis_parameters(payload)
    method = payload.get("method") or analysis_params.get("method")
    has_artifact = bool(
        payload.get("embedding_artifact_id") or analysis_params.get("embedding_artifact_id")
    )
    if method == "embedding_cosine" and not has_artifact:
        return True
    has_embeddings = bool(payload.get("has_embeddings") or analysis_params.get("has_embeddings"))
    return bool(has_embeddings and not has_artifact)


def completed_result_reuse_allowed(
    operation: str | None, params: dict[str, Any] | None = None
) -> bool:
    """Gate completed-result reuse for an operation + parameter snapshot."""
    if operation not in IDENTITY_COMPLETE_OPS:
        return False
    if operation == "similarity" and is_raw_embedding_similarity(params):
        return False
    return has_trustworthy_computation_hashes(params)


def _is_reusable_source(run: AnalysisRun) -> bool:
    """Require a complete, allowlisted identity before reusing a source run."""
    params = loads(getattr(run, "parameters_json", None), {}) or {}
    operation = resolve_operation(params)
    return completed_result_reuse_allowed(operation, params)


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
        # Stale/partial identities must not authorize reuse without hashes.
        if not has_trustworthy_computation_hashes(params):
            return None
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
    else:
        payload.pop("computation_identity", None)
    return payload


async def find_reusable_run(
    repo: Any,
    *,
    project_id: str,
    corpus_id: str,
    computation_identity_value: str,
    exclude_run_id: str | None = None,
    allow_completed: bool = True,
    request_params: dict[str, Any] | None = None,
) -> AnalysisRun | None:
    """Prefer an in-flight twin, else a completed immutable result.

    When ``request_params`` is provided, completed reuse additionally requires
    :func:`completed_result_reuse_allowed` for the request snapshot.
    """
    if request_params is not None and not has_trustworthy_computation_hashes(request_params):
        # Incomplete request identity: never join/reuse.
        return None
    if request_params is not None:
        operation = resolve_operation(request_params)
        if operation not in IDENTITY_COMPLETE_OPS:
            return None
        if operation == "similarity" and is_raw_embedding_similarity(request_params):
            return None

    active = await repo.find_run_by_computation_identity(
        project_id=project_id,
        corpus_id=corpus_id,
        computation_identity=computation_identity_value,
        statuses=sorted(ACTIVE_STATUSES),
        exclude_run_id=exclude_run_id,
    )
    if active is not None and active.status not in NEVER_REUSE and _is_reusable_source(active):
        return active

    if not allow_completed:
        return None
    if request_params is not None and not completed_result_reuse_allowed(
        resolve_operation(request_params), request_params
    ):
        return None

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

    if source.status in NEVER_REUSE or source.status not in REUSABLE_COMPLETED:
        raise ValueError(f"Refusing to reuse run {source.id} in status {source.status!r}")
    if not _is_reusable_source(source):
        raise ValueError(
            f"Refusing to reuse run {source.id}: identity is incomplete or not allowlisted"
        )

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
