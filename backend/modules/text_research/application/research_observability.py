"""Structured Text Research observability helpers (TASK-026).

Never log full document or query content — only IDs, counts, timings, and sizes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.observability import prometheus_metrics as metrics

logger = logging.getLogger("backend.modules.text_research.observability")


def _safe_byte_size(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        payload = json.dumps(value, default=str, ensure_ascii=False)
        return len(payload.encode("utf-8"))
    except Exception:
        return None


def log_schedule_decision(
    *,
    analysis_type: str,
    decision: str,
    run_id: str | None = None,
    n_units: int | None = None,
    n_documents: int | None = None,
    estimated_tokens: int | None = None,
    estimated_pairs: int | None = None,
) -> None:
    metrics.research_schedule_decisions_total.labels(
        analysis_type=analysis_type or "unknown",
        decision=decision,
    ).inc()
    logger.info(
        "research_schedule analysis_type=%s decision=%s run_id=%s n_units=%s "
        "n_documents=%s estimated_tokens=%s estimated_pairs=%s",
        analysis_type,
        decision,
        run_id,
        n_units,
        n_documents,
        estimated_tokens,
        estimated_pairs,
    )


def log_cache_outcome(
    *,
    cache: str,
    hit: bool,
    analysis_type: str | None = None,
    run_id: str | None = None,
    tier: str | None = None,
) -> None:
    outcome = "hit" if hit else "miss"
    metrics.research_cache_outcomes_total.labels(
        cache=cache or "unknown",
        outcome=outcome,
    ).inc()
    logger.info(
        "research_cache cache=%s outcome=%s tier=%s analysis_type=%s run_id=%s",
        cache,
        outcome,
        tier,
        analysis_type,
        run_id,
    )


def observe_stage_duration(
    *,
    stage: str,
    analysis_type: str,
    seconds: float,
    run_id: str | None = None,
) -> None:
    metrics.research_stage_duration_seconds.labels(
        stage=stage or "unknown",
        analysis_type=analysis_type or "unknown",
    ).observe(max(0.0, float(seconds)))
    logger.info(
        "research_stage stage=%s analysis_type=%s duration_s=%.4f run_id=%s",
        stage,
        analysis_type,
        seconds,
        run_id,
    )


def log_terminal_transition(
    *,
    run_id: str,
    analysis_type: str | None,
    status: str,
    artifact_bytes: int | None = None,
    result_bytes: int | None = None,
) -> None:
    metrics.research_run_transitions_total.labels(
        analysis_type=analysis_type or "unknown",
        status=status or "unknown",
    ).inc()
    if artifact_bytes is not None:
        metrics.research_artifact_bytes.labels(
            analysis_type=analysis_type or "unknown",
        ).observe(max(0, int(artifact_bytes)))
    if result_bytes is not None:
        metrics.research_result_bytes.labels(
            analysis_type=analysis_type or "unknown",
        ).observe(max(0, int(result_bytes)))
    logger.info(
        "research_terminal run_id=%s analysis_type=%s status=%s "
        "artifact_bytes=%s result_bytes=%s",
        run_id,
        analysis_type,
        status,
        artifact_bytes,
        result_bytes,
    )


def result_payload_bytes(results: Any, metrics_payload: Any = None) -> int | None:
    size = _safe_byte_size(results) or 0
    extra = _safe_byte_size(metrics_payload)
    if extra:
        size += extra
    return size or None
