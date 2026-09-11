"""Celery worker logging and observability hooks."""

from __future__ import annotations

import contextlib
import logging
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from celery.signals import (
    before_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    worker_process_init,
    worker_ready,
    worker_shutdown,
)

from backend.core.config import settings
from backend.core.log_context import reset_correlation_id, set_correlation_id
from backend.core.logging import setup_logging

logger = logging.getLogger("backend.worker")

_task_started_at: dict[str, float] = {}
_task_correlation_tokens: dict[str, Any] = {}
_r_worker_heartbeat_handles: dict[str, Any] = {}

CORRELATION_HEADER = "correlation_id"
ENQUEUED_AT_HEADER = "enqueued_at"


def _task_label(task: Any) -> str:
    name = getattr(task, "name", None) or "unknown"
    return str(name)[-80:]


def _safe_worker_metrics():
    try:
        from backend.observability import prometheus_metrics as m

        return m
    except Exception:
        return None


def _worker_queue_names(sender: Any) -> list[str]:
    """Read queues from the live Celery consumer, rather than configuration intent."""
    consumer = getattr(sender, "task_consumer", sender)
    queues = getattr(consumer, "queues", ())
    values = queues.values() if isinstance(queues, dict) else queues
    return sorted({str(getattr(queue, "name", queue)) for queue in values})


def _worker_id(sender: Any) -> str | None:
    hostname = getattr(sender, "hostname", None)
    return str(hostname) if hostname else None


def _start_r_worker_heartbeat(sender: Any) -> None:
    from backend.modules.text_research.infrastructure.r_runtime.capabilities import (
        R_WORKER_HEARTBEAT_INTERVAL_SECONDS,
        r_feature_enabled,
        refresh_r_worker_heartbeat_if_local_runtime_ready,
        worker_consumes_r_queue,
        worker_id_for_runtime,
    )

    queues = _worker_queue_names(sender)
    if not r_feature_enabled() or not worker_consumes_r_queue(queues):
        return
    worker_id = worker_id_for_runtime(_worker_id(sender))
    ready = refresh_r_worker_heartbeat_if_local_runtime_ready(
        worker_id=worker_id,
        queues=queues,
    )
    logger.info("R worker capability heartbeat published worker=%s ready=%s", worker_id, ready)

    timer = getattr(sender, "timer", None)
    if timer is None or worker_id in _r_worker_heartbeat_handles:
        return
    _r_worker_heartbeat_handles[worker_id] = timer.call_repeatedly(
        R_WORKER_HEARTBEAT_INTERVAL_SECONDS,
        refresh_r_worker_heartbeat_if_local_runtime_ready,
        kwargs={"worker_id": worker_id, "queues": queues},
        priority=10,
    )


@worker_process_init.connect
def configure_worker_process(**_kwargs) -> None:
    """Per-child init: cap BLAS/OpenMP/sklearn/joblib to avoid nested oversubscription."""
    from backend.workers.parallelism import configure_worker_parallelism

    report = configure_worker_parallelism(force=True)
    logger.info(
        "worker_process_init parallelism applied=%s blas=%s sklearn_n_jobs=%s",
        report.get("applied"),
        report.get("blas_threads"),
        report.get("sklearn_n_jobs"),
    )


@worker_ready.connect
def configure_worker_logging(sender: Any = None, **_kwargs) -> None:
    setup_logging()
    from backend.workers.parallelism import (
        configure_worker_parallelism,
        describe_parallelism_policy,
    )

    # Solo / threads pools may not emit worker_process_init the same way; apply once.
    configure_worker_parallelism(force=False)
    policy = describe_parallelism_policy()
    logger.info(
        "Celery worker ready broker=%s parallelism=%s",
        settings.celery_broker_url.split("@")[-1],
        {
            "blas_threads": policy["defaults"]["blas_threads"],
            "sklearn_n_jobs": policy["defaults"]["sklearn_n_jobs"],
            "joblib_n_jobs": policy["defaults"]["joblib_n_jobs"],
            "enabled": policy["defaults"]["enabled"],
        },
    )
    with contextlib.suppress(Exception):
        _start_r_worker_heartbeat(sender)


@worker_shutdown.connect
def close_worker_async_runtime(**_kwargs: Any) -> None:
    """Release clients owned by the dedicated worker event loop on exit."""
    with contextlib.suppress(Exception):
        from backend.modules.text_research.infrastructure.r_runtime.capabilities import (
            remove_r_worker_capabilities,
        )

        for worker_id, handle in _r_worker_heartbeat_handles.items():
            with contextlib.suppress(Exception):
                handle.cancel()
            remove_r_worker_capabilities(worker_id)
        _r_worker_heartbeat_handles.clear()
    with contextlib.suppress(Exception):
        from backend.workers.async_dispatch import shutdown_worker_async_runtime

        shutdown_worker_async_runtime()


@before_task_publish.connect
def attach_correlation_and_enqueue_time(
    sender: Any = None,
    headers: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> None:
    """Stamp correlation id + enqueue wall time onto Celery message headers."""
    if headers is None:
        return
    with contextlib.suppress(Exception):
        from backend.core.log_context import get_correlation_id

        correlation_id = get_correlation_id() or str(uuid4())
        headers.setdefault(CORRELATION_HEADER, correlation_id)
        headers.setdefault(ENQUEUED_AT_HEADER, time())


@task_prerun.connect
def log_task_start(task_id=None, task=None, **kwargs) -> None:
    if task_id is None:
        return
    _task_started_at[task_id] = perf_counter()
    headers = getattr(getattr(task, "request", None), "headers", None) or {}
    if not isinstance(headers, dict):
        headers = {}
    # Celery may also expose custom headers on the request directly.
    request = getattr(task, "request", None)
    correlation_id = (
        headers.get(CORRELATION_HEADER)
        or getattr(request, CORRELATION_HEADER, None)
        or str(uuid4())
    )
    with contextlib.suppress(Exception):
        _task_correlation_tokens[task_id] = set_correlation_id(str(correlation_id))

    enqueued_at = headers.get(ENQUEUED_AT_HEADER) or getattr(request, ENQUEUED_AT_HEADER, None)
    metrics = _safe_worker_metrics()
    if metrics is not None and enqueued_at is not None:
        with contextlib.suppress(Exception):
            delay = max(0.0, time() - float(enqueued_at))
            metrics.worker_task_queue_delay_seconds.labels(task=_task_label(task)).observe(delay)

    logger.info(
        "job_start task=%s id=%s correlation_id=%s",
        getattr(task, "name", "unknown"),
        task_id,
        correlation_id,
    )


@task_postrun.connect
def log_task_complete(task_id=None, task=None, state=None, **_kwargs) -> None:
    if task_id is None:
        return
    started = _task_started_at.pop(task_id, None)
    duration_s = (perf_counter() - started) if started is not None else -1.0
    duration_ms = duration_s * 1000 if duration_s >= 0 else -1.0
    task_name = getattr(task, "name", "unknown")
    label = _task_label(task)
    state_label = str(state or "UNKNOWN")

    metrics = _safe_worker_metrics()
    if metrics is not None and duration_s >= 0:
        with contextlib.suppress(Exception):
            metrics.worker_task_duration_seconds.labels(task=label, state=state_label).observe(
                duration_s
            )
            metrics.worker_tasks_total.labels(task=label, state=state_label).inc()

    token = _task_correlation_tokens.pop(task_id, None)
    if token is not None:
        with contextlib.suppress(Exception):
            reset_correlation_id(token)

    if state == "SUCCESS":
        if duration_ms >= settings.SLOW_JOB_MS:
            logger.warning(
                "slow_job task=%s id=%s duration_ms=%.2f state=%s",
                task_name,
                task_id,
                duration_ms,
                state,
            )
        else:
            logger.info(
                "job_complete task=%s id=%s duration_ms=%.2f state=%s",
                task_name,
                task_id,
                duration_ms,
                state,
            )
    else:
        logger.warning(
            "job_finished task=%s id=%s duration_ms=%.2f state=%s",
            task_name,
            task_id,
            duration_ms,
            state,
        )


@task_failure.connect
def log_task_failure(task_id=None, task=None, exception=None, **_kwargs) -> None:
    logger.error(
        "job_failed task=%s id=%s error_type=%s error_message=%s",
        getattr(task, "name", "unknown"),
        task_id,
        type(exception).__name__ if exception else "Exception",
        str(exception)[:300] if exception else "",
        exc_info=True,
    )
