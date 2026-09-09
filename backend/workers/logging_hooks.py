"""Celery worker logging hooks."""

from __future__ import annotations

import logging
from time import perf_counter

from celery.signals import task_failure, task_postrun, task_prerun, worker_process_init, worker_ready

from backend.core.config import settings
from backend.core.logging import setup_logging

logger = logging.getLogger("backend.worker")

_task_started_at: dict[str, float] = {}


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
def configure_worker_logging(**_kwargs) -> None:
    setup_logging()
    from backend.workers.parallelism import configure_worker_parallelism, describe_parallelism_policy

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


@task_prerun.connect
def log_task_start(task_id=None, task=None, **_kwargs) -> None:
    if task_id is None:
        return
    _task_started_at[task_id] = perf_counter()
    logger.info("job_start task=%s id=%s", getattr(task, "name", "unknown"), task_id)


@task_postrun.connect
def log_task_complete(task_id=None, task=None, state=None, **_kwargs) -> None:
    if task_id is None:
        return
    started = _task_started_at.pop(task_id, None)
    duration_ms = (perf_counter() - started) * 1000 if started is not None else -1.0
    task_name = getattr(task, "name", "unknown")
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
