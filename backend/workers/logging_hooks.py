"""Celery worker logging hooks."""

from __future__ import annotations

import logging
from time import perf_counter

from celery.signals import task_failure, task_postrun, task_prerun, worker_ready

from backend.core.config import settings
from backend.core.logging import setup_logging

logger = logging.getLogger("backend.worker")

_task_started_at: dict[str, float] = {}


@worker_ready.connect
def configure_worker_logging(**_kwargs) -> None:
    setup_logging()
    logger.info("Celery worker ready broker=%s", settings.celery_broker_url.split("@")[-1])


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
