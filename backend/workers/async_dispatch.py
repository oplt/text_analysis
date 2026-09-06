"""Shared helpers for running async jobs from sync Celery tasks or dev eager mode."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from backend.core.config import settings

logger = logging.getLogger(__name__)

_eager_warning_logged = False

T = TypeVar("T")


def run_async_in_sync_context(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from a sync worker entrypoint (Celery task thread)."""
    return asyncio.run(coro)


def dispatch_background_sync_job(
    *,
    target: Callable[..., None],
    kwargs: dict[str, Any],
    celery_task: Any,
    celery_kwargs: dict[str, Any],
    queue: str,
    job_name: str,
) -> None:
    """Queue work on Celery, or run it in a daemon thread when eager mode is enabled."""
    global _eager_warning_logged

    if settings.CELERY_TASK_ALWAYS_EAGER:
        if not _eager_warning_logged:
            logger.warning(
                "CELERY_TASK_ALWAYS_EAGER=true: %s jobs run in background threads inside "
                "the API process. Set CELERY_TASK_ALWAYS_EAGER=false and run Celery workers "
                "in production.",
                job_name,
            )
            _eager_warning_logged = True
        thread = threading.Thread(
            target=target,
            kwargs=kwargs,
            name=f"eager-{job_name}",
            daemon=True,
        )
        thread.start()
        return

    celery_task.apply_async(kwargs=celery_kwargs, queue=queue)


def log_eager_mode_startup_warning() -> None:
    """Emit a single startup warning when eager Celery mode is enabled."""
    if not settings.CELERY_TASK_ALWAYS_EAGER:
        return
    if settings.APP_ENV == "production":
        logger.warning(
            "CELERY_TASK_ALWAYS_EAGER is enabled in production. Background jobs execute "
            "inside the API process instead of dedicated Celery workers."
        )
        return
    logger.info(
        "CELERY_TASK_ALWAYS_EAGER=true for local development. Indexing and email jobs "
        "will not use a separate Celery worker process."
    )
