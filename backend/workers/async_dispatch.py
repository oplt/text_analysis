"""Shared helpers for running async jobs from sync Celery tasks or dev eager mode."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from typing import Any, TypeVar

from backend.core.config import settings

logger = logging.getLogger(__name__)

_eager_warning_logged = False
_eager_executor: ThreadPoolExecutor | None = None
_eager_slots: BoundedSemaphore | None = None
_eager_lock = Lock()
_worker_runtime: _WorkerAsyncRuntime | None = None
_worker_runtime_lock = Lock()

T = TypeVar("T")


class _WorkerAsyncRuntime:
    """One worker-process loop and pooled engine, isolated from FastAPI's loop."""

    def __init__(self) -> None:
        self._ready = Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sessionmaker: Any = None
        self._startup_error: BaseException | None = None
        self._jobs: Queue[tuple[Coroutine[Any, Any, Any], Future[Any]]] = Queue()
        self._thread = Thread(target=self._run, name="research-worker-async", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Unable to initialize worker async runtime") from self._startup_error
        assert self._loop is not None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from backend.db.session import create_worker_sessionmaker

            _engine, self._sessionmaker = create_worker_sessionmaker()
            self._loop = loop
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
        finally:
            self._ready.set()
        if self._startup_error is not None:
            return
        while True:
            coro, future = self._jobs.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(loop.run_until_complete(self._execute(coro)))
                except BaseException as exc:  # noqa: BLE001
                    future.set_exception(exc)

    async def _execute(self, coro: Coroutine[Any, Any, T]) -> T:
        from backend.db.session import _sessionmaker_override

        token = _sessionmaker_override.set(self._sessionmaker)
        try:
            return await coro
        finally:
            _sessionmaker_override.reset(token)

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        future: Future[T] = Future()
        self._jobs.put((coro, future))
        return future.result()


def _get_worker_runtime() -> _WorkerAsyncRuntime:
    global _worker_runtime
    with _worker_runtime_lock:
        if _worker_runtime is None:
            _worker_runtime = _WorkerAsyncRuntime()
        return _worker_runtime


def run_async_in_sync_context[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the worker-process event loop and pooled engine.

    The runtime is deliberately separate from FastAPI's event loop. It is
    initialized lazily inside the Celery/eager worker process and reused by
    subsequent jobs, so connection pools are not recreated per execution.
    """
    return _get_worker_runtime().run(coro)


def _eager_executor_for_settings() -> tuple[ThreadPoolExecutor, BoundedSemaphore]:
    """Lazily construct a process-local bounded executor for eager development."""
    global _eager_executor, _eager_slots
    with _eager_lock:
        if _eager_executor is None:
            workers = max(1, settings.EAGER_BACKGROUND_MAX_WORKERS)
            pending = max(workers, settings.EAGER_BACKGROUND_MAX_PENDING)
            _eager_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="eager-research",
            )
            _eager_slots = BoundedSemaphore(pending)
        assert _eager_slots is not None
        return _eager_executor, _eager_slots


def _release_eager_slot(_: Future[Any], slots: BoundedSemaphore) -> None:
    slots.release()


def dispatch_background_sync_job(
    *,
    target: Callable[..., None],
    kwargs: dict[str, Any],
    celery_task: Any,
    celery_kwargs: dict[str, Any],
    queue: str,
    job_name: str,
) -> str | None:
    """Queue work on Celery, or a bounded local executor in eager mode."""
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
        executor, slots = _eager_executor_for_settings()
        if not slots.acquire(blocking=False):
            raise RuntimeError(
                "Eager background capacity is exhausted; run Celery workers or retry shortly."
            )
        future = executor.submit(target, **kwargs)
        future.add_done_callback(lambda completed: _release_eager_slot(completed, slots))
        return None

    result = celery_task.apply_async(kwargs=celery_kwargs, queue=queue)
    return str(result.id)


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
