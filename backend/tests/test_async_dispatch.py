from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import backend.workers.async_dispatch as async_dispatch


@pytest.fixture(autouse=True)
def reset_eager_warning_flag() -> None:
    async_dispatch._eager_warning_logged = False
    yield
    async_dispatch._eager_warning_logged = False


def test_run_async_in_sync_context_executes_coroutine() -> None:
    seen: list[int] = []

    async def _work() -> None:
        seen.append(1)

    async_dispatch.run_async_in_sync_context(_work())
    assert seen == [1]


def test_run_async_in_sync_context_installs_isolated_sessionmaker() -> None:
    from backend.db import session as db_session

    observed: list[bool] = []

    async def _work() -> None:
        observed.append(db_session._sessionmaker_override.get() is not None)

    async_dispatch.run_async_in_sync_context(_work())
    assert observed == [True]
    assert db_session._sessionmaker_override.get() is None


def test_run_async_in_sync_context_reuses_one_worker_runtime() -> None:
    first = async_dispatch._get_worker_runtime()

    async def _work() -> None:
        return None

    async_dispatch.run_async_in_sync_context(_work())
    assert async_dispatch._get_worker_runtime() is first


@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True)
@patch("backend.workers.async_dispatch._eager_executor_for_settings")
def test_dispatch_background_sync_job_uses_bounded_executor_when_eager(
    mock_executor_for_settings: MagicMock,
) -> None:
    executor = MagicMock()
    slots = MagicMock()
    slots.acquire.return_value = True
    mock_future = MagicMock()
    executor.submit.return_value = mock_future
    mock_executor_for_settings.return_value = (executor, slots)
    celery_task = MagicMock()
    target = MagicMock()

    async_dispatch.dispatch_background_sync_job(
        target=target,
        kwargs={"document_id": "doc-1"},
        celery_task=celery_task,
        celery_kwargs={"document_id": "doc-1"},
        queue="default",
        job_name="rag-indexing",
    )

    executor.submit.assert_called_once_with(target, document_id="doc-1")
    mock_future.add_done_callback.assert_called_once()
    celery_task.apply_async.assert_not_called()


@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", False)
def test_dispatch_background_sync_job_uses_celery_when_not_eager() -> None:
    celery_task = MagicMock()

    async_dispatch.dispatch_background_sync_job(
        target=MagicMock(),
        kwargs={"document_id": "doc-1"},
        celery_task=celery_task,
        celery_kwargs={"document_id": "doc-1", "user_id": "user-1"},
        queue="default",
        job_name="rag-indexing",
    )

    celery_task.apply_async.assert_called_once()
    call_kwargs = celery_task.apply_async.call_args.kwargs
    assert call_kwargs["kwargs"] == {"document_id": "doc-1", "user_id": "user-1"}
    assert call_kwargs["queue"] == "default"
    assert "correlation_id" in call_kwargs["headers"]
    assert "enqueued_at" in call_kwargs["headers"]


@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True)
@patch("backend.workers.async_dispatch._eager_executor_for_settings")
def test_eager_dispatch_rejects_when_capacity_is_exhausted(
    mock_executor_for_settings: MagicMock,
) -> None:
    slots = MagicMock()
    slots.acquire.return_value = False
    mock_executor_for_settings.return_value = (MagicMock(), slots)

    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        async_dispatch.dispatch_background_sync_job(
            target=MagicMock(),
            kwargs={},
            celery_task=MagicMock(),
            celery_kwargs={},
            queue="default",
            job_name="research-test",
        )


@patch("backend.workers.async_dispatch.logger")
@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True)
@patch("backend.workers.async_dispatch.settings.APP_ENV", "production")
def test_log_eager_mode_startup_warning_in_production(mock_logger: MagicMock) -> None:
    async_dispatch.log_eager_mode_startup_warning()

    mock_logger.warning.assert_called_once()
    mock_logger.info.assert_not_called()


@patch("backend.workers.async_dispatch.logger")
@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True)
@patch("backend.workers.async_dispatch.settings.APP_ENV", "dev")
def test_log_eager_mode_startup_warning_in_dev(mock_logger: MagicMock) -> None:
    async_dispatch.log_eager_mode_startup_warning()

    mock_logger.info.assert_called_once()
    mock_logger.warning.assert_not_called()
