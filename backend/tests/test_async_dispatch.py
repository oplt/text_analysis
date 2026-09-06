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


@patch("backend.workers.async_dispatch.threading.Thread")
@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", True)
def test_dispatch_background_sync_job_uses_daemon_thread_when_eager(
    mock_thread_cls: MagicMock,
) -> None:
    mock_thread = MagicMock()
    mock_thread_cls.return_value = mock_thread
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

    mock_thread_cls.assert_called_once_with(
        target=target,
        kwargs={"document_id": "doc-1"},
        name="eager-rag-indexing",
        daemon=True,
    )
    mock_thread.start.assert_called_once()
    celery_task.apply_async.assert_not_called()


@patch("backend.workers.async_dispatch.threading.Thread")
@patch("backend.workers.async_dispatch.settings.CELERY_TASK_ALWAYS_EAGER", False)
def test_dispatch_background_sync_job_uses_celery_when_not_eager(
    mock_thread_cls: MagicMock,
) -> None:
    celery_task = MagicMock()

    async_dispatch.dispatch_background_sync_job(
        target=MagicMock(),
        kwargs={"document_id": "doc-1"},
        celery_task=celery_task,
        celery_kwargs={"document_id": "doc-1", "user_id": "user-1"},
        queue="default",
        job_name="rag-indexing",
    )

    mock_thread_cls.assert_not_called()
    celery_task.apply_async.assert_called_once_with(
        kwargs={"document_id": "doc-1", "user_id": "user-1"},
        queue="default",
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
