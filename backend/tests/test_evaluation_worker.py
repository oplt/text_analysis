from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.workers.evaluation import queue_evaluation_run


@patch("backend.workers.evaluation.dispatch_background_sync_job")
def test_queue_evaluation_run_dispatches_background_job(mock_dispatch: MagicMock) -> None:
    queue_evaluation_run(
        evaluation_run_id="run-1",
        user_id="user-1",
        dataset_id="dataset-1",
        prompt_version_id="version-1",
    )

    mock_dispatch.assert_called_once()
    kwargs = mock_dispatch.call_args.kwargs
    assert kwargs["job_name"] == "ai-evaluation"
    assert kwargs["celery_kwargs"] == {
        "evaluation_run_id": "run-1",
        "user_id": "user-1",
        "dataset_id": "dataset-1",
        "prompt_version_id": "version-1",
    }
