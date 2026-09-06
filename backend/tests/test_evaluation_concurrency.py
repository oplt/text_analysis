from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.modules.ai.evaluation_service as ai_service_module
from backend.modules.ai.service import AiService, _EvaluationCaseResult


@pytest.fixture
def service() -> AiService:
    db = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return AiService(db)


def test_list_all_dataset_cases_fetches_every_page(service: AiService) -> None:
    page_one = [SimpleNamespace(id="case-1"), SimpleNamespace(id="case-2")]
    page_two = [SimpleNamespace(id="case-3")]
    service.repo.list_dataset_cases = AsyncMock(
        side_effect=[
            (page_one, 3),
            (page_two, 3),
        ]
    )

    cases = asyncio.run(service._list_all_dataset_cases("dataset-1"))

    assert [case.id for case in cases] == ["case-1", "case-2", "case-3"]
    assert service.repo.list_dataset_cases.await_count == 2


def test_execute_evaluation_respects_concurrency_limit(
    service: AiService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ai_service_module.settings, "AI_EVALUATION_CONCURRENCY", 2)

    cases = [SimpleNamespace(id=f"case-{index}") for index in range(4)]
    dataset = SimpleNamespace(id="dataset-1")
    version = SimpleNamespace(id="version-1", prompt_template_id="template-1")
    template = SimpleNamespace(id="template-1", key="prompt-key")
    evaluation_run = SimpleNamespace(
        id="run-1",
        status="running",
        total_cases=4,
        passed_cases=0,
        average_score=0.0,
        completed_at=None,
    )
    user = SimpleNamespace(id="user-1")

    service.repo.get_dataset_for_user = AsyncMock(return_value=dataset)
    service.repo.get_prompt_version = AsyncMock(return_value=version)
    service.repo.get_prompt_template_for_user = AsyncMock(return_value=template)
    service.repo.get_evaluation_run_by_id = AsyncMock(return_value=evaluation_run)
    service._list_all_dataset_cases = AsyncMock(return_value=cases)
    service.repo.create_evaluation_run_items_batch = AsyncMock()

    with patch.object(
        ai_service_module.IdentityRepository,
        "get_user_by_id",
        AsyncMock(return_value=user),
    ):
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def fake_execute(**kwargs) -> _EvaluationCaseResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            async with lock:
                active -= 1
            case = kwargs["case"]
            return _EvaluationCaseResult(
                case_id=case.id,
                ai_run_id=f"ai-run-{case.id}",
                score=1.0,
                passed=True,
                notes="ok",
            )

        service._execute_evaluation_case = fake_execute  # type: ignore[method-assign]

        asyncio.run(
            service.execute_evaluation_run(
                evaluation_run_id="run-1",
                user_id="user-1",
                dataset_id="dataset-1",
                prompt_version_id="version-1",
            )
        )

    assert max_active <= 2
    assert max_active > 1
    assert service.repo.create_evaluation_run_items_batch.await_count == 1
    batch_payload = service.repo.create_evaluation_run_items_batch.await_args.args[0]
    assert len(batch_payload) == 4
    assert evaluation_run.status == "completed"
    assert evaluation_run.passed_cases == 4
    service.db.commit.assert_awaited()


@patch("backend.workers.evaluation.queue_evaluation_run")
def test_queue_evaluation_persists_running_run_and_dispatches_worker(
    mock_queue: MagicMock, service: AiService
) -> None:
    dataset = SimpleNamespace(id="dataset-1")
    version = SimpleNamespace(id="version-1", prompt_template_id="template-1")
    template = SimpleNamespace(id="template-1")
    evaluation_run = SimpleNamespace(
        id="run-1",
        dataset_id="dataset-1",
        prompt_version_id="version-1",
    )
    user = SimpleNamespace(id="user-1")

    service.repo.get_dataset_for_user = AsyncMock(return_value=dataset)
    service.repo.get_prompt_version = AsyncMock(return_value=version)
    service.repo.get_prompt_template_for_user = AsyncMock(return_value=template)
    service._list_all_dataset_cases = AsyncMock(return_value=[SimpleNamespace(id="case-1")])
    service.repo.create_evaluation_run = AsyncMock(return_value=evaluation_run)

    result = asyncio.run(
        service.queue_evaluation(user, "dataset-1", "version-1")  # type: ignore[arg-type]
    )

    assert result is evaluation_run
    service.repo.create_evaluation_run.assert_awaited_once()
    create_kwargs = service.repo.create_evaluation_run.await_args.kwargs
    assert create_kwargs["status"] == "running"
    assert create_kwargs["total_cases"] == 1
    service.db.commit.assert_awaited_once()
    mock_queue.assert_called_once_with(
        evaluation_run_id="run-1",
        user_id="user-1",
        dataset_id="dataset-1",
        prompt_version_id="version-1",
    )
