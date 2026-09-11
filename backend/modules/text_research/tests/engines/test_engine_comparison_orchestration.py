"""Event-driven engine-comparison finalization (no parent polling)."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.engine_comparison import (
    COMPARISON_AWAITING_STAGE,
    COMPARISON_PARENT_RUN_ID_KEY,
)
from backend.modules.text_research.application.quantitative_analysis_service import (
    QUANT_OP_KEY,
    QuantitativeAnalysisService,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps, loads


def _run(
    *,
    run_id: str,
    status: str,
    parameters: dict | None = None,
    results: dict | None = None,
    error_message: str | None = None,
    corpus_id: str = "corpus-1",
    created_by: str = "user-1",
    project_id: str = "project-1",
    run_type: str = "engine_comparison",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        status=status,
        progress_stage=status,
        parameters_json=dumps(parameters or {}),
        metrics_json=dumps({}),
        results_json=dumps(results or {}),
        error_message=error_message,
        corpus_id=corpus_id,
        created_by=created_by,
        project_id=project_id,
        run_type=run_type,
        started_at=None,
        completed_at=None,
        run_version=1,
    )


class _FakeRepo:
    def __init__(self, runs: dict[str, SimpleNamespace], corpus=None) -> None:
        self.runs = runs
        self.corpus = corpus or SimpleNamespace(id="corpus-1", project_id="project-1")
        self.update_calls: list[tuple[str, dict]] = []

    async def get_run(self, run_id: str):
        return self.runs.get(run_id)

    async def get_corpus(self, corpus_id: str):
        return self.corpus if corpus_id == self.corpus.id else None

    async def update_run(self, run, **fields):
        self.update_calls.append((run.id, dict(fields)))
        for key, value in fields.items():
            setattr(run, key, value)
        run.run_version = int(getattr(run, "run_version", 1) or 1) + 1
        return run

    async def create_run(self, run):
        self.runs[run.id] = run
        return run


def _service(repo: _FakeRepo) -> QuantitativeAnalysisService:
    db = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    service = QuantitativeAnalysisService(db=db)
    service.repo = repo
    return service


class EngineComparisonOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_orchestration_enqueues_children_and_exits_without_polling(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.QUEUED.value,
            parameters={
                QUANT_OP_KEY: "engine_comparison",
                "unit_type": "paragraph",
                "analysis_type": "frequencies",
                "analysis_parameters": {"top_n": 5},
                "filters": {},
            },
        )
        repo = _FakeRepo({"parent": parent})
        service = _service(repo)
        service._select = AsyncMock(
            return_value=(
                repo.corpus,
                [SimpleNamespace(text="alpha beta", id="u1")],
                [],
            )
        )
        service._ensure_r_execution_capability = AsyncMock()
        service._resolve_existing_run = AsyncMock(return_value=parent)

        async def _enqueue_child(*_args, role: str, **_kwargs):
            return _run(
                run_id=f"{role}-1",
                status=AnalysisRunStatus.QUEUED.value,
                parameters={COMPARISON_PARENT_RUN_ID_KEY: "parent"},
                run_type="frequency_analysis",
            )

        service._enqueue_comparison_child = AsyncMock(side_effect=_enqueue_child)

        with patch(
            "backend.modules.text_research.application.quantitative_analysis_service.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock:
            result = await service.engine_comparison(
                "corpus-1",
                user_id="user-1",
                unit_type="paragraph",
                analysis_type="frequencies",
                analysis_parameters={"top_n": 5},
                force_inline=True,
                existing_run_id="parent",
            )

        self.assertEqual(sleep_mock.await_count, 0)
        self.assertEqual(result.status, AnalysisRunStatus.RUNNING.value)
        self.assertEqual(result.progress_stage, COMPARISON_AWAITING_STAGE)
        params = loads(result.parameters_json, {}) or {}
        self.assertEqual(params["python_run_id"], "python-1")
        self.assertEqual(params["r_run_id"], "r-1")
        self.assertEqual(service._enqueue_comparison_child.await_count, 2)

    async def test_finalize_both_children_succeed(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                "unit_type": "paragraph",
                "analysis_type": "frequencies",
                "analysis_parameters": {},
                "filters": {},
                "python_run_id": "py-1",
                "r_run_id": "r-1",
                QUANT_OP_KEY: "engine_comparison",
            },
        )
        python_run = _run(
            run_id="py-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"frequencies": [{"term": "alpha", "count": 2}]},
            run_type="frequency_analysis",
        )
        r_run = _run(
            run_id="r-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={
                "analysis_result": {
                    "results": {"frequencies": [{"term": "alpha", "count": 2}]},
                    "runtime": {"engine": "r"},
                }
            },
            run_type="frequency_analysis",
        )
        repo = _FakeRepo({"parent": parent, "py-1": python_run, "r-1": r_run})
        service = _service(repo)
        with patch(
            "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
            new=AsyncMock(side_effect=lambda _repo, run: run),
        ):
            finalized = await service.finalize_engine_comparison("parent")
        self.assertIsNotNone(finalized)
        self.assertEqual(finalized.status, AnalysisRunStatus.COMPLETED.value)
        results = loads(finalized.results_json, {}) or {}
        self.assertTrue(results["comparison"]["count_equal"])
        self.assertEqual(results["python_run_id"], "py-1")
        self.assertEqual(results["r_run_id"], "r-1")

    async def test_finalize_python_child_fails(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                "analysis_type": "frequencies",
                "python_run_id": "py-1",
                "r_run_id": "r-1",
            },
        )
        python_run = _run(
            run_id="py-1",
            status=AnalysisRunStatus.FAILED.value,
            error_message="boom",
            run_type="frequency_analysis",
        )
        r_run = _run(
            run_id="r-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"frequencies": []},
            run_type="frequency_analysis",
        )
        repo = _FakeRepo({"parent": parent, "py-1": python_run, "r-1": r_run})
        service = _service(repo)
        finalized = await service.finalize_engine_comparison("parent")
        self.assertEqual(finalized.status, AnalysisRunStatus.FAILED.value)
        self.assertIn("boom", finalized.error_message or "")

    async def test_finalize_r_child_fails(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                "analysis_type": "dfm",
                "python_run_id": "py-1",
                "r_run_id": "r-1",
            },
        )
        python_run = _run(
            run_id="py-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"dfm": {"feature_names": [], "unit_ids": [], "sparse": {}}},
            run_type="dfm",
        )
        r_run = _run(
            run_id="r-1",
            status=AnalysisRunStatus.CANCELLED.value,
            error_message="cancelled",
            run_type="dfm",
        )
        repo = _FakeRepo({"parent": parent, "py-1": python_run, "r-1": r_run})
        service = _service(repo)
        finalized = await service.finalize_engine_comparison("parent")
        self.assertEqual(finalized.status, AnalysisRunStatus.FAILED.value)
        self.assertIn("cancelled", finalized.error_message or "")

    async def test_finalize_waits_when_one_child_unfinished(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                "python_run_id": "py-1",
                "r_run_id": "r-1",
                "analysis_type": "frequencies",
            },
        )
        python_run = _run(
            run_id="py-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"frequencies": []},
        )
        r_run = _run(run_id="r-1", status=AnalysisRunStatus.RUNNING.value)
        repo = _FakeRepo({"parent": parent, "py-1": python_run, "r-1": r_run})
        service = _service(repo)
        result = await service.finalize_engine_comparison("parent")
        self.assertIs(result, parent)
        self.assertEqual(result.status, AnalysisRunStatus.RUNNING.value)

    async def test_duplicate_finalizer_is_idempotent(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.COMPLETED.value,
            parameters={
                "python_run_id": "py-1",
                "r_run_id": "r-1",
                "analysis_type": "frequencies",
            },
            results={"comparison": {"count_equal": True}},
        )
        repo = _FakeRepo({"parent": parent})
        service = _service(repo)
        result = await service.finalize_engine_comparison("parent")
        self.assertIs(result, parent)
        self.assertEqual(repo.update_calls, [])

    async def test_finalizer_retry_after_worker_restart_completes_parent(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                "analysis_type": "frequencies",
                "python_run_id": "py-1",
                "r_run_id": "r-1",
                "filters": {},
                "unit_type": "paragraph",
                "analysis_parameters": {},
            },
        )
        python_run = _run(
            run_id="py-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"frequencies": [{"term": "x", "count": 1}]},
        )
        r_run = _run(
            run_id="r-1",
            status=AnalysisRunStatus.COMPLETED.value,
            results={"frequencies": [{"term": "x", "count": 1}]},
        )
        repo = _FakeRepo({"parent": parent, "py-1": python_run, "r-1": r_run})
        service = _service(repo)
        with patch(
            "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
            new=AsyncMock(side_effect=lambda _repo, run: run),
        ):
            first = await service.finalize_engine_comparison("parent")
            second = await service.finalize_engine_comparison("parent")
        self.assertEqual(first.status, AnalysisRunStatus.COMPLETED.value)
        self.assertEqual(second.status, AnalysisRunStatus.COMPLETED.value)

    async def test_two_comparison_parents_finalize_independently(self) -> None:
        parents = {
            "p1": _run(
                run_id="p1",
                status=AnalysisRunStatus.RUNNING.value,
                parameters={
                    "analysis_type": "frequencies",
                    "python_run_id": "py-a",
                    "r_run_id": "r-a",
                    "filters": {},
                    "unit_type": "paragraph",
                    "analysis_parameters": {},
                },
            ),
            "p2": _run(
                run_id="p2",
                status=AnalysisRunStatus.RUNNING.value,
                parameters={
                    "analysis_type": "frequencies",
                    "python_run_id": "py-b",
                    "r_run_id": "r-b",
                    "filters": {},
                    "unit_type": "paragraph",
                    "analysis_parameters": {},
                },
            ),
        }
        children = {
            "py-a": _run(
                run_id="py-a",
                status=AnalysisRunStatus.COMPLETED.value,
                results={"frequencies": [{"term": "a", "count": 1}]},
            ),
            "r-a": _run(
                run_id="r-a",
                status=AnalysisRunStatus.COMPLETED.value,
                results={"frequencies": [{"term": "a", "count": 1}]},
            ),
            "py-b": _run(
                run_id="py-b",
                status=AnalysisRunStatus.COMPLETED.value,
                results={"frequencies": [{"term": "b", "count": 2}]},
            ),
            "r-b": _run(
                run_id="r-b",
                status=AnalysisRunStatus.COMPLETED.value,
                results={"frequencies": [{"term": "b", "count": 2}]},
            ),
        }
        repo = _FakeRepo({**parents, **children})
        service = _service(repo)
        with patch(
            "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
            new=AsyncMock(side_effect=lambda _repo, run: run),
        ):
            out1, out2 = await asyncio.gather(
                service.finalize_engine_comparison("p1"),
                service.finalize_engine_comparison("p2"),
            )
        self.assertEqual(out1.status, AnalysisRunStatus.COMPLETED.value)
        self.assertEqual(out2.status, AnalysisRunStatus.COMPLETED.value)
        self.assertEqual(loads(out1.results_json)["python_run_id"], "py-a")
        self.assertEqual(loads(out2.results_json)["python_run_id"], "py-b")

    async def test_cpu_concurrency_one_safe_because_parent_does_not_wait(self) -> None:
        parent = _run(
            run_id="parent",
            status=AnalysisRunStatus.QUEUED.value,
            parameters={QUANT_OP_KEY: "engine_comparison"},
        )
        repo = _FakeRepo({"parent": parent})
        service = _service(repo)
        service._select = AsyncMock(
            return_value=(repo.corpus, [SimpleNamespace(text="t", id="u1")], [])
        )
        service._ensure_r_execution_capability = AsyncMock()
        service._resolve_existing_run = AsyncMock(return_value=parent)
        enqueued: list[str] = []

        async def _enqueue_child(*_a, role: str, **_k):
            enqueued.append(role)
            return _run(run_id=f"{role}-child", status=AnalysisRunStatus.QUEUED.value)

        service._enqueue_comparison_child = AsyncMock(side_effect=_enqueue_child)
        result = await service.engine_comparison(
            "corpus-1",
            user_id="user-1",
            unit_type="paragraph",
            analysis_type="frequencies",
            analysis_parameters={},
            force_inline=True,
            existing_run_id="parent",
        )
        self.assertEqual(enqueued, ["python", "r"])
        self.assertEqual(result.progress_stage, COMPARISON_AWAITING_STAGE)
        self.assertEqual(result.status, AnalysisRunStatus.RUNNING.value)

    async def test_child_persist_schedules_finalize(self) -> None:
        child = _run(
            run_id="py-1",
            status=AnalysisRunStatus.RUNNING.value,
            parameters={
                COMPARISON_PARENT_RUN_ID_KEY: "parent",
                QUANT_OP_KEY: "frequencies",
            },
            run_type="frequency_analysis",
        )
        repo = _FakeRepo({"py-1": child})
        service = _service(repo)
        with (
            patch(
                "backend.modules.text_research.application.run_lifecycle.ensure_not_cancelled",
                new=AsyncMock(side_effect=lambda _repo, run: run),
            ),
            patch.object(service, "_schedule_comparison_finalize") as schedule,
        ):
            await service._persist_run(
                repo.corpus,
                MagicMock(value="frequency_analysis"),
                user_id="user-1",
                parameters={"unit_type": "paragraph"},
                metrics={},
                results={"frequencies": []},
                existing_run=child,
            )
        schedule.assert_called_once_with("parent", user_id="user-1")


if __name__ == "__main__":
    unittest.main()
