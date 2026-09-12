"""Phase 5: workload estimator thresholds and large-job enqueue path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.analysis_executor import (
    execute_or_enqueue,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.infrastructure.workload_estimator import (
    estimate_pair_count,
    estimate_token_count,
    estimate_workload,
    exceeds_async_threshold,
)


class WorkloadEstimatorTests(unittest.TestCase):
    def test_token_and_pair_estimators(self) -> None:
        self.assertEqual(estimate_token_count(["abcd", "efgh"]), 2)  # 8 chars / 4
        self.assertEqual(estimate_pair_count(5), 10)
        self.assertEqual(estimate_pair_count(1), 0)
        # Bounded top-K pairwise after TASK-007.
        self.assertEqual(estimate_pair_count(100, mode="pairwise", top_k=20), min(4950, 100 * 20))
        self.assertEqual(estimate_pair_count(50, mode="query"), 50)
        self.assertEqual(estimate_pair_count(100, mode="group_centroid", n_groups=4), 6)
        self.assertEqual(estimate_pair_count(100, mode="cooccurrence", window_size=5), 500)
        self.assertEqual(estimate_pair_count(1000, mode="sweep", n_models=3), 3000)

    def test_similarity_topk_workload_uses_bounded_pairs(self) -> None:
        estimate = estimate_workload(
            analysis_type="similarity",
            n_units=1_000,
            pair_mode="pairwise",
            requested_top_k=50,
        )
        self.assertEqual(estimate.estimated_pairs, 1_000 * 50)
        query = estimate_workload(
            analysis_type="similarity",
            n_units=1_000,
            pair_mode="query",
            requested_top_k=50,
        )
        self.assertEqual(query.estimated_pairs, 1_000)

    def test_thresholds_from_settings(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RESEARCH_ASYNC_UNIT_THRESHOLD": "100",
                "RESEARCH_ASYNC_TOKEN_THRESHOLD": "1000",
                "RESEARCH_ASYNC_PAIR_THRESHOLD": "500",
            },
        ):
            # Force settings reload of thresholds via async_thresholds reading settings
            from backend.core.config import settings

            with (
                patch.object(settings, "RESEARCH_ASYNC_UNIT_THRESHOLD", 100),
                patch.object(settings, "RESEARCH_ASYNC_TOKEN_THRESHOLD", 1000),
                patch.object(settings, "RESEARCH_ASYNC_PAIR_THRESHOLD", 500),
            ):
                small = estimate_workload(analysis_type="frequencies", n_units=10, texts=["hi"])
                self.assertFalse(exceeds_async_threshold(small))
                self.assertFalse(should_enqueue_cpu_job(small))

                by_units = estimate_workload(analysis_type="dfm", n_units=100, texts=["x"] * 100)
                self.assertTrue(exceeds_async_threshold(by_units))
                self.assertTrue(should_enqueue_cpu_job(by_units))

                by_tokens = estimate_workload(
                    analysis_type="frequencies",
                    n_units=1,
                    estimated_tokens=1000,
                )
                self.assertTrue(exceeds_async_threshold(by_tokens))

                by_pairs = estimate_workload(
                    analysis_type="similarity",
                    n_units=50,
                    estimated_pairs=500,
                )
                self.assertTrue(exceeds_async_threshold(by_pairs))

                self.assertFalse(should_enqueue_cpu_job(by_units, force_inline=True))
                self.assertTrue(should_enqueue_cpu_job(small, force_async=True))


class LargeJobEnqueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_or_enqueue_chooses_enqueue_for_large(self) -> None:
        estimate = estimate_workload(analysis_type="dfm", n_units=10_000, texts=["word"] * 10_000)
        inline = AsyncMock(return_value="inline")
        enqueue = AsyncMock(return_value="queued")

        with patch(
            "backend.modules.text_research.application.analysis_executor.exceeds_async_threshold",
            return_value=True,
        ):
            result = await execute_or_enqueue(estimate=estimate, inline=inline, enqueue=enqueue)

        self.assertEqual(result, "queued")
        enqueue.assert_awaited_once()
        inline.assert_not_awaited()

    async def test_run_async_true_forces_queue_on_small_request(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(db=MagicMock())
        corpus = SimpleNamespace(id="c1", project_id="p1")
        units = [SimpleNamespace(id="u1", text="hi", corpus_document_id="d1")]
        documents = [SimpleNamespace(id="d1")]
        queued_run = SimpleNamespace(
            id="run-forced",
            status=AnalysisRunStatus.QUEUED.value,
            corpus_id="c1",
        )
        service._select = AsyncMock(return_value=(corpus, units, documents))
        service._enqueue_quantitative = AsyncMock(return_value=queued_run)

        result = await service.frequencies(
            "c1",
            user_id="user-1",
            unit_type="paragraph",
            run_async=True,
        )

        self.assertEqual(result.status, AnalysisRunStatus.QUEUED.value)
        service._enqueue_quantitative.assert_awaited_once()

    async def test_worker_reentry_never_requeues(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(db=MagicMock())
        corpus = SimpleNamespace(id="c1", project_id="p1")
        units = [SimpleNamespace(id="u1", text="hi", corpus_document_id="d1")]
        documents = [SimpleNamespace(id="d1")]
        completed = SimpleNamespace(
            id="run-inline",
            status=AnalysisRunStatus.COMPLETED.value,
            corpus_id="c1",
        )
        service._select = AsyncMock(return_value=(corpus, units, documents))
        service._enqueue_quantitative = AsyncMock()
        service._resolve_existing_run = AsyncMock(return_value=completed)
        service._resolve_config = AsyncMock(return_value=({}, {}))
        service._persist_run = AsyncMock(return_value=completed)

        with (
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service._prepare_with_identity",
                new=AsyncMock(
                    return_value=(
                        SimpleNamespace(
                            corpus_checksum="c",
                            pipeline_checksum="p",
                            token_sequences=[["hi"]],
                            texts_joined=["hi"],
                            preprocessing_profile={},
                        ),
                        {},
                    )
                ),
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.run_cpu_bound",
                new=AsyncMock(
                    return_value={
                        "frequencies": [],
                        "metadata": {
                            "unit_count": 1,
                            "token_count": 1,
                            "vocabulary_size": 1,
                            "terms_returned": 0,
                            "rate_per": 1000,
                        },
                    }
                ),
            ),
        ):
            result = await service.frequencies(
                "c1",
                user_id="user-1",
                unit_type="paragraph",
                run_async=True,
                force_inline=True,
                existing_run_id="run-inline",
            )

        self.assertEqual(result.status, AnalysisRunStatus.COMPLETED.value)
        service._enqueue_quantitative.assert_not_awaited()

    async def test_frequencies_large_job_enqueues_not_blocks(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(db=MagicMock())
        corpus = SimpleNamespace(id="c1", project_id="p1")
        units = [
            SimpleNamespace(id=f"u{i}", text=f"token{i} " * 20, corpus_document_id=f"d{i}")
            for i in range(50)
        ]
        documents = [SimpleNamespace(id=f"d{i}") for i in range(50)]

        queued_run = SimpleNamespace(
            id="run-queued",
            status=AnalysisRunStatus.QUEUED.value,
            corpus_id="c1",
        )

        service._select = AsyncMock(return_value=(corpus, units, documents))
        service._enqueue_quantitative = AsyncMock(return_value=queued_run)

        with (
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.estimate_workload",
                return_value=estimate_workload(analysis_type="frequencies", n_units=50_000),
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.should_enqueue_cpu_job",
                return_value=True,
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.execute_or_enqueue",
                side_effect=execute_or_enqueue,
            ),
        ):
            result = await service.frequencies(
                "c1",
                user_id="user-1",
                unit_type="paragraph",
            )

        self.assertEqual(result.status, AnalysisRunStatus.QUEUED.value)
        service._enqueue_quantitative.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
