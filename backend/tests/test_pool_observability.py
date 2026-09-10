"""Phase 19/22: database pool and research observability smoke tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.db.pool_observability import instrument_pool, update_pool_gauges


class PoolObservabilityTests(unittest.TestCase):
    def test_update_pool_gauges_sets_saturation(self) -> None:
        pool = MagicMock()
        pool.checkedout.return_value = 3
        pool.size.return_value = 5
        pool.overflow.return_value = 0
        pool._max_overflow = 5

        metrics = SimpleNamespace(
            database_pool_checked_out=MagicMock(),
            database_pool_saturation=MagicMock(),
            database_pool_size=MagicMock(),
            database_pool_overflow=MagicMock(),
        )
        for name in (
            "database_pool_checked_out",
            "database_pool_saturation",
            "database_pool_size",
            "database_pool_overflow",
        ):
            getattr(metrics, name).labels.return_value = MagicMock()

        with patch("backend.db.pool_observability._safe_metrics", return_value=metrics):
            update_pool_gauges(role="api", pool=pool)

        metrics.database_pool_checked_out.labels.assert_called_with(role="api")
        metrics.database_pool_saturation.labels.return_value.set.assert_called_with(0.3)

    def test_instrument_pool_wraps_do_get_and_records_wait(self) -> None:
        pool = MagicMock()
        pool.checkedout.return_value = 1
        pool.size.return_value = 2
        pool.overflow.return_value = 0
        pool._max_overflow = 1
        connection = object()
        pool._do_get = MagicMock(return_value=connection)

        wait_hist = MagicMock()
        metrics = SimpleNamespace(
            database_pool_wait_seconds=MagicMock(labels=MagicMock(return_value=wait_hist)),
            database_pool_checked_out=MagicMock(labels=MagicMock(return_value=MagicMock())),
            database_pool_saturation=MagicMock(labels=MagicMock(return_value=MagicMock())),
            database_pool_size=MagicMock(labels=MagicMock(return_value=MagicMock())),
            database_pool_overflow=MagicMock(labels=MagicMock(return_value=MagicMock())),
            database_pool_timeouts_total=MagicMock(labels=MagicMock(return_value=MagicMock())),
            database_pool_connection_failures_total=MagicMock(
                labels=MagicMock(return_value=MagicMock())
            ),
        )

        # Use a fresh MagicMock pool id each run by creating a unique object.
        unique_pool = MagicMock()
        unique_pool.checkedout = pool.checkedout
        unique_pool.size = pool.size
        unique_pool.overflow = pool.overflow
        unique_pool._max_overflow = 1
        unique_pool._do_get = pool._do_get

        with (
            patch("backend.db.pool_observability._safe_metrics", return_value=metrics),
            patch("backend.db.pool_observability.event.listens_for", MagicMock()),
        ):
            instrument_pool(role="worker", pool=unique_pool)
            got = unique_pool._do_get()

        self.assertIs(got, connection)
        wait_hist.observe.assert_called()
        self.assertGreaterEqual(wait_hist.observe.call_args.args[0], 0.0)


class PrometheusMetricsImportTests(unittest.TestCase):
    def test_phase22_metric_names_are_defined(self) -> None:
        from backend.observability import prometheus_metrics as m

        for attr in (
            "research_stage_cache_hits_total",
            "research_prepared_corpus_cache_hits_total",
            "research_preprocessing_duration_seconds",
            "research_sse_active_connections",
            "worker_task_queue_delay_seconds",
            "worker_task_duration_seconds",
            "database_pool_saturation",
            "database_query_duration_seconds",
            "embedding_cache_hits",
        ):
            # embedding_cache_hits lives in embedding_cache; others in prometheus_metrics
            if attr == "embedding_cache_hits":
                from backend.lib import embedding_cache

                self.assertTrue(hasattr(embedding_cache, attr))
            else:
                self.assertTrue(hasattr(m, attr), attr)


if __name__ == "__main__":
    unittest.main()
