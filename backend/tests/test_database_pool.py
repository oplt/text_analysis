"""Explicit API/worker SQLAlchemy pool configuration (Phase 7)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.db import session as db_session
from backend.workers import parallelism


class DatabasePoolConfigTests(unittest.TestCase):
    def test_api_and_worker_kwargs_are_explicit_and_distinct(self) -> None:
        with patch.object(db_session, "settings") as settings:
            settings.DB_POOL_SIZE = 8
            settings.DB_MAX_OVERFLOW = 12
            settings.DB_POOL_TIMEOUT = 45
            settings.DB_POOL_RECYCLE = 900
            settings.DB_WORKER_POOL_SIZE = 1
            settings.DB_WORKER_MAX_OVERFLOW = 0
            settings.DB_WORKER_POOL_TIMEOUT = 20
            settings.DB_WORKER_POOL_RECYCLE = 600

            api = db_session.database_engine_kwargs(role="api")
            worker = db_session.database_engine_kwargs(role="worker")

        self.assertEqual(api["pool_size"], 8)
        self.assertEqual(api["max_overflow"], 12)
        self.assertEqual(api["pool_timeout"], 45)
        self.assertEqual(api["pool_recycle"], 900)
        self.assertTrue(api["pool_pre_ping"])

        self.assertEqual(worker["pool_size"], 1)
        self.assertEqual(worker["max_overflow"], 0)
        self.assertEqual(worker["pool_timeout"], 20)
        self.assertEqual(worker["pool_recycle"], 600)
        self.assertNotEqual(api["pool_size"], worker["pool_size"])

    def test_describe_database_pool_policy_includes_pgbouncer(self) -> None:
        policy = db_session.describe_database_pool_policy()
        self.assertIn("api", policy)
        self.assertIn("worker", policy)
        self.assertTrue(policy["pgbouncer"]["recommended"])
        self.assertEqual(policy["pgbouncer"]["pool_mode"], "transaction")
        self.assertIn("postgres-pgbouncer.md", policy["pgbouncer"]["runbook"])
        self.assertGreaterEqual(policy["api"]["max_connections_per_api_process"], 1)
        self.assertGreaterEqual(policy["worker"]["max_connections_per_worker_process"], 1)

    def test_parallelism_policy_documents_concurrency_split(self) -> None:
        policy = parallelism.describe_parallelism_policy()
        concurrency = policy["concurrency_policy"]
        self.assertIn("asyncio", concurrency)
        self.assertIn("celery_prefork", concurrency)
        self.assertIn("gpu_workers", concurrency)
        self.assertIn("n_jobs=-1", concurrency["threads"])
        self.assertIn("DB_WORKER_POOL_SIZE", policy["environment_variables"])


if __name__ == "__main__":
    unittest.main()
