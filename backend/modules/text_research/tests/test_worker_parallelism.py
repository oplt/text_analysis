"""Tests for Celery nested-parallelism controls (§22)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from backend.workers import parallelism


class WorkerParallelismTests(unittest.TestCase):
    def setUp(self) -> None:
        parallelism.reset_parallelism_for_tests()
        self._env_backup = {
            name: os.environ.get(name)
            for name in (
                *parallelism._BLAS_THREAD_ENV_VARS,
                *parallelism._JOBLIB_ENV_VARS,
            )
        }

    def tearDown(self) -> None:
        parallelism.reset_parallelism_for_tests()
        for name, value in self._env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_configure_sets_blas_and_joblib_env(self) -> None:
        with mock.patch.object(
            parallelism,
            "_settings_values",
            return_value={
                "enabled": True,
                "blas_threads": 1,
                "sklearn_n_jobs": 1,
                "joblib_n_jobs": 1,
            },
        ):
            report = parallelism.configure_worker_parallelism(force=True, overwrite_env=True)

        self.assertTrue(report["applied"])
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("LOKY_MAX_CPU_COUNT"), "1")

    def test_configure_respects_disabled_flag(self) -> None:
        with mock.patch.object(
            parallelism,
            "_settings_values",
            return_value={
                "enabled": False,
                "blas_threads": 1,
                "sklearn_n_jobs": 1,
                "joblib_n_jobs": 1,
            },
        ):
            report = parallelism.configure_worker_parallelism(force=True)

        self.assertFalse(report["applied"])
        self.assertEqual(report["skipped_reason"], "RESEARCH_APPLY_THREAD_LIMITS=false")

    def test_sklearn_n_jobs_clamps_all_cores_request(self) -> None:
        with mock.patch.object(
            parallelism,
            "_settings_values",
            return_value={
                "enabled": True,
                "blas_threads": 1,
                "sklearn_n_jobs": 1,
                "joblib_n_jobs": 1,
            },
        ):
            self.assertEqual(parallelism.sklearn_n_jobs(None), 1)
            self.assertEqual(parallelism.sklearn_n_jobs(-1), 1)
            self.assertEqual(parallelism.sklearn_n_jobs(8), 1)

        with mock.patch.object(
            parallelism,
            "_settings_values",
            return_value={
                "enabled": True,
                "blas_threads": 2,
                "sklearn_n_jobs": 2,
                "joblib_n_jobs": 2,
            },
        ):
            self.assertEqual(parallelism.sklearn_n_jobs(8), 2)
            self.assertEqual(parallelism.joblib_n_jobs(-1), 2)

    def test_describe_policy_includes_env_documentation(self) -> None:
        policy = parallelism.describe_parallelism_policy()
        self.assertIn("RESEARCH_WORKER_BLAS_THREADS", policy["environment_variables"])
        self.assertIn("OMP_NUM_THREADS", policy["blas_env_vars"])
        self.assertEqual(policy["recommended_cpu_worker"]["RESEARCH_SKLEARN_N_JOBS"], 1)
        self.assertIn("celery_prefork", policy["concurrency_policy"])

    def test_logging_hooks_register_process_init(self) -> None:
        from backend.workers import logging_hooks

        self.assertTrue(callable(logging_hooks.configure_worker_process))
        self.assertTrue(callable(logging_hooks.configure_worker_logging))
