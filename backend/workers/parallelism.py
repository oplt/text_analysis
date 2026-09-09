"""Prevent nested CPU oversubscription in Celery research workers.

Celery prefork workers already multiply processes. Allowing each process to also
spin sklearn/joblib workers and OpenMP/MKL/OpenBLAS threads causes
``concurrency × n_jobs × BLAS_threads`` contention.

Defaults keep **intra-op parallelism at 1** inside each worker process so
horizontal scale comes from Celery concurrency / queue layout, not nested
threads. Override via settings / environment variables when a machine is
dedicated to a single heavy job.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CONFIGURED = False

# Env vars that control native math / OpenMP thread pools.
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# Hint loky (joblib process backend) not to claim all CPUs.
_JOBLIB_ENV_VARS = ("LOKY_MAX_CPU_COUNT",)


def _settings_values() -> dict[str, Any]:
    try:
        from backend.core.config import settings

        return {
            "enabled": bool(getattr(settings, "RESEARCH_APPLY_THREAD_LIMITS", True)),
            "blas_threads": max(1, int(getattr(settings, "RESEARCH_WORKER_BLAS_THREADS", 1))),
            "sklearn_n_jobs": int(getattr(settings, "RESEARCH_SKLEARN_N_JOBS", 1)),
            "joblib_n_jobs": int(getattr(settings, "RESEARCH_JOBLIB_N_JOBS", 1)),
        }
    except Exception:
        return {
            "enabled": True,
            "blas_threads": 1,
            "sklearn_n_jobs": 1,
            "joblib_n_jobs": 1,
        }


def sklearn_n_jobs(requested: int | None = None) -> int:
    """Resolve sklearn ``n_jobs`` under worker caps.

    ``None`` / ``0`` → configured default. Negative values (e.g. ``-1``) are
    clamped to the configured positive cap so callers cannot accidentally
    request “all CPUs” inside a Celery worker.
    """
    configured = _settings_values()["sklearn_n_jobs"]
    if configured < 1:
        configured = 1
    if requested is None or requested == 0:
        return configured
    if requested < 0:
        return configured
    return min(int(requested), configured) if configured > 0 else int(requested)


def joblib_n_jobs(requested: int | None = None) -> int:
    """Same clamping policy for joblib ``n_jobs``."""
    configured = _settings_values()["joblib_n_jobs"]
    if configured < 1:
        configured = 1
    if requested is None or requested == 0:
        return configured
    if requested < 0:
        return configured
    return min(int(requested), configured)


def _set_env_int(name: str, value: int, *, overwrite: bool) -> None:
    if overwrite or name not in os.environ:
        os.environ[name] = str(value)


def configure_worker_parallelism(
    *, force: bool = False, overwrite_env: bool = True
) -> dict[str, Any]:
    """Apply BLAS / OpenMP / joblib / sklearn thread caps in this process.

    Safe to call multiple times; subsequent calls are no-ops unless ``force``.
    Prefer invoking from Celery ``worker_process_init`` so forked children
    inherit limits after pool spawn.
    """
    global _CONFIGURED
    values = _settings_values()
    report: dict[str, Any] = {
        "applied": False,
        "enabled": values["enabled"],
        "blas_threads": values["blas_threads"],
        "sklearn_n_jobs": values["sklearn_n_jobs"],
        "joblib_n_jobs": values["joblib_n_jobs"],
        "env": {},
        "threadpoolctl": None,
        "joblib_backend": None,
    }

    if not values["enabled"]:
        report["skipped_reason"] = "RESEARCH_APPLY_THREAD_LIMITS=false"
        return report

    if _CONFIGURED and not force:
        report["skipped_reason"] = "already_configured"
        report["applied"] = True
        return report

    blas_threads = values["blas_threads"]
    for name in _BLAS_THREAD_ENV_VARS:
        _set_env_int(name, blas_threads, overwrite=overwrite_env)
        report["env"][name] = os.environ.get(name)

    joblib_cap = max(1, values["joblib_n_jobs"])
    for name in _JOBLIB_ENV_VARS:
        _set_env_int(name, joblib_cap, overwrite=overwrite_env)
        report["env"][name] = os.environ.get(name)

    try:
        import joblib

        report["joblib_version"] = getattr(joblib, "__version__", None)
        report["joblib_backend"] = "env_capped"
    except Exception as exc:  # pragma: no cover
        report["joblib_backend"] = f"unavailable:{exc}"

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=blas_threads)
        report["threadpoolctl"] = {"limits": blas_threads}
    except Exception as exc:
        report["threadpoolctl"] = {"error": str(exc)}

    _CONFIGURED = True
    report["applied"] = True
    logger.info(
        "worker_parallelism applied blas_threads=%s sklearn_n_jobs=%s joblib_n_jobs=%s",
        blas_threads,
        values["sklearn_n_jobs"],
        values["joblib_n_jobs"],
    )
    return report


def describe_parallelism_policy() -> dict[str, Any]:
    """Documented policy snapshot for runbooks / health checks."""
    values = _settings_values()
    return {
        "goal": (
            "Avoid Celery worker processes × sklearn/joblib n_jobs × BLAS/OpenMP "
            "threads oversubscription."
        ),
        "defaults": values,
        "concurrency_policy": {
            "asyncio": [
                "PostgreSQL (async SQLAlchemy / asyncpg)",
                "Redis",
                "HTTP / FastAPI request handlers",
                "network and object-storage I/O",
            ],
            "celery_prefork": [
                "sklearn estimators",
                "topic models",
                "CPU-heavy NLP",
                "statistics / quantitative pipelines",
            ],
            "gpu_workers": [
                "sentence-transformers",
                "large embedding models",
            ],
            "threads": (
                "Only where native libraries release the GIL or I/O requires them. "
                "Do not set sklearn/joblib n_jobs=-1 inside Celery workers."
            ),
            "blas_and_joblib": "Capped via configure_worker_parallelism (default 1).",
        },
        "environment_variables": {
            "RESEARCH_APPLY_THREAD_LIMITS": "Master switch (default true)",
            "RESEARCH_WORKER_BLAS_THREADS": (
                "Caps OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, "
                "NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS, BLIS_NUM_THREADS"
            ),
            "RESEARCH_SKLEARN_N_JOBS": "Default/clamp for sklearn estimator n_jobs",
            "RESEARCH_JOBLIB_N_JOBS": "Default/clamp for joblib + LOKY_MAX_CPU_COUNT",
            "CELERY_CONCURRENCY": "Process count (Procfile.dev / celery --concurrency)",
            "DB_WORKER_POOL_SIZE": "Per-process SQLAlchemy pool (see db.session)",
            "DB_WORKER_MAX_OVERFLOW": "Per-process SQLAlchemy overflow",
        },
        "recommended_cpu_worker": {
            "CELERY_CONCURRENCY": "match physical cores or slightly below",
            "RESEARCH_WORKER_BLAS_THREADS": 1,
            "RESEARCH_SKLEARN_N_JOBS": 1,
            "RESEARCH_JOBLIB_N_JOBS": 1,
            "DB_WORKER_POOL_SIZE": 2,
            "DB_WORKER_MAX_OVERFLOW": 2,
            "note": (
                "Prefer more Celery processes with 1 thread each over one process "
                "with nested all-core parallelism when running research_cpu queues."
            ),
        },
        "blas_env_vars": list(_BLAS_THREAD_ENV_VARS),
        "configured_in_process": _CONFIGURED,
        "current_env": {name: os.environ.get(name) for name in _BLAS_THREAD_ENV_VARS},
    }


def reset_parallelism_for_tests() -> None:
    """Test helper to allow re-applying configuration."""
    global _CONFIGURED
    _CONFIGURED = False
