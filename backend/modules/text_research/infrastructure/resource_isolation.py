"""Effective worker resource isolation for provenance (cgroup + operator config).

Hard limits for the dedicated R worker are applied at the container/cgroup layer
(Compose ``mem_limit`` / ``cpus`` / ``pids_limit``, or equivalent K8s limits).
Per-analysis ``ExecutionSpec.cpu`` / ``memory_mb`` are advisory only.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any

_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_MEMORY = Path("/sys/fs/cgroup/memory")
_CGROUP_V1_CPU = Path("/sys/fs/cgroup/cpu")
_CGROUP_V1_PIDS = Path("/sys/fs/cgroup/pids")


def _read_text(path: Path) -> str | None:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def _parse_unlimited(raw: str | None) -> int | None:
    if raw is None or raw == "" or raw.lower() == "max":
        return None
    try:
        value = int(raw.split()[0])
    except (TypeError, ValueError):
        return None
    # cgroup v1 often uses near-2^63 as "unlimited"
    if value >= (1 << 62):
        return None
    return value


def _cgroup_v2_limits() -> dict[str, Any]:
    memory_raw = _read_text(_CGROUP_V2_ROOT / "memory.max")
    cpu_raw = _read_text(_CGROUP_V2_ROOT / "cpu.max")
    pids_raw = _read_text(_CGROUP_V2_ROOT / "pids.max")
    if memory_raw is None and cpu_raw is None and pids_raw is None:
        return {}

    out: dict[str, Any] = {"cgroup_version": 2}
    memory_bytes = _parse_unlimited(memory_raw)
    if memory_bytes is not None:
        out["memory_bytes"] = memory_bytes
        out["memory_mb"] = memory_bytes // (1024 * 1024)

    if cpu_raw and cpu_raw != "max":
        parts = cpu_raw.split()
        if len(parts) >= 2:
            try:
                quota = int(parts[0])
                period = int(parts[1])
            except ValueError:
                quota = period = 0
            if quota > 0 and period > 0:
                out["cpu_quota_us"] = quota
                out["cpu_period_us"] = period
                out["cpu_cores"] = round(quota / period, 4)

    pids_max = _parse_unlimited(pids_raw)
    if pids_max is not None:
        out["pids_max"] = pids_max
    return out


def _cgroup_v1_limits() -> dict[str, Any]:
    memory_raw = _read_text(_CGROUP_V1_MEMORY / "memory.limit_in_bytes")
    quota_raw = _read_text(_CGROUP_V1_CPU / "cpu.cfs_quota_us")
    period_raw = _read_text(_CGROUP_V1_CPU / "cpu.cfs_period_us")
    pids_raw = _read_text(_CGROUP_V1_PIDS / "pids.max")
    if memory_raw is None and quota_raw is None and pids_raw is None:
        return {}

    out: dict[str, Any] = {"cgroup_version": 1}
    memory_bytes = _parse_unlimited(memory_raw)
    if memory_bytes is not None:
        out["memory_bytes"] = memory_bytes
        out["memory_mb"] = memory_bytes // (1024 * 1024)

    try:
        quota = int(quota_raw) if quota_raw is not None else -1
        period = int(period_raw) if period_raw is not None else 0
    except ValueError:
        quota, period = -1, 0
    if quota > 0 and period > 0:
        out["cpu_quota_us"] = quota
        out["cpu_period_us"] = period
        out["cpu_cores"] = round(quota / period, 4)

    pids_max = _parse_unlimited(pids_raw)
    if pids_max is not None:
        out["pids_max"] = pids_max
    return out


def read_cgroup_limits(*, root: Path | None = None) -> dict[str, Any]:
    """Best-effort read of effective cgroup memory/CPU/PID limits.

    ``root`` is only for tests (cgroup v2 layout under a temp directory).
    """
    if root is not None:
        memory_raw = _read_text(root / "memory.max")
        cpu_raw = _read_text(root / "cpu.max")
        pids_raw = _read_text(root / "pids.max")
        out: dict[str, Any] = {"cgroup_version": 2}
        memory_bytes = _parse_unlimited(memory_raw)
        if memory_bytes is not None:
            out["memory_bytes"] = memory_bytes
            out["memory_mb"] = memory_bytes // (1024 * 1024)
        if cpu_raw and cpu_raw != "max":
            parts = cpu_raw.split()
            if len(parts) >= 2:
                try:
                    quota = int(parts[0])
                    period = int(parts[1])
                except ValueError:
                    quota = period = 0
                if quota > 0 and period > 0:
                    out["cpu_cores"] = round(quota / period, 4)
                    out["cpu_quota_us"] = quota
                    out["cpu_period_us"] = period
        pids_max = _parse_unlimited(pids_raw)
        if pids_max is not None:
            out["pids_max"] = pids_max
        return out

    v2 = _cgroup_v2_limits()
    if v2:
        return v2
    return _cgroup_v1_limits()


def _configured_worker_limits() -> dict[str, Any]:
    try:
        from backend.core.config import settings

        memory = (settings.RESEARCH_R_WORKER_MEMORY_LIMIT or "").strip()
        cpus = (settings.RESEARCH_R_WORKER_CPUS or "").strip()
        pids = settings.RESEARCH_R_WORKER_PIDS_LIMIT
        concurrency = int(settings.CELERY_R_CONCURRENCY)
        timeout = int(settings.RESEARCH_R_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001
        memory = (os.environ.get("RESEARCH_R_WORKER_MEMORY_LIMIT") or "").strip()
        cpus = (os.environ.get("RESEARCH_R_WORKER_CPUS") or "").strip()
        pids_raw = os.environ.get("RESEARCH_R_WORKER_PIDS_LIMIT")
        try:
            pids = int(pids_raw) if pids_raw else None
        except ValueError:
            pids = None
        try:
            concurrency = int(os.environ.get("CELERY_R_CONCURRENCY") or "1")
        except ValueError:
            concurrency = 1
        try:
            timeout = int(os.environ.get("RESEARCH_R_TIMEOUT_SECONDS") or "600")
        except ValueError:
            timeout = 600

    out: dict[str, Any] = {
        "celery_concurrency": concurrency,
        "timeout_seconds": timeout,
    }
    if memory:
        out["configured_memory_limit"] = memory
    if cpus:
        out["configured_cpu_limit"] = cpus
        with suppress(ValueError):
            out["configured_cpu_cores"] = float(cpus)
    if pids is not None and pids > 0:
        out["configured_pids_limit"] = pids
    return out


def effective_resource_limits(*, cgroup_root: Path | None = None) -> dict[str, Any]:
    """Merge cgroup observations with operator-configured R-worker limits.

    Always records that ``ExecutionSpec`` fields are advisory and that
    enforceable isolation is container/cgroup based.
    """
    cgroup = read_cgroup_limits(root=cgroup_root)
    configured = _configured_worker_limits()
    payload: dict[str, Any] = {
        "enforcement": "container_cgroup",
        "execution_spec_policy": "advisory",
        "note": (
            "ExecutionSpec.cpu / memory_mb are advisory hints only; "
            "hard limits come from the worker container cgroup and Celery concurrency."
        ),
        **configured,
    }
    if cgroup:
        payload["cgroup"] = cgroup
        if "memory_bytes" in cgroup:
            payload["memory_bytes"] = cgroup["memory_bytes"]
            payload["memory_mb"] = cgroup["memory_mb"]
        if "cpu_cores" in cgroup:
            payload["cpu_cores"] = cgroup["cpu_cores"]
        if "pids_max" in cgroup:
            payload["pids_max"] = cgroup["pids_max"]
    return payload
