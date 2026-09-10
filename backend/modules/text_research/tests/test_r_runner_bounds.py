"""Unit tests for bounded R subprocess helpers and sanitized failures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.modules.text_research.domain.exceptions import (
    NON_RETRYABLE_RESEARCH_ERRORS,
    RExecutionFailed,
    RInvalidResult,
    RUnsupportedAnalysis,
)
from backend.modules.text_research.infrastructure.r_runtime import runner


def test_non_retryable_errors_include_deterministic_r_failures() -> None:
    assert RUnsupportedAnalysis in NON_RETRYABLE_RESEARCH_ERRORS
    assert RInvalidResult in NON_RETRYABLE_RESEARCH_ERRORS
    assert RExecutionFailed in NON_RETRYABLE_RESEARCH_ERRORS


def test_execution_failed_is_safe_for_clients() -> None:
    exc = RExecutionFailed("R_ANALYSIS_FAILED: R analysis failed (exit code 1)", run_id="job-1", exit_code=1)
    assert "R_ANALYSIS_FAILED" in str(exc)
    assert exc.run_id == "job-1"
    assert "/tmp" not in str(exc)


def test_communicate_bounded_caps_noisy_stderr(tmp_path: Path) -> None:
    import sys

    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('x' * (2 * 1024 * 1024))\n"
        "sys.stderr.flush()\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _stdout, stderr = runner._communicate_bounded(proc, timeout=10, limit=64 * 1024)
    assert proc.returncode == 0
    assert len(stderr) <= 64 * 1024


def test_save_unit_table_columnar_avoids_row_dicts(tmp_path: Path) -> None:
    from backend.modules.text_research.infrastructure.parquet_artifacts import save_unit_table

    target = tmp_path / "tokens.parquet"
    meta = save_unit_table(
        target,
        columns={
            "unit_id": ["u1", "u1", "u2"],
            "token_position": [0, 1, 0],
            "token": ["alpha", "beta", "gamma"],
        },
    )
    assert meta["row_count"] == 3
    assert Path(meta["path"]).is_file()
