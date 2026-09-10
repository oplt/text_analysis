"""Controlled Rscript runner with time, output, and cleanup limits."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from backend.core.config import settings
from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.domain.exceptions import (
    RExecutionFailed,
    RExecutionTimeout,
    RRuntimeUnavailable,
)
from backend.modules.text_research.infrastructure.r_runtime.serializer import RJobBundle
from backend.modules.text_research.infrastructure.r_runtime.validator import validate_r_result


def _entrypoint() -> Path:
    configured = Path(settings.RESEARCH_R_ENGINE_ENTRYPOINT)
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[4] / configured).resolve()


def r_runtime_available() -> bool:
    return (
        bool(settings.RESEARCH_R_ENABLED)
        and shutil.which(settings.RESEARCH_RSCRIPT_PATH) is not None
        and _entrypoint().is_file()
    )


def run_r_job(bundle: RJobBundle, *, expected_engine_version: str) -> AnalysisResult:
    """Execute the pinned entrypoint only; always remove per-job sensitive data."""
    try:
        if not r_runtime_available():
            raise RRuntimeUnavailable("R analysis runtime is not available")
        max_bytes = settings.RESEARCH_R_MAX_OUTPUT_MB * 1024 * 1024
        try:
            completed = subprocess.run(
                [
                    settings.RESEARCH_RSCRIPT_PATH,
                    "--vanilla",
                    str(_entrypoint()),
                    str(bundle.manifest_path),
                ],
                cwd=bundle.workdir,
                capture_output=True,
                text=True,
                timeout=settings.RESEARCH_R_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RExecutionTimeout("R analysis exceeded its time limit") from exc
        stderr = (completed.stderr or "")[:8192]
        if completed.returncode != 0:
            raise RExecutionFailed(
                f"R analysis failed (exit code {completed.returncode}): {stderr}"
            )
        import json

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        expected_identity = {**manifest["identity"], "engine_version": expected_engine_version}
        return validate_r_result(
            bundle.result_path,
            expected_analysis_type=manifest["analysis"]["type"],
            expected_identity=expected_identity,
            max_bytes=max_bytes,
        )
    finally:
        shutil.rmtree(bundle.workdir, ignore_errors=True)
