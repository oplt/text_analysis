"""Controlled Rscript runner with time, output, and cleanup limits."""

from __future__ import annotations

import hashlib
import logging
import os
import selectors
import shutil
import signal
import subprocess
import time
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

logger = logging.getLogger(__name__)

_MAX_STREAM_BYTES = 1 * 1024 * 1024  # 1 MiB per stream
_USER_ERROR_CODE = "R_ANALYSIS_FAILED"


def _entrypoint() -> Path:
    configured = Path(settings.RESEARCH_R_ENGINE_ENTRYPOINT)
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[4] / configured).resolve()


def r_runtime_available() -> bool:
    """True when *this process* can execute Rscript (worker-side check)."""
    return (
        bool(settings.RESEARCH_R_ENABLED)
        and shutil.which(settings.RESEARCH_RSCRIPT_PATH) is not None
        and _entrypoint().is_file()
    )


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _append_bounded(chunks: list[bytes], total: int, data: bytes, limit: int) -> int:
    if total >= limit or not data:
        return total
    keep = data[: max(0, limit - total)]
    if keep:
        chunks.append(keep)
    return total + len(keep)


def _communicate_bounded(
    proc: subprocess.Popen[bytes],
    *,
    timeout: float,
    limit: int = _MAX_STREAM_BYTES,
) -> tuple[bytes, bytes]:
    """Read stdout/stderr with a hard byte cap while waiting for the process.

    Continues draining the pipes after the cap so the child cannot block on a
    full OS pipe buffer. Raises :class:`RExecutionTimeout` and kills the whole
    process group when ``timeout`` elapses.
    """
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_total = 0
    stderr_total = 0
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        if proc.stdout is not None:
            sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
        if proc.stderr is not None:
            sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(proc)
                raise RExecutionTimeout("R analysis exceeded its time limit")
            for key, _ in sel.select(timeout=min(0.5, remaining)):
                chunk = key.fileobj.read(65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    stdout_total = _append_bounded(stdout_chunks, stdout_total, chunk, limit)
                else:
                    stderr_total = _append_bounded(stderr_chunks, stderr_total, chunk, limit)
    if proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(proc)
            raise RExecutionTimeout("R analysis exceeded its time limit")
        try:
            proc.wait(timeout=max(0.1, remaining))
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(proc)
            raise RExecutionTimeout("R analysis exceeded its time limit") from exc
    return b"".join(stdout_chunks), b"".join(stderr_chunks)


def _sanitize_stderr_for_logs(stderr: str) -> str:
    work = str(Path(settings.RESEARCH_R_WORK_DIR).expanduser())
    return stderr.replace(work, "<r_work_dir>")


def _persist_job_artifacts(bundle: RJobBundle, result: AnalysisResult) -> AnalysisResult:
    """Move durable R artifacts out of the ephemeral job directory before cleanup."""
    artifacts_dir = bundle.workdir / "artifacts"
    if not artifacts_dir.is_dir():
        return result
    durable_root = (
        Path(settings.RESEARCH_R_WORK_DIR).expanduser() / "artifacts" / bundle.workdir.name
    )
    durable_root.mkdir(parents=True, exist_ok=True)
    persisted: list[dict] = list(result.artifacts)
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifacts_dir)
        target = durable_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        persisted.append(
            {
                "kind": "r_artifact",
                "name": relative.as_posix(),
                "path": str(target),
                "sha256": digest,
                "bytes": target.stat().st_size,
            }
        )
    return result.model_copy(update={"artifacts": persisted})


def run_r_job(bundle: RJobBundle, *, expected_engine_version: str) -> AnalysisResult:
    """Execute the pinned entrypoint only; always remove per-job sensitive data."""
    proc: subprocess.Popen[bytes] | None = None
    try:
        if not r_runtime_available():
            raise RRuntimeUnavailable("R analysis runtime is not available")
        max_bytes = settings.RESEARCH_R_MAX_OUTPUT_MB * 1024 * 1024
        try:
            proc = subprocess.Popen(
                [
                    settings.RESEARCH_RSCRIPT_PATH,
                    "--vanilla",
                    str(_entrypoint()),
                    str(bundle.manifest_path),
                ],
                cwd=bundle.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise RRuntimeUnavailable("Failed to start Rscript") from exc

        try:
            _stdout, stderr_raw = _communicate_bounded(
                proc, timeout=float(settings.RESEARCH_R_TIMEOUT_SECONDS)
            )
        except RExecutionTimeout:
            raise
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        if stderr_text:
            logger.warning(
                "R stderr job=%s exit=%s detail=%s",
                bundle.workdir.name,
                proc.returncode,
                _sanitize_stderr_for_logs(stderr_text)[:2048],
            )
        if proc.returncode != 0:
            raise RExecutionFailed(
                f"{_USER_ERROR_CODE}: R analysis failed (exit code {proc.returncode})",
                run_id=bundle.workdir.name,
                exit_code=proc.returncode,
            )

        import json

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        expected_identity = {**manifest["identity"], "engine_version": expected_engine_version}
        result = validate_r_result(
            bundle.result_path,
            expected_analysis_type=manifest["analysis"]["type"],
            expected_identity=expected_identity,
            max_bytes=max_bytes,
        )
        return _persist_job_artifacts(bundle, result)
    finally:
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
        shutil.rmtree(bundle.workdir, ignore_errors=True)
