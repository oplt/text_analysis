"""Retry, timeout, checkpoint, and idempotency policies for research execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.modules.text_research.domain.execution_defaults import (
    ENGINE_VERSION,
    computation_identity,
    retry_policy_for,
    timeout_for,
)

__all__ = [
    "Checkpoint",
    "computation_identity",
    "ENGINE_VERSION",
    "is_idempotent_hit",
    "load_checkpoint",
    "retry_policy_for",
    "save_checkpoint",
    "timeout_for",
]


@dataclass
class Checkpoint:
    stage: str
    progress: float
    payload: dict[str, Any]
    updated_at: str

    @classmethod
    def now(
        cls, *, stage: str, progress: float, payload: dict[str, Any] | None = None
    ) -> Checkpoint:
        return cls(
            stage=stage,
            progress=progress,
            payload=payload or {},
            updated_at=datetime.now(tz=UTC).isoformat(),
        )


def _checkpoint_root(root: Path | None) -> Path:
    if root is not None:
        return root
    configured = settings.RESEARCH_ARTIFACT_ROOT.strip()
    if configured:
        return Path(configured)
    return Path(settings.RESEARCH_ARTIFACT_DIR)


def save_checkpoint(run_id: str, checkpoint: Checkpoint, root: Path | None = None) -> Path:
    """Persist a run checkpoint under ``checkpoints/{run_id}.json``."""
    base = _checkpoint_root(root) / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{run_id}.json"
    path.write_text(json.dumps(asdict(checkpoint), sort_keys=True), encoding="utf-8")
    return path


def load_checkpoint(run_id: str, root: Path | None = None) -> Checkpoint | None:
    """Load a previously saved checkpoint, or ``None`` when absent."""
    path = _checkpoint_root(root) / "checkpoints" / f"{run_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Checkpoint(**payload)


def is_idempotent_hit(
    *,
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str = ENGINE_VERSION,
    engine_name: str = "python",
    pipeline_checksum: str | None = None,
    stage_cache_get: Callable[[str], Any] | None = None,
) -> str | None:
    """Return computation identity when an identical cached result exists."""
    identity = computation_identity(
        spec_hash,
        corpus_snapshot_hash,
        engine_version=engine_version,
        engine_name=engine_name,
        pipeline_checksum=pipeline_checksum,
    )
    getter = stage_cache_get
    if getter is None:
        from backend.modules.text_research.infrastructure.stage_cache import get_stage

        getter = get_stage
    hit = getter(identity)
    if hit is None or hit is False:
        return None
    if isinstance(hit, str):
        return hit
    return identity
