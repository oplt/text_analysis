"""Disk-backed content-addressable cache for pipeline stage outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.modules.text_research.infrastructure import artifact_registry
from backend.modules.text_research.infrastructure.pipeline_compiler import computation_identity

PayloadFormat = str  # "json" | "joblib" | "npz"


def stage_cache_key(
    *,
    engine_version: str,
    stage_name: str,
    input_checksum: str,
    spec_hash: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Stable sha256 key over stage identity inputs."""
    payload = json.dumps(
        {
            "engine_version": engine_version,
            "stage_name": stage_name,
            "input_checksum": input_checksum,
            "spec_hash": spec_hash,
            "params": params or {},
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_identity_lookup(
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str,
) -> str:
    """Wrap :func:`computation_identity` for prepared-corpus stage lookups."""
    return computation_identity(spec_hash, corpus_snapshot_hash, engine_version)


def _stage_cache_root() -> Path:
    env_override = os.environ.get("RESEARCH_ARTIFACT_DIR")
    if env_override:
        return Path(env_override) / "stage_cache"

    try:
        from backend.core.config import settings

        configured = getattr(settings, "RESEARCH_ARTIFACT_DIR", "") or ""
        if configured:
            return Path(configured) / "stage_cache"
    except Exception:
        pass

    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / "var" / "research_artifacts" / "stage_cache"


def _entry_dir(key: str) -> Path:
    return _stage_cache_root() / key


def _meta_path(key: str) -> Path:
    return _entry_dir(key) / "meta.json"


def _payload_extension(payload_format: str) -> str:
    if payload_format == "json":
        return ".json"
    if payload_format == "joblib":
        return ".joblib"
    if payload_format == "npz":
        return ".npz"
    raise ValueError(f"Unsupported payload_format {payload_format!r}")


def has_stage(key: str) -> bool:
    return _meta_path(key).is_file()


def get_stage(key: str) -> dict[str, Any] | None:
    """Load stage sidecar metadata; include ``payload`` when present on disk."""
    meta_file = _meta_path(key)
    if not meta_file.is_file():
        return None

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    payload_path = meta.get("payload_path")
    if payload_path:
        resolved = Path(payload_path)
        if resolved.is_file():
            payload_format = meta.get("payload_format", "json")
            if payload_format == "json":
                meta["payload"] = json.loads(resolved.read_text(encoding="utf-8"))
            elif payload_format == "joblib":
                meta["payload"] = joblib.load(resolved)
            elif payload_format == "npz":
                loaded = np.load(resolved, allow_pickle=False)
                meta["payload"] = {name: loaded[name] for name in loaded.files}
    return meta


def put_stage(
    key: str,
    *,
    meta: dict[str, Any],
    payload: Any | None = None,
    payload_format: PayloadFormat = "json",
) -> str:
    """Persist stage metadata (and optional payload); returns artifact directory path."""
    if has_stage(key):
        return str(_entry_dir(key))

    entry = _entry_dir(key)
    entry.mkdir(parents=True, exist_ok=True)

    sidecar = dict(meta)
    sidecar["cache_key"] = key
    sidecar.setdefault("created_at", datetime.now(UTC).isoformat())

    artifact_path = str(entry)
    payload_path: Path | None = None
    if payload is not None:
        payload_path = entry / f"payload{_payload_extension(payload_format)}"
        if payload_format == "json":
            payload_path.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str),
                encoding="utf-8",
            )
        elif payload_format == "joblib":
            joblib.dump(payload, payload_path)
        elif payload_format == "npz":
            if isinstance(payload, dict):
                np.savez_compressed(payload_path, **payload)
            else:
                np.savez_compressed(payload_path, data=payload)

    if payload_path is not None:
        sidecar["payload_path"] = str(payload_path)
        sidecar["payload_format"] = payload_format

    _meta_path(key).write_text(
        json.dumps(sidecar, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )

    checksum = key
    artifact_registry.register(
        "stage",
        key,
        {
            "artifact_path": artifact_path,
            "stage_name": sidecar.get("stage_name"),
            "payload_format": sidecar.get("payload_format"),
        },
        checksum,
    )
    return artifact_path


def invalidate(key: str | None = None) -> None:
    """Remove one cached stage or clear the entire stage cache."""
    root = _stage_cache_root()
    if key is None:
        if root.exists():
            shutil.rmtree(root)
        return

    entry = _entry_dir(key)
    if entry.exists():
        shutil.rmtree(entry)


def remember_computation(
    *,
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str,
    meta: dict[str, Any],
    payload: Any | None = None,
    payload_format: PayloadFormat = "json",
) -> str:
    """Store a full-analysis result under its computation identity key."""
    key = compute_identity_lookup(spec_hash, corpus_snapshot_hash, engine_version)
    sidecar = dict(meta)
    sidecar.setdefault("stage_name", "computation")
    sidecar["spec_hash"] = spec_hash
    sidecar["corpus_snapshot_hash"] = corpus_snapshot_hash
    sidecar["engine_version"] = engine_version
    return put_stage(key, meta=sidecar, payload=payload, payload_format=payload_format)
