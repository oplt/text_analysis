"""Artifact storage for trained vectorizers/classifiers/topic models.

When object storage (S3/MinIO) is configured, artifacts are uploaded there and
paths are stored as ``s3://{bucket}/{key}`` references with SHA-256 metadata.
Workers download and cache locally on demand. When storage is not configured,
artifacts remain on the local filesystem under RESEARCH_ARTIFACT_ROOT.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

logger = logging.getLogger(__name__)

S3_PREFIX = "s3://"


def _default_artifact_root() -> Path:
    env_override = os.environ.get("RESEARCH_ARTIFACT_ROOT")
    if env_override:
        return Path(env_override)

    try:
        from backend.core.config import settings

        configured = getattr(settings, "RESEARCH_ARTIFACT_ROOT", "") or ""
        if configured:
            return Path(configured)
    except Exception:
        pass

    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".research_artifacts"


ARTIFACT_ROOT: Path = _default_artifact_root()
_LOCAL_CACHE = ARTIFACT_ROOT / ".object_cache"


def ensure_artifact_dir(project_id: str, run_id: str) -> Path:
    directory = ARTIFACT_ROOT / str(project_id) / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_object_ref(path: str | Path) -> bool:
    return str(path).startswith(S3_PREFIX)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_joblib(obj: Any, path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, resolved)
    return resolved


def load_joblib(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact not found: {resolved}")
    return joblib.load(resolved)


def _storage_configured() -> bool:
    try:
        from backend.core.storage import object_storage

        return object_storage.is_configured
    except Exception:
        return False


def save_artifact_with_metadata(
    obj: Any,
    *,
    category: str,
    namespace: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Persist an artifact and return its stable reference plus audit metadata."""
    artifact_id = str(uuid4())
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_joblib(obj, tmp_path)
        payload = tmp_path.read_bytes()
        digest = _sha256_bytes(payload)
        filename = f"{artifact_id}-{digest[:12]}.joblib"

        if namespace:
            namespace_path = Path(namespace)
            if namespace_path.is_absolute() or ".." in namespace_path.parts:
                raise ValueError("Artifact namespace must be a relative path")
            relative_directory = namespace_path / category
        else:
            relative_directory = Path(category)

        if _storage_configured():
            from backend.core.config import settings
            from backend.core.storage import object_storage

            object_key = f"research-artifacts/{relative_directory.as_posix()}/{filename}"
            object_storage.upload_bytes_sync(
                object_key=object_key,
                body=payload,
                content_type="application/octet-stream",
                metadata={
                    "sha256": digest,
                    "size": str(len(payload)),
                    "model_type": category,
                    "serialization": "joblib",
                },
            )
            reference = f"{S3_PREFIX}{settings.STORAGE_BUCKET}/{object_key}"
            object_key_value: str | None = object_key
        else:
            directory = ARTIFACT_ROOT / relative_directory
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / filename
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as staged:
                staged.write(payload)
                staged_path = Path(staged.name)
            os.replace(staged_path, path)
            reference = str(path)
            object_key_value = None

        return reference, {
            "reference": reference,
            "object_key": object_key_value,
            "sha256": digest,
            "size": len(payload),
            "model_type": category,
            "serialization_format": "joblib",
            "serialization_version": getattr(joblib, "__version__", None),
            "created_at": datetime.now(UTC).isoformat(),
        }
    finally:
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)


def save_artifact(obj: Any, *, category: str, namespace: str | None = None) -> str:
    """Backward-compatible artifact save API returning only its reference."""
    reference, _metadata = save_artifact_with_metadata(
        obj,
        category=category,
        namespace=namespace,
    )
    return reference


def _resolve_object_ref(path: str) -> Path:
    from backend.core.config import settings
    from backend.core.storage import object_storage

    raw = path[len(S3_PREFIX) :]
    bucket, _, key = raw.partition("/")
    if not key:
        raise FileNotFoundError(f"Invalid object artifact reference: {path}")
    cache_path = _LOCAL_CACHE / (bucket or settings.STORAGE_BUCKET) / key
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    body = object_storage.download_bytes_sync(key, bucket=bucket or None)
    cache_path.write_bytes(body)
    return cache_path


def load_artifact(path: str | Path) -> Any:
    text = str(path)
    if is_object_ref(text):
        return load_joblib(_resolve_object_ref(text))
    return load_joblib(path)


def delete_artifact(path: str | Path) -> None:
    text = str(path)
    if is_object_ref(text):
        try:
            from backend.core.config import settings
            from backend.core.storage import object_storage

            raw = text[len(S3_PREFIX) :]
            bucket, _, key = raw.partition("/")
            object_storage.delete_object_sync(key, bucket=bucket or settings.STORAGE_BUCKET)
        except Exception as exc:
            logger.warning("failed to delete object artifact %s: %s", text, exc)
        return
    resolved = Path(path)
    if resolved.exists():
        resolved.unlink()
