"""Artifact storage for trained vectorizers/classifiers/topic models.

When object storage (S3/MinIO) is configured, artifacts are uploaded there and
paths are stored as ``s3://{bucket}/{key}`` references with SHA-256 metadata.
Workers download and cache locally on demand. When storage is not configured,
artifacts remain on the local filesystem under RESEARCH_ARTIFACT_ROOT.

Run-scoped lifecycle (LATEST-022):
1. Stage under ``{artifact_namespace}/tmp/{category}/…``
2. Re-check cancellation
3. Promote into ``{artifact_namespace}/artifacts/{category}/…`` only on success
4. Delete the ``tmp`` tree on failure/cancel (never touch other runs' published
   artifacts)
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

logger = logging.getLogger(__name__)

S3_PREFIX = "s3://"
TMP_SEGMENT = "tmp"
PUBLISHED_SEGMENT = "artifacts"


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


def default_artifact_namespace(project_id: str, run_id: str) -> str:
    """Canonical relative namespace for a research analysis run."""
    return f"runs/{project_id}/{run_id}"


def staging_namespace(artifact_namespace: str) -> str:
    """Relative namespace used while artifacts are not yet published."""
    return f"{_validate_relative_namespace(artifact_namespace)}/{TMP_SEGMENT}"


def published_namespace(artifact_namespace: str) -> str:
    """Relative namespace for immutable artifacts after successful completion."""
    return f"{_validate_relative_namespace(artifact_namespace)}/{PUBLISHED_SEGMENT}"


def is_object_ref(path: str | Path) -> bool:
    return str(path).startswith(S3_PREFIX)


def _validate_relative_namespace(namespace: str) -> str:
    namespace_path = Path(namespace)
    if namespace_path.is_absolute() or ".." in namespace_path.parts or not namespace.strip():
        raise ValueError("Artifact namespace must be a non-empty relative path")
    return namespace_path.as_posix().strip("/")


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
            relative_directory = Path(_validate_relative_namespace(namespace)) / category
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
            "staged": TMP_SEGMENT in relative_directory.parts,
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


def stage_artifact_with_metadata(
    obj: Any,
    *,
    category: str,
    artifact_namespace: str,
) -> tuple[str, dict[str, Any]]:
    """Write an artifact into the run-scoped temporary namespace."""
    reference, metadata = save_artifact_with_metadata(
        obj,
        category=category,
        namespace=staging_namespace(artifact_namespace),
    )
    metadata["staged"] = True
    metadata["artifact_namespace"] = _validate_relative_namespace(artifact_namespace)
    return reference, metadata


def _local_path_from_reference(reference: str) -> Path | None:
    if is_object_ref(reference):
        return None
    return Path(reference)


def _object_key_from_reference(reference: str) -> tuple[str | None, str]:
    raw = reference[len(S3_PREFIX) :]
    bucket, _, key = raw.partition("/")
    if not key:
        raise FileNotFoundError(f"Invalid object artifact reference: {reference}")
    return bucket or None, key


def promote_staged_artifact(
    staged_reference: str,
    *,
    category: str,
    artifact_namespace: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Move a staged artifact into the immutable published run namespace.

    Published artifacts intentionally live outside ``tmp/`` so cancel/fail
    cleanup can wipe staging without deleting successfully published outputs
    from completed reusable runs.
    """
    meta = dict(metadata or {})
    digest = meta.get("sha256")
    filename: str | None = None
    published_ns = published_namespace(artifact_namespace)
    relative_directory = Path(published_ns) / category

    if is_object_ref(staged_reference):
        from backend.core.config import settings
        from backend.core.storage import object_storage

        bucket, source_key = _object_key_from_reference(staged_reference)
        filename = Path(source_key).name
        dest_key = f"research-artifacts/{relative_directory.as_posix()}/{filename}"
        if source_key != dest_key:
            body = object_storage.download_bytes_sync(source_key, bucket=bucket)
            if not digest:
                digest = _sha256_bytes(body)
            object_storage.upload_bytes_sync(
                object_key=dest_key,
                body=body,
                content_type="application/octet-stream",
                metadata={
                    "sha256": str(digest),
                    "size": str(len(body)),
                    "model_type": category,
                    "serialization": "joblib",
                },
            )
            with contextlib.suppress(Exception):
                object_storage.delete_object_sync(
                    source_key, bucket=bucket or settings.STORAGE_BUCKET
                )
        published_reference = f"{S3_PREFIX}{settings.STORAGE_BUCKET}/{dest_key}"
        meta.update(
            {
                "reference": published_reference,
                "object_key": dest_key,
                "sha256": digest,
                "staged": False,
                "published": True,
                "artifact_namespace": _validate_relative_namespace(artifact_namespace),
            }
        )
        return published_reference, meta

    source = Path(staged_reference)
    if not source.exists():
        raise FileNotFoundError(f"Staged artifact not found: {staged_reference}")
    filename = source.name
    if not digest:
        digest = _sha256_bytes(source.read_bytes())
    directory = ARTIFACT_ROOT / relative_directory
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if source.resolve() != destination.resolve():
        os.replace(source, destination)
    published_reference = str(destination)
    meta.update(
        {
            "reference": published_reference,
            "object_key": None,
            "sha256": digest,
            "size": meta.get("size") or destination.stat().st_size,
            "staged": False,
            "published": True,
            "artifact_namespace": _validate_relative_namespace(artifact_namespace),
        }
    )
    return published_reference, meta


def cleanup_run_tmp(artifact_namespace: str | None) -> None:
    """Delete only the run-scoped temporary staging tree."""
    if not artifact_namespace:
        return
    relative = staging_namespace(artifact_namespace)
    _delete_namespace_tree(relative)


def cleanup_run_namespace(
    artifact_namespace: str | None,
    *,
    include_published: bool = False,
) -> None:
    """Delete staging (and optionally published) artifacts for an unfinished run.

    Completed reusable runs must pass ``include_published=False`` (default) so
    only ``tmp/`` is removed. Cancel/fail before successful completion may set
    ``include_published=True`` to remove unpublished promotions.
    """
    if not artifact_namespace:
        return
    cleanup_run_tmp(artifact_namespace)
    if include_published:
        _delete_namespace_tree(published_namespace(artifact_namespace))


def _delete_namespace_tree(relative_namespace: str) -> None:
    relative = _validate_relative_namespace(relative_namespace)
    local_dir = ARTIFACT_ROOT / relative
    if local_dir.is_dir():
        shutil.rmtree(local_dir, ignore_errors=True)

    if not _storage_configured():
        return
    try:
        from backend.core.config import settings
        from backend.core.storage import object_storage

        prefix = f"research-artifacts/{relative}/"
        # Best-effort prefix cleanup when the storage backend supports listing.
        list_fn = getattr(object_storage, "list_object_keys_sync", None)
        delete_fn = getattr(object_storage, "delete_object_sync", None)
        if callable(list_fn) and callable(delete_fn):
            for key in list_fn(prefix=prefix, bucket=settings.STORAGE_BUCKET) or []:
                with contextlib.suppress(Exception):
                    delete_fn(key, bucket=settings.STORAGE_BUCKET)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to cleanup object namespace %s: %s", relative, exc)


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
