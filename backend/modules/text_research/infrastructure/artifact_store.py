"""Typed, content-addressable store for research analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from backend.modules.text_research.infrastructure import artifact_registry, stage_cache

ArtifactType = Literal[
    "prepared_corpus",
    "dfm",
    "embeddings",
    "model",
    "vectorizer",
    "topic_model",
    "topic_matrix",
    "matrix",
    "export",
    "manifest",
    "r_artifact",
]


@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_id: str
    artifact_type: ArtifactType
    checksum: str
    implementation_version: str
    parent_artifact_ids: tuple[str, ...] = ()
    producing_run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


def _artifact_root() -> Path:
    import os

    from backend.core.config import settings

    env_override = os.environ.get("RESEARCH_ARTIFACT_DIR")
    if env_override:
        return Path(env_override)
    configured = (settings.RESEARCH_ARTIFACT_DIR or "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "var" / "research_artifacts"


def stream_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    """Hash a file in chunks; return ``(hex_digest, byte_size)``."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class ArtifactStore:
    """Persist typed artifacts with deterministic references and lineage."""

    def put(
        self,
        artifact_type: ArtifactType,
        payload: Any,
        *,
        checksum: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        producing_run_id: str | None = None,
        implementation_version: str = "1",
        metadata: dict[str, Any] | None = None,
        payload_format: str = "joblib",
    ) -> ArtifactDescriptor:
        resolved_checksum = checksum or self._payload_checksum(payload)
        descriptor = ArtifactDescriptor(
            artifact_id=f"{artifact_type}:{resolved_checksum}",
            artifact_type=artifact_type,
            checksum=resolved_checksum,
            implementation_version=implementation_version,
            parent_artifact_ids=tuple(parent_artifact_ids or []),
            producing_run_id=producing_run_id,
            metadata=dict(metadata or {}),
        )
        key = stage_cache.stage_cache_key(
            engine_version=implementation_version,
            stage_name=f"artifact:{artifact_type}",
            input_checksum=resolved_checksum,
            spec_hash=resolved_checksum,
            params={"parents": descriptor.parent_artifact_ids},
        )
        stage_cache.put_stage(
            key,
            meta={"stage_name": "artifact", "descriptor": asdict(descriptor)},
            payload=payload,
            payload_format=payload_format,
        )
        artifact_registry.register(
            artifact_type,
            descriptor.artifact_id,
            {"cache_key": key, "descriptor": asdict(descriptor)},
            resolved_checksum,
        )
        return descriptor

    def put_file(
        self,
        artifact_type: ArtifactType,
        source_path: Path,
        *,
        filename: str,
        producing_run_id: str | None = None,
        implementation_version: str = "1",
        metadata: dict[str, Any] | None = None,
        content_type: str = "application/octet-stream",
        parent_artifact_ids: list[str] | None = None,
    ) -> ArtifactDescriptor:
        """Persist a filesystem blob under shared artifact storage (streamed hash)."""
        checksum, byte_size = stream_sha256(source_path)
        safe_name = Path(filename).name
        storage_key = f"r_artifacts/{checksum[:2]}/{checksum}/{safe_name}"
        dest = _artifact_root() / storage_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with source_path.open("rb") as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)

        object_key: str | None = None
        storage_backend = "local"
        try:
            from backend.core.config import settings
            from backend.core.storage import object_storage

            if object_storage.is_configured:
                object_key = f"research-r-artifacts/{checksum}/{safe_name}"
                with dest.open("rb") as handle:
                    object_storage.upload_fileobj_sync(
                        object_key=object_key,
                        fileobj=handle,
                        content_type=content_type,
                    )
                storage_backend = "object"
                object_uri = f"s3://{settings.STORAGE_BUCKET}/{object_key}"
            else:
                object_uri = None
        except Exception:
            object_key = None
            storage_backend = "local"
            object_uri = None

        meta = {
            **dict(metadata or {}),
            "filename": safe_name,
            "storage_key": storage_key,
            "storage_backend": storage_backend,
            "object_key": object_key,
            "content_type": content_type,
            "bytes": byte_size,
            "sha256": checksum,
        }
        if object_uri:
            meta["object_uri"] = object_uri

        descriptor = ArtifactDescriptor(
            artifact_id=f"{artifact_type}:{checksum}",
            artifact_type=artifact_type,
            checksum=checksum,
            implementation_version=implementation_version,
            parent_artifact_ids=tuple(parent_artifact_ids or []),
            producing_run_id=producing_run_id,
            metadata=meta,
        )
        key = stage_cache.stage_cache_key(
            engine_version=implementation_version,
            stage_name=f"artifact:{artifact_type}",
            input_checksum=checksum,
            spec_hash=checksum,
            params={"filename": safe_name},
        )
        stage_cache.put_stage(
            key,
            meta={
                "stage_name": "artifact",
                "descriptor": asdict(descriptor),
                "payload_path": str(dest),
                "storage_backend": storage_backend,
                "object_key": object_key,
                "payload_format": "bytes",
            },
            payload=None,
        )
        artifact_registry.register(
            artifact_type,
            descriptor.artifact_id,
            {
                "cache_key": key,
                "descriptor": asdict(descriptor),
                "payload_path": str(dest),
                "storage_backend": storage_backend,
                "object_key": object_key,
            },
            checksum,
        )
        return descriptor

    def get(self, artifact_id: str) -> ArtifactDescriptor | None:
        record = artifact_registry.get(artifact_id)
        if record is None:
            return None
        raw = record.payload_meta.get("descriptor")
        if not isinstance(raw, dict):
            return None
        return ArtifactDescriptor(
            artifact_id=raw["artifact_id"],
            artifact_type=raw["artifact_type"],
            checksum=raw["checksum"],
            implementation_version=raw["implementation_version"],
            parent_artifact_ids=tuple(raw.get("parent_artifact_ids") or ()),
            producing_run_id=raw.get("producing_run_id"),
            created_at=raw["created_at"],
            metadata=dict(raw.get("metadata") or {}),
        )

    def load(self, artifact_id: str) -> Any:
        record = artifact_registry.get(artifact_id)
        if record is None:
            raise KeyError(f"Unknown artifact {artifact_id!r}")
        cached = stage_cache.get_stage(record.payload_meta["cache_key"])
        if cached is not None and "payload" in cached:
            return cached["payload"]
        return self.load_bytes(artifact_id)

    def load_bytes(self, artifact_id: str) -> bytes:
        """Load a durable file artifact without exposing worker-local paths."""
        descriptor = self.get(artifact_id)
        if descriptor is None:
            raise KeyError(f"Unknown artifact {artifact_id!r}")
        meta = descriptor.metadata
        object_key = meta.get("object_key")
        if object_key:
            try:
                from backend.core.storage import object_storage

                if object_storage.is_configured:
                    return object_storage.download_bytes_sync(object_key)
            except Exception:
                pass
        storage_key = meta.get("storage_key")
        if isinstance(storage_key, str) and storage_key:
            path = _artifact_root() / storage_key
            if path.is_file():
                return path.read_bytes()
        record = artifact_registry.get(artifact_id)
        if record is not None:
            payload_path = record.payload_meta.get("payload_path")
            if isinstance(payload_path, str) and Path(payload_path).is_file():
                return Path(payload_path).read_bytes()
            cache_key = record.payload_meta.get("cache_key")
            if cache_key:
                cached = stage_cache.get_stage(str(cache_key))
                if cached and isinstance(cached.get("payload"), (bytes, bytearray)):
                    return bytes(cached["payload"])
        raise FileNotFoundError(f"Artifact payload is unavailable for {artifact_id!r}")

    @staticmethod
    def _payload_checksum(payload: Any) -> str:
        try:
            raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except TypeError:
            raw = repr(payload).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


default_artifact_store = ArtifactStore()
