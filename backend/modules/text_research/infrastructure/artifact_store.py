"""Typed, content-addressable store for research analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
            parent_artifact_ids=tuple(parent_artifact_ids or ()),
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
        if cached is None or "payload" not in cached:
            raise FileNotFoundError(f"Artifact payload is unavailable for {artifact_id!r}")
        return cached["payload"]

    @staticmethod
    def _payload_checksum(payload: Any) -> str:
        try:
            raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except TypeError:
            raw = repr(payload).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


default_artifact_store = ArtifactStore()
