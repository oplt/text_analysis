"""In-process typed registry for content-addressable research artifacts."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

ARTIFACT_KINDS = frozenset(
    {
        "prepared_corpus",
        "stage",
        "dfm",
        "embeddings",
        "model",
        "topic_model",
        "matrix",
        "vectorizer",
        "topic_matrix",
        "export",
        "manifest",
    }
)


@dataclass
class ArtifactRecord:
    kind: str
    artifact_id: str
    payload_meta: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


class ArtifactRegistry:
    """Thread-safe, in-process artifact index keyed by id and kind+checksum."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, ArtifactRecord] = {}
        self._by_checksum: dict[tuple[str, str], str] = {}

    def register(
        self,
        kind: str,
        artifact_id: str,
        payload_meta: dict[str, Any],
        checksum: str,
    ) -> str:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unknown artifact kind {kind!r}; expected one of {sorted(ARTIFACT_KINDS)}")

        with self._lock:
            existing_id = self._by_checksum.get((kind, checksum))
            if existing_id is not None:
                return existing_id

            resolved_id = artifact_id or str(uuid4())
            if resolved_id in self._by_id:
                existing = self._by_id[resolved_id]
                if existing.kind == kind and existing.checksum == checksum:
                    self._by_checksum[(kind, checksum)] = resolved_id
                    return resolved_id
                raise ValueError(
                    f"artifact_id {resolved_id!r} already registered with a different checksum"
                )

            record = ArtifactRecord(
                kind=kind,
                artifact_id=resolved_id,
                payload_meta=dict(payload_meta),
                checksum=checksum,
            )
            self._by_id[resolved_id] = record
            self._by_checksum[(kind, checksum)] = resolved_id
            return resolved_id

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            return self._by_id.get(artifact_id)

    def find_by_checksum(self, kind: str, checksum: str) -> ArtifactRecord | None:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"Unknown artifact kind {kind!r}; expected one of {sorted(ARTIFACT_KINDS)}")
        with self._lock:
            artifact_id = self._by_checksum.get((kind, checksum))
            if artifact_id is None:
                return None
            return self._by_id.get(artifact_id)

    def list_artifacts(self, kind: str | None = None) -> list[ArtifactRecord]:
        with self._lock:
            records = list(self._by_id.values())
        if kind is not None:
            if kind not in ARTIFACT_KINDS:
                raise ValueError(f"Unknown artifact kind {kind!r}; expected one of {sorted(ARTIFACT_KINDS)}")
            records = [record for record in records if record.kind == kind]
        return sorted(records, key=lambda record: (record.kind, record.artifact_id))


_default_registry = ArtifactRegistry()


def register(
    kind: str,
    artifact_id: str,
    payload_meta: dict[str, Any],
    checksum: str,
) -> str:
    return _default_registry.register(kind, artifact_id, payload_meta, checksum)


def get(artifact_id: str) -> ArtifactRecord | None:
    return _default_registry.get(artifact_id)


def find_by_checksum(kind: str, checksum: str) -> ArtifactRecord | None:
    return _default_registry.find_by_checksum(kind, checksum)


def list_artifacts(kind: str | None = None) -> list[ArtifactRecord]:
    return _default_registry.list_artifacts(kind)
