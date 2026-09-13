"""Detect oversized analysis result payloads and serve authorized retrieval.

Small summaries stay inline in ``AnalysisRun.results_json``. Large arrays
(pair rows, sparse structures, feature lists, …) are stored in the artifact
store with a compact preview + reference left on the run.

Retrieval always goes through an authorized run: callers resolve project
access via ``RunService.get_run`` before loading artifact bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.modules.text_research.domain.models import AnalysisRun, loads
from backend.modules.text_research.infrastructure.artifact_store import (
    ArtifactDescriptor,
    ArtifactStore,
)

# Serialized JSON byte threshold before offloading the bulk payload.
DEFAULT_INLINE_RESULTS_MAX_BYTES = 512_000
# Keep a small preview of large list/dict fields in the inline JSON.
DEFAULT_PREVIEW_ROWS = 50
RESULTS_ARTIFACT_SCHEMA_VERSION = 1
RESULTS_ARTIFACT_MEDIA_TYPE = "application/json"

# Keys commonly holding large row payloads across quantitative analyses.
_LARGE_ARRAY_KEYS = (
    "pairs",
    "items",
    "rows",
    "cooccurrence",
    "frequencies",
    "ngrams",
    "feature_names",
    "assignments",
    "coordinates",
    "document_topics",
    "topics",
    "duplicates",
    "clusters",
    "network",
    "report",
    "sparse",
    "matrix",
)


def serialized_size_bytes(payload: Any) -> int:
    try:
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(repr(payload).encode("utf-8"))


def _preview_value(value: Any, *, preview_rows: int) -> Any:
    if isinstance(value, list):
        return value[:preview_rows]
    if isinstance(value, dict):
        # Preserve small dicts; truncate list-valued children.
        out: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(child, list) and len(child) > preview_rows:
                out[key] = child[:preview_rows]
            else:
                out[key] = child
        return out
    return value


def maybe_artifactize_results(
    results: dict[str, Any],
    *,
    max_inline_bytes: int = DEFAULT_INLINE_RESULTS_MAX_BYTES,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    producing_run_id: str | None = None,
    store: ArtifactStore | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Return ``(inline_results, artifact_ref_or_none)``.

    When the serialized payload exceeds ``max_inline_bytes``, store the full
    results in the artifact registry and keep metrics-friendly previews inline.
    The managed result reference is ``results_artifact_id`` — not
    ``AnalysisRun.artifact_path`` (reserved for model/vectorizer paths).
    """
    if not isinstance(results, dict):
        return results, None
    size = serialized_size_bytes(results)
    if size <= max_inline_bytes:
        return results, None

    artifact_store = store or ArtifactStore()
    descriptor = artifact_store.put(
        "export",
        results,
        producing_run_id=producing_run_id,
        metadata={
            "kind": "analysis_results",
            "bytes": size,
            "schema_version": RESULTS_ARTIFACT_SCHEMA_VERSION,
            "media_type": RESULTS_ARTIFACT_MEDIA_TYPE,
            "preview_rows": preview_rows,
        },
        payload_format="json",
    )

    inline: dict[str, Any] = {
        "artifactized": True,
        "results_artifact_id": descriptor.artifact_id,
        "results_checksum": descriptor.checksum,
        "results_bytes": size,
        "results_media_type": RESULTS_ARTIFACT_MEDIA_TYPE,
        "results_schema_version": RESULTS_ARTIFACT_SCHEMA_VERSION,
        "preview_rows": preview_rows,
    }
    # Preserve scalar / small summary fields; preview large arrays.
    for key, value in results.items():
        if key in _LARGE_ARRAY_KEYS or (isinstance(value, list) and len(value) > preview_rows):
            inline[f"{key}_preview"] = _preview_value(value, preview_rows=preview_rows)
            if isinstance(value, list):
                inline[f"{key}_total"] = len(value)
        elif (
            isinstance(value, (str, int, float, bool))
            or value is None
            or (isinstance(value, dict) and serialized_size_bytes(value) <= max_inline_bytes // 4)
        ):
            inline[key] = value
        else:
            inline[f"{key}_preview"] = _preview_value(value, preview_rows=preview_rows)

    return inline, descriptor.artifact_id


def resolve_results_artifact_id(run: AnalysisRun | Any) -> str | None:
    """Prefer explicit results artifact refs over generic ``artifact_path``."""
    results = loads(getattr(run, "results_json", None), {}) or {}
    metrics = loads(getattr(run, "metrics_json", None), {}) or {}
    for source in (results, metrics):
        if not isinstance(source, dict):
            continue
        artifact_id = source.get("results_artifact_id")
        if isinstance(artifact_id, str) and artifact_id.strip():
            return artifact_id.strip()
    return None


def inline_results_payload(run: AnalysisRun | Any) -> dict[str, Any]:
    payload = loads(getattr(run, "results_json", None), {}) or {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class RunResultsPage:
    """Authorized paged view of inline or artifactized run results."""

    artifact_id: str | None
    checksum: str | None
    media_type: str | None
    schema_version: int | None
    byte_size: int | None
    preview_rows: int | None
    key: str | None
    items: list[Any] | None
    data: Any
    total: int | None
    row_count: int | None
    limit: int
    offset: int
    artifactized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "checksum": self.checksum,
            "media_type": self.media_type,
            "schema_version": self.schema_version,
            "byte_size": self.byte_size,
            "preview_rows": self.preview_rows,
            "key": self.key,
            "items": self.items,
            "data": self.data,
            "total": self.total,
            "row_count": self.row_count,
            "limit": self.limit,
            "offset": self.offset,
            "artifactized": self.artifactized,
        }


def _descriptor_meta(
    descriptor: ArtifactDescriptor | None,
    inline: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(descriptor.metadata) if descriptor is not None else {}
    return {
        "checksum": (
            descriptor.checksum if descriptor is not None else inline.get("results_checksum")
        ),
        "media_type": (
            meta.get("media_type")
            or inline.get("results_media_type")
            or (RESULTS_ARTIFACT_MEDIA_TYPE if descriptor is not None else None)
        ),
        "schema_version": (
            meta.get("schema_version")
            or inline.get("results_schema_version")
            or (RESULTS_ARTIFACT_SCHEMA_VERSION if descriptor is not None else None)
        ),
        "byte_size": meta.get("bytes") or inline.get("results_bytes"),
        "preview_rows": meta.get("preview_rows") or inline.get("preview_rows"),
    }


def page_run_results(
    run: AnalysisRun | Any,
    *,
    key: str | None = None,
    limit: int = 100,
    offset: int = 0,
    store: ArtifactStore | None = None,
) -> RunResultsPage:
    """Page a result array (or return a scalar payload) for an authorized run."""
    artifact_store = store or ArtifactStore()
    inline = inline_results_payload(run)
    artifact_id = resolve_results_artifact_id(run)
    descriptor: ArtifactDescriptor | None = None
    payload: Any = inline
    if artifact_id:
        descriptor = artifact_store.get(artifact_id)
        if descriptor is None:
            raise FileNotFoundError("Result artifact is unavailable.")
        payload = artifact_store.load(artifact_id)

    selected_key = key
    if key is not None:
        if not isinstance(payload, dict) or key not in payload:
            raise KeyError(key)
        payload = payload[key]

    meta = _descriptor_meta(descriptor, inline)
    if isinstance(payload, list):
        total = len(payload)
        return RunResultsPage(
            artifact_id=artifact_id,
            checksum=meta["checksum"] if isinstance(meta["checksum"], str) else None,
            media_type=meta["media_type"] if isinstance(meta["media_type"], str) else None,
            schema_version=int(meta["schema_version"])
            if isinstance(meta["schema_version"], int)
            else None,
            byte_size=int(meta["byte_size"]) if isinstance(meta["byte_size"], int) else None,
            preview_rows=int(meta["preview_rows"])
            if isinstance(meta["preview_rows"], int)
            else None,
            key=selected_key,
            items=payload[offset : offset + limit],
            data=None,
            total=total,
            row_count=total,
            limit=limit,
            offset=offset,
            artifactized=artifact_id is not None,
        )

    return RunResultsPage(
        artifact_id=artifact_id,
        checksum=meta["checksum"] if isinstance(meta["checksum"], str) else None,
        media_type=meta["media_type"] if isinstance(meta["media_type"], str) else None,
        schema_version=int(meta["schema_version"])
        if isinstance(meta["schema_version"], int)
        else None,
        byte_size=int(meta["byte_size"]) if isinstance(meta["byte_size"], int) else None,
        preview_rows=int(meta["preview_rows"]) if isinstance(meta["preview_rows"], int) else None,
        key=selected_key,
        items=None,
        data=payload,
        total=None,
        row_count=None,
        limit=limit,
        offset=offset,
        artifactized=artifact_id is not None,
    )


def load_full_run_results(
    run: AnalysisRun | Any,
    *,
    store: ArtifactStore | None = None,
) -> tuple[Any, str | None, ArtifactDescriptor | None]:
    """Load the complete result payload for download after run authorization."""
    artifact_store = store or ArtifactStore()
    artifact_id = resolve_results_artifact_id(run)
    if artifact_id:
        descriptor = artifact_store.get(artifact_id)
        if descriptor is None:
            raise FileNotFoundError("Result artifact is unavailable.")
        return artifact_store.load(artifact_id), artifact_id, descriptor
    return inline_results_payload(run), None, None
