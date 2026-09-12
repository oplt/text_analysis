"""Detect oversized analysis result payloads and offload them to artifacts.

Small summaries stay inline in ``AnalysisRun.results_json``. Large arrays
(pair rows, sparse structures, feature lists, …) are stored in the artifact
store with a compact preview + reference left on the run.
"""

from __future__ import annotations

import json
from typing import Any

from backend.modules.text_research.infrastructure.artifact_store import ArtifactStore

# Serialized JSON byte threshold before offloading the bulk payload.
DEFAULT_INLINE_RESULTS_MAX_BYTES = 512_000
# Keep a small preview of large list/dict fields in the inline JSON.
DEFAULT_PREVIEW_ROWS = 50

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
        metadata={"kind": "analysis_results", "bytes": size},
        payload_format="json",
    )

    inline: dict[str, Any] = {
        "artifactized": True,
        "results_artifact_id": descriptor.artifact_id,
        "results_checksum": descriptor.checksum,
        "results_bytes": size,
        "preview_rows": preview_rows,
    }
    # Preserve scalar / small summary fields; preview large arrays.
    for key, value in results.items():
        if key in _LARGE_ARRAY_KEYS or (
            isinstance(value, list) and len(value) > preview_rows
        ):
            inline[f"{key}_preview"] = _preview_value(value, preview_rows=preview_rows)
            if isinstance(value, list):
                inline[f"{key}_total"] = len(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            inline[key] = value
        elif isinstance(value, dict) and serialized_size_bytes(value) <= max_inline_bytes // 4:
            inline[key] = value
        else:
            inline[f"{key}_preview"] = _preview_value(value, preview_rows=preview_rows)

    return inline, descriptor.artifact_id
