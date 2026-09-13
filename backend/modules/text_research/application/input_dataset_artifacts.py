"""Immutable input datasets for statistical / measurement analyses (LATEST-015).

Raw observation matrices and aligned value arrays are stored in ArtifactStore,
never in AnalysisRun.parameters_json. Runs keep only ``input_artifact_id``,
checksum, and compact summaries (row counts, variable names, source labels).
"""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.infrastructure.artifact_store import (
    ArtifactDescriptor,
    ArtifactStore,
)

STATISTICAL_INPUT_KIND = "statistical_model_input"
MEASUREMENT_INPUT_KIND = "measurement_comparison_input"


def build_statistical_input_payload(
    *,
    model: str,
    dependent_var: str,
    independent_vars: list[str],
    rows: list[dict[str, Any]],
    add_intercept: bool,
) -> dict[str, Any]:
    """Normalize the exact dataset + model knobs used for fitting."""
    return {
        "schema": {
            "dependent_var": dependent_var,
            "independent_vars": list(independent_vars),
            "row_fields": sorted({dependent_var, *independent_vars}),
        },
        "model": model,
        "dependent_var": dependent_var,
        "independent_vars": list(independent_vars),
        "rows": list(rows),
        "add_intercept": bool(add_intercept),
        "row_count": len(rows),
        "variable_names": [dependent_var, *list(independent_vars)],
    }


def statistical_input_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": STATISTICAL_INPUT_KIND,
        "schema": payload.get("schema"),
        "row_count": int(payload.get("row_count") or len(payload.get("rows") or [])),
        "variable_names": list(payload.get("variable_names") or []),
        "model": payload.get("model"),
        "add_intercept": payload.get("add_intercept"),
    }


def build_measurement_input_payload(
    *,
    source_a: str,
    values_a: list[Any],
    source_b: str,
    values_b: list[Any],
    ids: list[str] | None,
    value_kind: str,
    subgroup: list[str] | None,
) -> dict[str, Any]:
    """Normalize the exact paired series used for triangulation."""
    return {
        "schema": {
            "value_kind": value_kind,
            "fields": ["values_a", "values_b", "ids", "subgroup"],
        },
        "source_a": source_a,
        "values_a": list(values_a),
        "source_b": source_b,
        "values_b": list(values_b),
        "ids": list(ids) if ids is not None else None,
        "value_kind": value_kind,
        "subgroup": list(subgroup) if subgroup is not None else None,
        "row_count": len(values_a),
        "source_labels": [source_a, source_b],
    }


def measurement_input_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": MEASUREMENT_INPUT_KIND,
        "schema": payload.get("schema"),
        "row_count": int(payload.get("row_count") or len(payload.get("values_a") or [])),
        "source_labels": list(payload.get("source_labels") or []),
        "value_kind": payload.get("value_kind"),
    }


def store_input_dataset(payload: dict[str, Any], *, metadata: dict[str, Any]) -> ArtifactDescriptor:
    """Persist a content-addressed JSON input dataset artifact."""
    return ArtifactStore().put(
        "manifest",
        payload,
        metadata=metadata,
        payload_format="json",
    )


def input_dataset_checksum(payload: dict[str, Any]) -> str:
    """Stable SHA-256 of a normalized input payload (same as ArtifactStore)."""
    return ArtifactStore._payload_checksum(payload)
