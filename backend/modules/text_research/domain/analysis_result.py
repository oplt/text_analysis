"""Language-independent result contract for scientific analysis engines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field


class RuntimeInfo(BaseModel):
    engine: str
    implementation: str
    runtime_version: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class AnalysisInputIdentity(BaseModel):
    """One scientific input corpus participating in an analysis.

    Canonical multi-input identity (prompt: ``ScientificInputIdentity``).
    Ordinary analyses carry target only; keyness carries target + reference.
    """

    role: Literal["target", "reference"]
    corpus_checksum: str
    pipeline_checksum: str
    snapshot_id: str | None = None
    snapshot_hash: str | None = None


# Prompt / architecture alias — same model.
ScientificInputIdentity = AnalysisInputIdentity


class AnalysisIdentity(BaseModel):
    """Engine result identity.

    ``corpus_checksum`` / ``pipeline_checksum`` remain target-input compatibility
    fields mirroring ``inputs`` where ``role == "target"``. Prefer ``inputs`` for
    multi-corpus scientific identity.
    """

    spec_hash: str
    corpus_checksum: str
    pipeline_checksum: str
    engine_name: str
    engine_version: str
    inputs: list[AnalysisInputIdentity] | None = None


class AnalysisTiming(BaseModel):
    elapsed_seconds: float | None = None


class AnalysisResult(BaseModel):
    """Validated normalized output shared by Python and R execution engines."""

    schema_version: str = "1.0"
    analysis_type: str
    runtime: RuntimeInfo
    identity: AnalysisIdentity
    results: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    timing: AnalysisTiming = Field(default_factory=AnalysisTiming)


def _input_from_mapping(payload: Mapping[str, Any]) -> AnalysisInputIdentity:
    return AnalysisInputIdentity.model_validate(dict(payload))


def build_scientific_inputs(
    *,
    target_corpus_checksum: str,
    target_pipeline_checksum: str,
    target_snapshot_id: str | None = None,
    target_snapshot_hash: str | None = None,
    reference_corpus_checksum: str | None = None,
    reference_pipeline_checksum: str | None = None,
    reference_snapshot_id: str | None = None,
    reference_snapshot_hash: str | None = None,
) -> list[AnalysisInputIdentity]:
    """Build the canonical scientific-input collection for an analysis."""
    inputs = [
        AnalysisInputIdentity(
            role="target",
            corpus_checksum=target_corpus_checksum,
            pipeline_checksum=target_pipeline_checksum,
            snapshot_id=target_snapshot_id,
            snapshot_hash=target_snapshot_hash or target_corpus_checksum,
        )
    ]
    if reference_corpus_checksum is not None:
        if not reference_pipeline_checksum:
            raise ValueError("reference_pipeline_checksum is required when reference is set")
        inputs.append(
            AnalysisInputIdentity(
                role="reference",
                corpus_checksum=reference_corpus_checksum,
                pipeline_checksum=reference_pipeline_checksum,
                snapshot_id=reference_snapshot_id,
                snapshot_hash=reference_snapshot_hash or reference_corpus_checksum,
            )
        )
    return inputs


def coerce_scientific_inputs(
    inputs: Sequence[AnalysisInputIdentity | Mapping[str, Any]] | None,
) -> list[AnalysisInputIdentity] | None:
    """Normalize dict/model input lists; return ``None`` when empty."""
    if not inputs:
        return None
    return [
        item if isinstance(item, AnalysisInputIdentity) else _input_from_mapping(item)
        for item in inputs
    ]


def scientific_inputs_digest(
    inputs: Sequence[AnalysisInputIdentity | Mapping[str, Any]] | None,
) -> str:
    """Stable digest of multi-input scientific identity.

    Single-target (or empty) collections return ``""`` so ordinary analysis
    computation identities stay backward-compatible.
    """
    coerced = coerce_scientific_inputs(inputs)
    if coerced is None or len(coerced) <= 1:
        return ""
    payload = [
        {
            "role": item.role,
            "corpus_checksum": item.corpus_checksum,
            "pipeline_checksum": item.pipeline_checksum,
            "snapshot_id": item.snapshot_id,
            "snapshot_hash": item.snapshot_hash,
        }
        for item in sorted(coerced, key=lambda row: row.role)
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_analysis_identity(
    *,
    spec_hash: str,
    engine_name: str,
    engine_version: str,
    inputs: Sequence[AnalysisInputIdentity | Mapping[str, Any]],
) -> AnalysisIdentity:
    """Assemble ``AnalysisIdentity`` from the canonical scientific-input list."""
    coerced = coerce_scientific_inputs(list(inputs))
    if not coerced:
        raise ValueError("scientific inputs must include at least the target corpus")
    target = next((item for item in coerced if item.role == "target"), None)
    if target is None:
        raise ValueError("scientific inputs must include a target role")
    return AnalysisIdentity(
        spec_hash=spec_hash,
        corpus_checksum=target.corpus_checksum,
        pipeline_checksum=target.pipeline_checksum,
        engine_name=engine_name,
        engine_version=engine_version,
        inputs=coerced,
    )


def scientific_inputs_as_dicts(
    inputs: Sequence[AnalysisInputIdentity | Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """JSON-ready dump of scientific inputs for manifests/provenance/cache meta."""
    coerced = coerce_scientific_inputs(inputs)
    if coerced is None:
        return None
    return [item.model_dump(mode="json") for item in coerced]
