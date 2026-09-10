"""Language-independent result contract for scientific analysis engines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeInfo(BaseModel):
    engine: str
    implementation: str
    runtime_version: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class AnalysisIdentity(BaseModel):
    spec_hash: str
    corpus_checksum: str
    pipeline_checksum: str
    engine_name: str
    engine_version: str


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
