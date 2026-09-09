"""Compile AnalysisSpecification objects into deterministic execution plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification

ENGINE_VERSION = "text_research.pipeline/1"

BASE_STAGES: tuple[str, ...] = ("validate_spec", "resolve_corpus", "prepare_corpus")
TERMINAL_STAGES: tuple[str, ...] = ("persist_run",)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    input_artifact_type: str | None = None
    output_artifact_type: str | None = None
    cacheable: bool = True


@dataclass(frozen=True)
class ResourceProfile:
    resource_class: str
    cpu: int | None = None
    memory_mb: int | None = None


@dataclass
class ExecutionContext:
    specification: AnalysisSpecification
    artifacts: dict[str, Any] = field(default_factory=dict)
    resource_profile: ResourceProfile | None = None


class AnalysisPlugin(Protocol):
    name: str
    implementation_version: str
    deterministic: bool
    cacheable: bool

    def compile(self, specification: AnalysisSpecification) -> list[PipelineStep]: ...


@dataclass(frozen=True)
class ExecutionPlan:
    stages: list[str]
    spec_hash: str
    engine_version: str = field(default=ENGINE_VERSION)
    steps: tuple[PipelineStep, ...] = ()


def compile_plan(spec: AnalysisSpecification) -> ExecutionPlan:
    """Build a stage list for the normalized, validated specification."""
    normalized = spec.normalize()
    normalized.validate()

    stages = list(BASE_STAGES)
    stages.append(normalized.analysis.type)
    stages.extend(TERMINAL_STAGES)
    if normalized.output.include_manifest:
        stages.append("build_manifest")

    steps = tuple(PipelineStep(name=stage) for stage in stages)
    return ExecutionPlan(
        stages=stages,
        spec_hash=normalized.spec_hash(),
        engine_version=ENGINE_VERSION,
        steps=steps,
    )


def computation_identity(
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str = ENGINE_VERSION,
) -> str:
    """Stable identity for a prepared-corpus computation artifact."""
    payload = f"{spec_hash}:{corpus_snapshot_hash}:{engine_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
