"""Execution task contract for compiled research analyses."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.modules.text_research.domain.analysis_specification import ANALYSIS_TYPE_ALIASES
from backend.modules.text_research.infrastructure.execution_policy import (
    retry_policy_for,
    timeout_for,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import computation_identity

__all__ = [
    "AnalysisTask",
    "RESOURCE_CLASSES",
    "computation_identity",
    "resource_class_for",
]


RESOURCE_CLASSES: frozenset[str] = frozenset(
    {
        "research_light",
        "research_cpu",
        "research_io",
        "research_nlp",
        "research_memory",
        "research_gpu",
    }
)

_LIGHT_ANALYSES = frozenset(
    {"frequencies", "kwic", "dictionary", "readability", "keyness", "cooccurrence"}
)
_MEMORY_ANALYSES = frozenset({"dfm"})
_NLP_ANALYSES = frozenset({"embedding", "ner", "linguistic_features"})
_GPU_ANALYSES = frozenset({"topic_model"})
_CPU_ANALYSES = frozenset({"classification", "clustering", "similarity", "statistical_model"})


def resource_class_for(analysis_type: str) -> str:
    """Map an analysis type to a worker resource class."""
    normalized = ANALYSIS_TYPE_ALIASES.get(analysis_type, analysis_type)
    if normalized in _LIGHT_ANALYSES:
        return "research_light"
    if normalized in _MEMORY_ANALYSES:
        return "research_memory"
    if normalized in _NLP_ANALYSES:
        return "research_nlp"
    if normalized in _GPU_ANALYSES:
        return "research_gpu"
    if normalized in _CPU_ANALYSES:
        return "research_cpu"
    return "research_cpu"


@dataclass
class AnalysisTask:
    id: str
    specification_hash: str
    input_artifact_ids: list[str]
    resource_class: str  # research_light|research_cpu|research_memory|research_gpu
    retry_policy: dict[str, Any]
    timeout_seconds: int | None
    progress: float
    checkpoint: dict[str, Any] | None
    output_artifact_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.resource_class not in RESOURCE_CLASSES:
            raise ValueError(
                f"resource_class must be one of {sorted(RESOURCE_CLASSES)}; "
                f"got {self.resource_class!r}"
            )

    @classmethod
    def create(
        cls,
        analysis_type: str,
        spec_hash: str,
        input_artifact_ids: list[str] | None = None,
    ) -> AnalysisTask:
        """Build a task with resource class, retry policy, and timeout defaults."""
        resource_class = resource_class_for(analysis_type)
        return cls(
            id=str(uuid.uuid4()),
            specification_hash=spec_hash,
            input_artifact_ids=list(input_artifact_ids or []),
            resource_class=resource_class,
            retry_policy=retry_policy_for(resource_class),
            timeout_seconds=timeout_for(resource_class),
            progress=0.0,
            checkpoint=None,
        )
