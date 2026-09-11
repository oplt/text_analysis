"""Stable boundary between the research pipeline and scientific runtimes."""

from __future__ import annotations

from typing import Any, Protocol

from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact


class AnalysisEngine(Protocol):
    name: str
    implementation: str
    implementation_version: str
    supported_analyses: frozenset[str]

    def supports(self, analysis_type: str) -> bool: ...

    def execute(
        self,
        specification: AnalysisSpecification,
        prepared: PreparedCorpusArtifact,
        **context: Any,
    ) -> AnalysisResult: ...
