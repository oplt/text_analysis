"""Compatibility wrapper for the established Python analysis implementations."""

from __future__ import annotations

import time
from typing import Any

from backend.modules.text_research.domain.analysis_result import (
    AnalysisIdentity,
    AnalysisResult,
    AnalysisTiming,
    RuntimeInfo,
)
from backend.modules.text_research.domain.analysis_specification import (
    ANALYSIS_TYPES,
    AnalysisSpecification,
)
from backend.modules.text_research.domain.execution_defaults import ENGINE_VERSION
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact


class PythonAnalysisEngine:
    name = "python"
    implementation_version = ENGINE_VERSION

    def supports(self, analysis_type: str) -> bool:
        return analysis_type in ANALYSIS_TYPES

    def execute(
        self,
        specification: AnalysisSpecification,
        prepared: PreparedCorpusArtifact,
        **context: Any,
    ) -> AnalysisResult:
        if not self.supports(specification.analysis.type):
            raise ValueError(f"Unsupported Python analysis: {specification.analysis.type}")
        from backend.modules.text_research.infrastructure.stage_runner import (
            execute_python_analysis,
        )

        mutable_context = context["pipeline_context"]
        started = time.perf_counter()
        results = execute_python_analysis(mutable_context, context["plan"])
        return AnalysisResult(
            analysis_type=specification.analysis.type,
            runtime=RuntimeInfo(engine=self.name, implementation="python"),
            identity=AnalysisIdentity(
                spec_hash=specification.spec_hash(),
                corpus_checksum=prepared.corpus_checksum,
                pipeline_checksum=prepared.pipeline_checksum,
                engine_name=self.name,
                engine_version=self.implementation_version,
            ),
            results=results,
            timing=AnalysisTiming(elapsed_seconds=time.perf_counter() - started),
        )
