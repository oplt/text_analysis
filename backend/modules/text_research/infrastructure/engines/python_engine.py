"""Compatibility wrapper for the established Python analysis implementations."""

from __future__ import annotations

import time
from typing import Any

from backend.modules.text_research.domain.analysis_result import (
    AnalysisResult,
    AnalysisTiming,
    RuntimeInfo,
    build_analysis_identity,
    build_scientific_inputs,
)
from backend.modules.text_research.domain.analysis_specification import (
    ANALYSIS_TYPES,
    AnalysisSpecification,
    canonical_implementation_for,
)
from backend.modules.text_research.domain.execution_defaults import ENGINE_VERSION
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact


class PythonAnalysisEngine:
    name = "python"
    implementation = canonical_implementation_for("python")
    implementation_version = ENGINE_VERSION
    supported_analyses = ANALYSIS_TYPES

    def supports(self, analysis_type: str) -> bool:
        return analysis_type in self.supported_analyses

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
        reference = mutable_context.get("prepared_b")
        inputs = build_scientific_inputs(
            target_corpus_checksum=prepared.corpus_checksum,
            target_pipeline_checksum=prepared.pipeline_checksum,
            reference_corpus_checksum=(
                reference.corpus_checksum if reference is not None else None
            ),
            reference_pipeline_checksum=(
                reference.pipeline_checksum if reference is not None else None
            ),
        )
        return AnalysisResult(
            analysis_type=specification.analysis.type,
            runtime=RuntimeInfo(engine=self.name, implementation=self.implementation),
            identity=build_analysis_identity(
                spec_hash=specification.spec_hash(),
                engine_name=self.name,
                engine_version=self.implementation_version,
                inputs=inputs,
            ),
            results=results,
            timing=AnalysisTiming(elapsed_seconds=time.perf_counter() - started),
        )
