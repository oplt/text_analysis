"""Thin adapter around the isolated repository-owned R runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.domain.analysis_specification import (
    AnalysisSpecification,
    canonical_implementation_for,
)
from backend.modules.text_research.domain.exceptions import RUnsupportedAnalysis
from backend.modules.text_research.domain.execution_defaults import R_ENGINE_VERSION
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.r_runtime.capabilities import r_feature_enabled
from backend.modules.text_research.infrastructure.r_runtime.runner import (
    r_runtime_available,
    run_r_job,
)
from backend.modules.text_research.infrastructure.r_runtime.serializer import serialize_r_job

# Headless / CLI executions without a persisted AnalysisRun.
IN_MEMORY_ANALYSIS_RUN_ID = "in-memory"


def resolve_analysis_run_id(
    *,
    run_id: Any = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the AnalysisRun id for R I/O; never infer from a workdir name."""
    candidates = (run_id, (pipeline_context or {}).get("run_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return IN_MEMORY_ANALYSIS_RUN_ID


class RAnalysisEngine:
    name = "r"
    implementation = canonical_implementation_for("r")
    implementation_version = R_ENGINE_VERSION
    supported_analyses = frozenset(
        {"frequencies", "dfm", "kwic", "dictionary", "keyness", "cooccurrence"}
    )

    def supports(self, analysis_type: str) -> bool:
        return analysis_type in self.supported_analyses

    @classmethod
    def available(cls) -> bool:
        """Feature flag: R may be selected / queued without local Rscript on the API."""
        return r_feature_enabled()

    @classmethod
    def runtime_ready(cls) -> bool:
        """Worker-local check: this process can execute Rscript."""
        return r_runtime_available()

    def execute(
        self,
        specification: AnalysisSpecification,
        prepared: PreparedCorpusArtifact,
        **context: Any,
    ) -> AnalysisResult:
        if specification.engine.preprocessing_mode != "standardized":
            raise ValueError("R native preprocessing is not implemented")
        if not self.supports(specification.analysis.type):
            raise RUnsupportedAnalysis(f"Unsupported R analysis: {specification.analysis.type}")
        pipeline_context = context.get("pipeline_context") or {}
        comparison_prepared = pipeline_context.get("prepared_b")
        if specification.analysis.type == "keyness" and comparison_prepared is None:
            texts_b = pipeline_context.get("texts_b")
            if not texts_b:
                raise RUnsupportedAnalysis("R keyness requires comparison corpus tokens")
            comparison_prepared = prepare_texts(
                texts_b,
                prepared.preprocessing_profile,
                unit_ids=pipeline_context.get("unit_ids_b"),
                document_ids=pipeline_context.get("document_ids_b"),
                force_in_memory=pipeline_context.get("force_in_memory", False),
            )
            pipeline_context["prepared_b"] = comparison_prepared
        analysis_run_id = resolve_analysis_run_id(
            run_id=context.get("run_id"),
            pipeline_context=pipeline_context if isinstance(pipeline_context, Mapping) else None,
        )
        bundle = serialize_r_job(
            specification=specification,
            prepared=prepared,
            comparison_prepared=comparison_prepared,
            run_id=analysis_run_id,
            engine_version=self.implementation_version,
        )
        return run_r_job(bundle, expected_engine_version=self.implementation_version)
