"""Thin adapter around the isolated repository-owned R runtime."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.domain.analysis_result import AnalysisResult
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.exceptions import RUnsupportedAnalysis
from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.r_runtime.runner import (
    r_runtime_available,
    run_r_job,
)
from backend.modules.text_research.infrastructure.r_runtime.serializer import serialize_r_job


class RAnalysisEngine:
    name = "r"
    implementation_version = "r-quanteda-1"
    supported_analyses = frozenset(
        {"frequencies", "dfm", "kwic", "dictionary", "keyness", "cooccurrence"}
    )

    def supports(self, analysis_type: str) -> bool:
        return analysis_type in self.supported_analyses

    @classmethod
    def available(cls) -> bool:
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
                force_in_memory=pipeline_context.get("force_in_memory", False),
            )
            pipeline_context["prepared_b"] = comparison_prepared
        bundle = serialize_r_job(
            specification=specification,
            prepared=prepared,
            comparison_prepared=comparison_prepared,
            run_id=str(context.get("run_id") or "in-memory"),
            engine_version=self.implementation_version,
        )
        return run_r_job(bundle, expected_engine_version=self.implementation_version)
