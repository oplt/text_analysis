"""Versioned analysis specification shared across research endpoints (§53).

Endpoints may still expose flat request fields; this object is the reusable
configuration vocabulary for corpus selection, preprocessing, features,
model, validation, and output.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusSelection(BaseModel):
    corpus_id: str
    unit_type: str = "paragraph"
    filters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_profile_id: str | None = None
    cleaning_profile_id: str | None = None


class FeatureExtractionSpec(BaseModel):
    type: Literal["count", "binary", "tfidf", "sublinear_tf", "char_ngrams", "bm25"] = "tfidf"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: float | int = 1
    max_df: float | int = 1.0
    max_features: int | None = None
    weighting: str | None = None


class ModelSpec(BaseModel):
    family: str = "logistic_regression"
    task_type: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    class_weight: str | None = "balanced"
    calibration_method: str | None = None


class ValidationSpec(BaseModel):
    strategy: Literal[
        "holdout",
        "grouped_holdout",
        "grouped_cv",
        "leave_one_group_out",
        "temporal",
    ] = "grouped_holdout"
    group_field: str = "source_document"
    temporal_field: str | None = None
    test_size: float = 0.2
    n_splits: int = 5
    random_seed: int = 42


class AnalysisBlock(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class OutputSpec(BaseModel):
    include_manifest: bool = True
    include_unit_ids: bool = True
    max_preview_rows: int = 50


class AnalysisSpecification(BaseModel):
    """Reusable, versioned analysis configuration."""

    spec_version: str = "1.0"
    corpus: CorpusSelection
    feature_extraction: FeatureExtractionSpec = Field(default_factory=FeatureExtractionSpec)
    analysis: AnalysisBlock
    model: ModelSpec | None = None
    validation: ValidationSpec | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    random_seed: int = 42
    notes: str | None = None

    def to_run_parameters(self) -> dict[str, Any]:
        """Flatten into AnalysisRun.parameters_json-compatible dict."""
        payload = self.model_dump()
        payload["filters"] = self.corpus.filters
        payload["unit_type"] = self.corpus.unit_type
        payload["preprocessing_profile_id"] = self.corpus.preprocessing_profile_id
        return payload

    @classmethod
    def from_flat(
        cls,
        *,
        corpus_id: str,
        analysis_type: str,
        unit_type: str = "paragraph",
        filters: dict[str, Any] | None = None,
        preprocessing_profile_id: str | None = None,
        feature: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        analysis_parameters: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> AnalysisSpecification:
        return cls(
            corpus=CorpusSelection(
                corpus_id=corpus_id,
                unit_type=unit_type,
                filters=filters or {},
                preprocessing_profile_id=preprocessing_profile_id,
            ),
            feature_extraction=FeatureExtractionSpec(**(feature or {})),
            analysis=AnalysisBlock(type=analysis_type, parameters=analysis_parameters or {}),
            model=ModelSpec(**model) if model else None,
            validation=ValidationSpec(**validation) if validation else None,
            random_seed=random_seed,
        )
