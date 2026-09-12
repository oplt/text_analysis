"""Versioned analysis specification — heart of the research analysis backend.

Flat HTTP request fields are accepted at the edge, but every analysis should
normalize through :class:`AnalysisSpecification` (v2): validate → normalize →
``spec_hash`` → :mod:`pipeline_compiler` → execution plan → artifacts →
``AnalysisRun``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ANALYSIS_TYPES: frozenset[str] = frozenset(
    {
        "frequencies",
        "dfm",
        "kwic",
        "dictionary",
        "keyness",
        "cooccurrence",
        "topic_model",
        "classification",
        "clustering",
        "similarity",
        "statistical_model",
        "measurement_validation",
        "readability",
    }
)

ANALYSIS_TYPE_ALIASES: dict[str, str] = {
    "frequency_analysis": "frequencies",
    "ngram_analysis": "frequencies",
    "dictionary_analysis": "dictionary",
    "classifier_training": "classification",
    "classifier_prediction": "classification",
}


class CorpusSelection(BaseModel):
    corpus_id: str
    snapshot_id: str | None = None
    unit_type: str = "paragraph"
    filters: dict[str, Any] = Field(default_factory=dict)
    language_mode: Literal["auto", "manual"] = "auto"


class PreprocessingSpec(BaseModel):
    preprocessing_profile_id: str | None = None
    cleaning_profile_id: str | None = None
    # Resolved config fingerprint — not the mutable DB profile alone.
    preprocessing_config_hash: str | None = None


class FeatureSelectionSpec(BaseModel):
    method: Literal["none", "chi2", "mutual_info", "l1"] = "none"
    k: int | str = "all"
    percentile: float | None = None


class FeatureExtractionSpec(BaseModel):
    type: Literal["count", "binary", "tfidf", "sublinear_tf", "char_ngrams", "bm25"] = "tfidf"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: float | int = 1
    max_df: float | int = 1.0
    max_features: int | None = None
    weighting: str | None = None
    selection: FeatureSelectionSpec | None = None


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


class ExecutionSpec(BaseModel):
    cpu: int | None = None
    memory_mb: int | None = None


class AnalysisBlock(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class OutputSpec(BaseModel):
    include_manifest: bool = True
    include_unit_ids: bool = True
    max_preview_rows: int = 50


class AnalysisSpecification(BaseModel):
    """Reusable, versioned analysis configuration."""

    spec_version: str = "2.0"
    corpus: CorpusSelection
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    feature_extraction: FeatureExtractionSpec = Field(default_factory=FeatureExtractionSpec)
    analysis: AnalysisBlock
    model: ModelSpec | None = None
    validation: ValidationSpec | None = None
    execution: ExecutionSpec | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    random_seed: int = 42
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_features_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "feature_extraction" not in data and "features" in data:
            data = dict(data)
            data["feature_extraction"] = data.pop("features")
        return data

    def normalize(self) -> AnalysisSpecification:
        """Return a canonical copy with stable key ordering and defaults filled."""
        payload = self.model_dump()
        payload["corpus"]["filters"] = normalize_corpus_filters(payload["corpus"]["filters"])
        payload["analysis"]["parameters"] = _sort_dict_deep(payload["analysis"]["parameters"])
        payload["feature_extraction"] = _sort_dict_deep(payload["feature_extraction"])
        if payload.get("model"):
            payload["model"]["hyperparameters"] = _sort_dict_deep(
                payload["model"]["hyperparameters"]
            )
        analysis_type = _canonical_analysis_type(payload["analysis"]["type"])
        payload["analysis"]["type"] = analysis_type
        return AnalysisSpecification.model_validate(payload)

    def validate(self) -> None:
        """Raise ValueError when the specification describes an invalid combination."""
        normalized = self.normalize()
        if not normalized.corpus.corpus_id.strip():
            raise ValueError("corpus.corpus_id must be non-empty")
        if normalized.analysis.type not in ANALYSIS_TYPES:
            raise ValueError(f"unsupported analysis.type: {normalized.analysis.type!r}")

        if (
            normalized.corpus.language_mode == "manual"
            and "language" not in normalized.corpus.filters
        ):
            raise ValueError("corpus.filters.language is required when language_mode is 'manual'")

        if normalized.validation is not None:
            strategy = normalized.validation.strategy
            if (
                strategy in {"grouped_holdout", "grouped_cv", "leave_one_group_out"}
                and not normalized.validation.group_field.strip()
            ):
                raise ValueError(
                    "validation.group_field is required for grouped validation strategies"
                )
            if strategy == "temporal" and not normalized.validation.temporal_field:
                raise ValueError(
                    "validation.temporal_field is required when strategy is 'temporal'"
                )

        if normalized.analysis.type == "classification":
            if normalized.model is None:
                raise ValueError("model is required for classification analyses")
            if normalized.validation is None:
                raise ValueError("validation is required for classification analyses")

        if normalized.execution is not None:
            if normalized.execution.cpu is not None and normalized.execution.cpu <= 0:
                raise ValueError("execution.cpu must be positive when set")
            if normalized.execution.memory_mb is not None and normalized.execution.memory_mb <= 0:
                raise ValueError("execution.memory_mb must be positive when set")

    def spec_hash(self) -> str:
        """Stable sha256 digest of the normalized specification JSON."""
        normalized = self.normalize()
        canonical = json.dumps(
            normalized.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_run_parameters(self) -> dict[str, Any]:
        """Flatten into AnalysisRun.parameters_json-compatible dict."""
        normalized = self.normalize()
        payload = normalized.model_dump(mode="json")
        payload["filters"] = dict(normalized.corpus.filters)
        payload["unit_type"] = normalized.corpus.unit_type
        payload["preprocessing_profile_id"] = normalized.preprocessing.preprocessing_profile_id
        payload["cleaning_profile_id"] = normalized.preprocessing.cleaning_profile_id
        payload["preprocessing_config_hash"] = normalized.preprocessing.preprocessing_config_hash
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
        cleaning_profile_id: str | None = None,
        preprocessing_config_hash: str | None = None,
        snapshot_id: str | None = None,
        language_mode: Literal["auto", "manual"] = "auto",
        feature: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        analysis_parameters: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> AnalysisSpecification:
        return cls(
            corpus=CorpusSelection(
                corpus_id=corpus_id,
                snapshot_id=snapshot_id,
                unit_type=unit_type,
                filters=normalize_corpus_filters(filters),
                language_mode=language_mode,
            ),
            preprocessing=PreprocessingSpec(
                preprocessing_profile_id=preprocessing_profile_id,
                cleaning_profile_id=cleaning_profile_id,
                preprocessing_config_hash=preprocessing_config_hash,
            ),
            feature_extraction=FeatureExtractionSpec(**(feature or {})),
            analysis=AnalysisBlock(
                type=_canonical_analysis_type(analysis_type),
                parameters=analysis_parameters or {},
            ),
            model=ModelSpec(**model) if model else None,
            validation=ValidationSpec(**validation) if validation else None,
            execution=ExecutionSpec(**execution) if execution else None,
            output=OutputSpec(**(output or {})),
            random_seed=random_seed,
        )


_FILTER_KEY_ALIASES: dict[str, str] = {
    "year": "publication_year",
    "year_min": "publication_year_min",
    "year_max": "publication_year_max",
}

_FILTER_INT_KEYS: frozenset[str] = frozenset(
    {
        "publication_year",
        "publication_year_min",
        "publication_year_max",
    }
)


def normalize_corpus_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Canonicalize corpus selection filters for stable hashing.

    - Drops null values
    - Maps year aliases onto publication_year*
    - Coerces year fields to int when possible
    - Sorts list values (e.g. document_ids)
    - Deep-sorts keys
    """
    if not filters:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        canon = _FILTER_KEY_ALIASES.get(key, key)
        if canon in _FILTER_INT_KEYS:
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        elif isinstance(value, list):
            value = sorted((_normalize_filter_list_item(item) for item in value), key=str)
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        normalized[canon] = value
    return _sort_dict_deep(normalized)


def _normalize_filter_list_item(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _canonical_analysis_type(analysis_type: str) -> str:
    normalized = analysis_type.strip().lower()
    return ANALYSIS_TYPE_ALIASES.get(normalized, normalized)


def _sort_dict_deep(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_dict_deep(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_sort_dict_deep(item) for item in value]
    return value


def preprocessing_config_fingerprint(config: dict[str, Any] | None) -> str | None:
    """Stable hash of a resolved preprocessing configuration dict."""
    if not config:
        return None
    canonical = json.dumps(
        _sort_dict_deep(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
