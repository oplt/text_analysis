"""Versioned rerun adapters keyed by ``AnalysisRun.run_type``.

Each adapter validates persisted normalized parameters and either reconstructs
the original service call exactly or declares the run non-rerunnable with a
clear reason. ``RunService.rerun`` and run API responses use this registry so
“Reproduce” is scientifically faithful or explicitly unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun

ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class RerunCapability:
    """Whether a persisted run can be reproduced exactly."""

    rerunnable: bool
    block_reason: str | None = None
    adapter_version: str = ADAPTER_VERSION


class NonRerunnableError(ValueError):
    """Raised during normalization when exact reproduction is impossible."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _require(params: dict[str, Any], key: str, *, run_type: str) -> Any:
    if key not in params or params[key] is None:
        raise NonRerunnableError(
            f"Cannot reproduce {run_type}: required parameter '{key}' is missing "
            "from persisted run parameters."
        )
    return params[key]


def _filters(params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("filters")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise NonRerunnableError(
            "Cannot reproduce run: persisted 'filters' must be an object."
        )
    return dict(raw)


def _pick(params: dict[str, Any], key: str, default: Any) -> Any:
    return default if key not in params else params[key]


class RunAdapter(ABC):
    """Normalize persisted params and re-invoke the originating service."""

    run_type: str
    version: str = ADAPTER_VERSION

    def capability(self, params: dict[str, Any] | None) -> RerunCapability:
        try:
            self.normalize(params or {})
        except NonRerunnableError as exc:
            return RerunCapability(
                rerunnable=False,
                block_reason=exc.reason,
                adapter_version=self.version,
            )
        return RerunCapability(rerunnable=True, adapter_version=self.version)

    @abstractmethod
    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and return kwargs for the service method (excluding user_id/db)."""

    @abstractmethod
    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        ...


class UnsupportedRunAdapter(RunAdapter):
    """Explicit non-rerunnable registration with a stable reason."""

    def __init__(self, run_type: str, reason: str) -> None:
        self.run_type = run_type
        self._reason = reason

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NonRerunnableError(self._reason)

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        raise HTTPException(status_code=400, detail=self._reason)


# ---------------------------------------------------------------------------
# Classifier / topic / robustness (full parameter round-trip)
# ---------------------------------------------------------------------------


class ClassifierTrainingAdapter(RunAdapter):
    run_type = AnalysisRunType.CLASSIFIER_TRAINING.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = _require(params, "snapshot_id", run_type=self.run_type)
        return {
            "snapshot_id": snapshot_id,
            "algorithm": _pick(params, "algorithm", "logistic_regression"),
            "task_type": params.get("task_type"),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "vectorizer": _pick(params, "vectorizer", "tfidf"),
            "use_word_ngrams": _pick(params, "use_word_ngrams", True),
            "ngram_min": _pick(params, "ngram_min", 1),
            "ngram_max": _pick(params, "ngram_max", 1),
            "use_char_ngrams": _pick(params, "use_char_ngrams", False),
            "char_ngram_min": _pick(params, "char_ngram_min", 3),
            "char_ngram_max": _pick(params, "char_ngram_max", 5),
            "min_df": _pick(params, "min_df", 1),
            "max_df": _pick(params, "max_df", 1.0),
            "max_features": params.get("max_features"),
            "feature_selection_method": _pick(params, "feature_selection_method", "none"),
            "feature_selection_k": _pick(params, "feature_selection_k", "all"),
            "feature_selection_percentile": params.get("feature_selection_percentile"),
            "class_weight": params.get("class_weight"),
            "regularization_c": _pick(params, "regularization_c", 1.0),
            "nb_alpha": _pick(params, "nb_alpha", 1.0),
            "sgd_loss": _pick(params, "sgd_loss", "log_loss"),
            "test_size": _pick(params, "test_size", 0.2),
            "val_size": _pick(params, "val_size", 0.2),
            "random_seed": _pick(params, "random_seed", 42),
            "tune_hyperparameters": _pick(params, "tune_hyperparameters", False),
            "hyperparameter_search_type": _pick(params, "hyperparameter_search_type", "grid"),
            "hyperparameter_param_grid": params.get("hyperparameter_param_grid"),
            "hyperparameter_n_iter": _pick(params, "hyperparameter_n_iter", 10),
            "hyperparameter_scoring": _pick(params, "hyperparameter_scoring", "f1_macro"),
            "tune_thresholds": _pick(params, "tune_thresholds", True),
            "n_bootstrap": _pick(params, "n_bootstrap", 200),
            "ci_confidence_level": _pick(params, "ci_confidence_level", 0.95),
            "calibration_method": _pick(params, "calibration_method", "sigmoid"),
            "validation_strategy": _pick(params, "validation_strategy", "holdout"),
            "nested_cv_outer_splits": _pick(params, "nested_cv_outer_splits", 5),
            "nested_cv_inner_splits": _pick(params, "nested_cv_inner_splits", 3),
            "embedding_provider": _pick(params, "embedding_provider", "hashing"),
            "threshold_objective": _pick(params, "threshold_objective", "f1"),
            "threshold_utility_tp": _pick(params, "threshold_utility_tp", 1.0),
            "threshold_utility_tn": _pick(params, "threshold_utility_tn", 1.0),
            "threshold_utility_fp": _pick(params, "threshold_utility_fp", -1.0),
            "threshold_utility_fn": _pick(params, "threshold_utility_fn", -1.0),
            "name": params.get("name"),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.classification_service import (
            ClassificationService,
        )

        return await ClassificationService(db).train(
            user_id=user_id,
            run_async=run_async,
            **normalized,
        )


class TopicModelAdapter(RunAdapter):
    run_type = AnalysisRunType.TOPIC_MODEL.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        unit_type = _require(params, "unit_type", run_type=self.run_type)
        filters = _filters(params)
        return {
            "unit_type": unit_type,
            "algorithm": _pick(params, "algorithm", "lda"),
            "n_topics": _pick(params, "n_topics", 5),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "max_iterations": _pick(params, "max_iterations", 25),
            "random_seed": _pick(params, "random_seed", 42),
            "group_by": params.get("group_by"),
            "holdout_fraction": params.get("holdout_fraction"),
            "holdout_unit_ids": params.get("holdout_unit_ids"),
            "embedding_provider": params.get("embedding_provider"),
            "embedding_model_name": params.get("embedding_model_name"),
            "persist_embedding_artifacts": _pick(params, "persist_embedding_artifacts", True),
            "filters": filters,
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.topic_model_service import (
            TopicModelService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await TopicModelService(db).train(
            run.corpus_id,
            user_id=user_id,
            run_async=run_async,
            **normalized,
            **filters,
        )


class RobustnessAdapter(RunAdapter):
    run_type = AnalysisRunType.ROBUSTNESS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = _require(params, "snapshot_id", run_type=self.run_type)
        return {
            "snapshot_id": snapshot_id,
            "algorithm": _pick(params, "algorithm", "logistic_regression"),
            "seeds": params.get("seeds"),
            "cv_folds": _pick(params, "cv_folds", 5),
            "class_weights": params.get("class_weights"),
            "test_size": _pick(params, "test_size", 0.25),
            "group_field": _pick(params, "group_field", "organization"),
            "max_groups": _pick(params, "max_groups", 25),
            "temporal_field": _pick(params, "temporal_field", "publication_year"),
            "temporal_windows": _pick(params, "temporal_windows", False),
            "transfer_field": params.get("transfer_field"),
            "transfer_train_values": params.get("transfer_train_values"),
            "transfer_test_values": params.get("transfer_test_values"),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.robustness_service import (
            RobustnessService,
        )

        snapshot_id = normalized.pop("snapshot_id")
        return await RobustnessService(db).run_sweep(
            snapshot_id,
            user_id=user_id,
            run_async=run_async,
            **normalized,
        )


# ---------------------------------------------------------------------------
# Quantitative / other types with persisted inputs
# ---------------------------------------------------------------------------


class SegmentationAdapter(RunAdapter):
    run_type = AnalysisRunType.SEGMENTATION.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"unit_type": _require(params, "unit_type", run_type=self.run_type)}

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.segmentation_service import (
            SegmentationService,
        )

        return await SegmentationService(db).start_segmentation(
            run.corpus_id, user_id=user_id, unit_type=normalized["unit_type"]
        )


class ReliabilityAdapter(RunAdapter):
    run_type = AnalysisRunType.RELIABILITY.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "codebook_id": _require(params, "codebook_id", run_type=self.run_type),
            "label_ids": params.get("label_ids"),
            "campaign_id": params.get("campaign_id"),
            "unit_type": params.get("unit_type"),
            "annotator_ids": params.get("annotator_ids"),
            "bootstrap_samples": int(params.get("bootstrap_samples") or 2000),
            "confidence_level": float(params.get("confidence_level") or 0.95),
            "random_seed": params.get("random_seed"),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.reliability_service import (
            ReliabilityService,
        )

        return await ReliabilityService(db).compute_reliability(
            run.corpus_id, user_id=user_id, **normalized
        )


class CorpusStatsAdapter(RunAdapter):
    run_type = AnalysisRunType.CORPUS_STATS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).corpus_stats(
            run.corpus_id,
            user_id=user_id,
            unit_type=normalized["unit_type"],
            preprocessing_profile_id=normalized.get("preprocessing_profile_id"),
            **filters,
        )


class FrequencyAnalysisAdapter(RunAdapter):
    run_type = AnalysisRunType.FREQUENCY_ANALYSIS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "top_n": _pick(params, "top_n", 50),
            "rate_per": _pick(params, "rate_per", 1000),
            "group_by": params.get("group_by"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).frequencies(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class NgramAnalysisAdapter(RunAdapter):
    run_type = AnalysisRunType.NGRAM_ANALYSIS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "n": _pick(params, "n", 2),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "top_n": _pick(params, "top_n", 50),
            "rate_per": _pick(params, "rate_per", 1000),
            "skip": _pick(params, "skip", 0),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).ngrams(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class DfmAdapter(RunAdapter):
    run_type = AnalysisRunType.DFM.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "weighting": _pick(params, "weighting", "count"),
            "k1": params.get("k1"),
            "b": params.get("b"),
            "smooth_idf": params.get("smooth_idf"),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "force_sparse_only": _pick(params, "force_sparse_only", False),
            "trim": params.get("trim"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).dfm(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class KwicAdapter(RunAdapter):
    run_type = AnalysisRunType.KWIC.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "keyword": _require(params, "keyword", run_type=self.run_type),
            "window_size": _pick(params, "window_size", 5),
            "case_sensitive": _pick(params, "case_sensitive", False),
            "query_mode": _pick(params, "query_mode", "auto"),
            "query_language": params.get("query_language") or params.get("language"),
            "token_attribute": params.get("token_attribute"),
            "max_matches": params.get("max_matches"),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).kwic(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class DictionaryAnalysisAdapter(RunAdapter):
    run_type = AnalysisRunType.DICTIONARY_ANALYSIS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "dictionary_terms": params.get("dictionary_terms"),
            "dictionary_id": params.get("dictionary_id"),
            "hierarchy": params.get("hierarchy"),
            "exclusions": params.get("exclusions"),
            "dictionary_language": params.get("dictionary_language"),
            "case_sensitive": _pick(params, "case_sensitive", False),
            "rate_per": _pick(params, "rate_per", 1000.0),
            "group_by": params.get("group_by"),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).dictionary(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class KeynessAdapter(RunAdapter):
    run_type = AnalysisRunType.KEYNESS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "filters_a": _require(params, "filters_a", run_type=self.run_type),
            "filters_b": _require(params, "filters_b", run_type=self.run_type),
            "group_field": params.get("group_field"),
            "method": _pick(params, "method", "log_likelihood"),
            "correction": _pick(params, "correction", "bh"),
            "min_frequency": _pick(params, "min_frequency", 1),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "top_n": _pick(params, "top_n", 50),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        return await QuantitativeAnalysisService(db).keyness(
            run.corpus_id, user_id=user_id, **normalized
        )


class CooccurrenceAdapter(RunAdapter):
    run_type = AnalysisRunType.COOCCURRENCE.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "window_size": _pick(params, "window_size", 5),
            "top_n": _pick(params, "top_n", 50),
            "association_method": _pick(params, "association_method", "pmi"),
            "directional": _pick(params, "directional", False),
            "min_frequency": _pick(params, "min_frequency", 1),
            "min_count": _pick(params, "min_count", 1),
            "include_network": _pick(params, "include_network", True),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).cooccurrence(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class SimilarityAdapter(RunAdapter):
    """Lexical similarity is rerunnable; raw-vector embedding_cosine is not."""

    run_type = AnalysisRunType.SIMILARITY.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure import similarity as sim_mod

        method = _pick(params, "method", "tfidf_cosine")
        try:
            canonical = sim_mod.normalize_similarity_method(method)
        except ValueError as exc:
            raise NonRerunnableError(str(exc)) from exc

        has_artifact = bool(
            params.get("embedding_artifact_id") or params.get("embeddings_artifact_path")
        )
        uses_raw_vectors = canonical == "embedding_cosine" or bool(params.get("has_embeddings"))
        if uses_raw_vectors:
            if has_artifact:
                raise NonRerunnableError(
                    "Raw-vector embedding_cosine similarity cannot be reproduced yet: "
                    "embedding artifact loading for rerun is not implemented."
                )
            raise NonRerunnableError(
                "Raw-vector embedding_cosine similarity cannot be reproduced: "
                "original embeddings were not persisted as managed artifacts. "
                "Re-submit the analysis with the same vectors, or store them as "
                "an embedding artifact first."
            )

        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "method": canonical,
            "mode": _pick(params, "mode", "pairwise"),
            "top_k": _pick(params, "top_k", 20),
            "min_score": params.get("min_score"),
            "group_by": params.get("group_by"),
            "centroid_target": _pick(params, "centroid_target", "between_groups"),
            "query_text": params.get("query_text"),
            "query_unit_id": params.get("query_unit_id"),
            "preprocessing_profile_id": params.get("preprocessing_profile_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).similarity(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class DuplicateDetectionAdapter(RunAdapter):
    run_type = AnalysisRunType.DUPLICATE_DETECTION.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "methods": params.get("methods"),
            "lexical_threshold": _pick(params, "lexical_threshold", 0.85),
            "char_ngram_size": _pick(params, "char_ngram_size", 5),
            "use_minhash": _pick(params, "use_minhash", False),
            "minhash_num_perm": _pick(params, "minhash_num_perm", 64),
            "minhash_shingle_size": _pick(params, "minhash_shingle_size", 3),
            "minhash_threshold": _pick(params, "minhash_threshold", 0.8),
            "max_pairs": _pick(params, "max_pairs", 1000),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await QuantitativeAnalysisService(db).duplicate_detection(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


class ClassifierPredictionAdapter(RunAdapter):
    run_type = AnalysisRunType.CLASSIFIER_PREDICTION.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_id": _require(params, "model_id", run_type=self.run_type),
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "only_unannotated": _pick(params, "only_unannotated", False),
            "filters": params.get("filters"),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.prediction_service import (
            PredictionService,
        )

        model_id = normalized.pop("model_id")
        return await PredictionService(db).predict(
            model_id, user_id=user_id, **normalized
        )


class ComparativeAnalysisAdapter(RunAdapter):
    run_type = AnalysisRunType.COMPARATIVE_ANALYSIS.value

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_type": _require(params, "unit_type", run_type=self.run_type),
            "codebook_id": _require(params, "codebook_id", run_type=self.run_type),
            "label_ids": _require(params, "label_ids", run_type=self.run_type),
            "group_by": _require(params, "group_by", run_type=self.run_type),
            "provenance_mode": _pick(params, "provenance_mode", "human_only"),
            "model_id": params.get("model_id"),
            "filters": _filters(params),
        }

    async def execute(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        *,
        user_id: str,
        run_async: bool,
        normalized: dict[str, Any],
    ) -> AnalysisRun:
        from backend.modules.text_research.application.comparative_analysis_service import (
            ComparativeAnalysisService,
        )

        filters = dict(normalized.pop("filters") or {})
        return await ComparativeAnalysisService(db).prevalence_by_metadata(
            run.corpus_id, user_id=user_id, **normalized, **filters
        )


# ---------------------------------------------------------------------------
# Explicitly unsupported types
# ---------------------------------------------------------------------------

_UNSUPPORTED: dict[str, str] = {
    AnalysisRunType.CLUSTERING.value: (
        "Clustering runs are not rerunnable: exact reproduction is not registered "
        "until all clustering inputs are covered by a versioned adapter."
    ),
    AnalysisRunType.DIMENSIONALITY_REDUCTION.value: (
        "Dimensionality-reduction runs are not rerunnable: exact reproduction is "
        "not registered until all inputs are covered by a versioned adapter."
    ),
    AnalysisRunType.READABILITY.value: (
        "Readability runs are not rerunnable: exact reproduction is not registered "
        "until all inputs are covered by a versioned adapter."
    ),
    AnalysisRunType.STATISTICAL_MODEL.value: (
        "Statistical-model runs are not rerunnable: request rows are not persisted "
        "in a form that guarantees exact reproduction."
    ),
    AnalysisRunType.MEASUREMENT_VALIDATION.value: (
        "Measurement-validation runs are not rerunnable: input arrays are not "
        "persisted for exact reproduction."
    ),
    AnalysisRunType.DRIFT_MONITORING.value: (
        "Drift-monitoring runs are not rerunnable from the Runs UI."
    ),
    AnalysisRunType.PREPROCESSING.value: (
        "Preprocessing runs are not rerunnable from the Runs UI."
    ),
    AnalysisRunType.EXPORT.value: (
        "Export runs are not rerunnable from the Runs UI."
    ),
    AnalysisRunType.CORPUS_SYNTHESIS.value: (
        "Corpus-synthesis runs are not rerunnable from the Runs UI."
    ),
    AnalysisRunType.INGESTION_QA.value: (
        "Ingestion-QA runs are not rerunnable from the Runs UI."
    ),
    AnalysisRunType.DOCUMENT_CLEANING.value: (
        "Document-cleaning runs are not rerunnable from the Runs UI."
    ),
}


def _build_registry() -> dict[str, RunAdapter]:
    adapters: list[RunAdapter] = [
        ClassifierTrainingAdapter(),
        TopicModelAdapter(),
        RobustnessAdapter(),
        SegmentationAdapter(),
        ReliabilityAdapter(),
        CorpusStatsAdapter(),
        FrequencyAnalysisAdapter(),
        NgramAnalysisAdapter(),
        DfmAdapter(),
        KwicAdapter(),
        DictionaryAnalysisAdapter(),
        KeynessAdapter(),
        CooccurrenceAdapter(),
        SimilarityAdapter(),
        DuplicateDetectionAdapter(),
        ClassifierPredictionAdapter(),
        ComparativeAnalysisAdapter(),
    ]
    registry: dict[str, RunAdapter] = {adapter.run_type: adapter for adapter in adapters}
    for run_type, reason in _UNSUPPORTED.items():
        registry[run_type] = UnsupportedRunAdapter(run_type, reason)
    return registry


RUN_ADAPTERS: dict[str, RunAdapter] = _build_registry()


class _HasParams(Protocol):
    run_type: str
    parameters_json: Any


def describe_rerun_capability(
    run_type: str,
    params: dict[str, Any] | None = None,
) -> RerunCapability:
    adapter = RUN_ADAPTERS.get(run_type)
    if adapter is None:
        return RerunCapability(
            rerunnable=False,
            block_reason=f"Rerun is not supported for run_type '{run_type}'.",
            adapter_version=ADAPTER_VERSION,
        )
    return adapter.capability(params)


def capability_for_run(run: AnalysisRun | _HasParams, params: dict[str, Any] | None = None) -> RerunCapability:
    if params is None:
        from backend.modules.text_research.domain.models import loads

        params = loads(getattr(run, "parameters_json", None), {}) or {}
    return describe_rerun_capability(run.run_type, params)


async def execute_rerun(
    db: AsyncSession,
    run: AnalysisRun,
    *,
    user_id: str,
    run_async: bool = False,
    params: dict[str, Any] | None = None,
) -> AnalysisRun:
    """Validate via the registry and re-execute, or raise HTTP 400 with reason."""
    from backend.modules.text_research.domain.models import loads

    if params is None:
        params = loads(run.parameters_json, {}) or {}
    adapter = RUN_ADAPTERS.get(run.run_type)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Rerun is not supported for run_type '{run.run_type}'",
        )
    try:
        normalized = adapter.normalize(params)
    except NonRerunnableError as exc:
        raise HTTPException(status_code=400, detail=exc.reason) from exc
    return await adapter.execute(
        db,
        run,
        user_id=user_id,
        run_async=run_async,
        normalized=normalized,
    )
