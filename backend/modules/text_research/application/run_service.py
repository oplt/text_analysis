"""List/get/clone/rerun for `AnalysisRun` records.

`rerun` re-executes the persisted parameters through the *same* application
service that originally produced the run, so results are always freshly
computed from current data rather than copied — this matters because
annotations, documents, or models may have changed since the original run.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, loads


class RunService(ResearchAccessMixin):
    async def list_runs(
        self,
        *,
        project_id: str,
        user_id: str,
        corpus_id: str | None = None,
        run_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AnalysisRun], int]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_runs(
            project_id, corpus_id=corpus_id, run_type=run_type, limit=limit, offset=offset
        )

    async def get_run(self, run_id: str, *, user_id: str) -> AnalysisRun:
        return await self.get_run_or_404(run_id, user_id=user_id)

    async def clone_parameters(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.provenance import (
            extract_reproduce_request,
        )

        run = await self.get_run_or_404(run_id, user_id=user_id)
        parameters = loads(run.parameters_json, {})
        reproduce = extract_reproduce_request(
            parameters, run_type=run.run_type, run_id=run.id
        )
        return {
            "run_type": run.run_type,
            "corpus_id": run.corpus_id,
            "parameters": parameters,
            "reproduce": reproduce,
            "analysis_specification": reproduce.get("analysis_specification"),
            "analysis_spec_hash": reproduce.get("analysis_spec_hash"),
        }

    async def get_provenance(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.provenance import (
            extract_reproduce_request,
            runtime_environment,
        )

        run = await self.get_run_or_404(run_id, user_id=user_id)
        parameters = loads(run.parameters_json, {})
        provenance = parameters.get("provenance") if isinstance(parameters.get("provenance"), dict) else {}
        return {
            "run_id": run.id,
            "run_type": run.run_type,
            "status": run.status,
            "random_seed": run.random_seed,
            "artifact_path": run.artifact_path,
            "provenance": provenance,
            "reproduce": extract_reproduce_request(
                parameters, run_type=run.run_type, run_id=run.id
            ),
            "runtime_now": runtime_environment(),
        }

    async def rerun(self, run_id: str, *, user_id: str, run_async: bool = False) -> AnalysisRun:
        """One-click reproducible re-execution using the original run parameters."""
        run = await self.get_run_or_404(run_id, user_id=user_id)
        params = loads(run.parameters_json, {})
        inner_filters = dict(params.get("filters") or {})

        if run.run_type == AnalysisRunType.SEGMENTATION.value:
            from backend.modules.text_research.application.segmentation_service import (
                SegmentationService,
            )

            return await SegmentationService(self.db).start_segmentation(
                run.corpus_id, user_id=user_id, unit_type=params["unit_type"]
            )

        if run.run_type == AnalysisRunType.RELIABILITY.value:
            from backend.modules.text_research.application.reliability_service import (
                ReliabilityService,
            )

            return await ReliabilityService(self.db).compute_reliability(
                run.corpus_id,
                user_id=user_id,
                codebook_id=params["codebook_id"],
                label_ids=params.get("label_ids"),
            )

        if run.run_type == AnalysisRunType.CORPUS_STATS.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).corpus_stats(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.FREQUENCY_ANALYSIS.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).frequencies(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                top_n=params.get("top_n", 50),
                rate_per=params.get("rate_per", 1000),
                group_by=params.get("group_by"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.NGRAM_ANALYSIS.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).ngrams(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                n=params.get("n", 2),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                top_n=params.get("top_n", 50),
                rate_per=params.get("rate_per", 1000),
                skip=params.get("skip", 0),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.DFM.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).dfm(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                weighting=params.get("weighting", "count"),
                k1=params.get("k1"),
                b=params.get("b"),
                smooth_idf=params.get("smooth_idf"),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                force_sparse_only=params.get("force_sparse_only", False),
                trim=params.get("trim"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.KWIC.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).kwic(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                keyword=params["keyword"],
                window_size=params.get("window_size", 5),
                case_sensitive=params.get("case_sensitive", False),
                query_mode=params.get("query_mode", "auto"),
                language=params.get("language"),
                token_attribute=params.get("token_attribute"),
                max_matches=params.get("max_matches"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.DICTIONARY_ANALYSIS.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).dictionary(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                dictionary_terms=params.get("dictionary_terms"),
                dictionary_id=params.get("dictionary_id"),
                hierarchy=params.get("hierarchy"),
                exclusions=params.get("exclusions"),
                dictionary_language=params.get("dictionary_language"),
                case_sensitive=params.get("case_sensitive", False),
                rate_per=params.get("rate_per", 1000.0),
                group_by=params.get("group_by"),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.KEYNESS.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).keyness(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                filters_a=params["filters_a"],
                filters_b=params["filters_b"],
                group_field=params.get("group_field"),
                method=params.get("method", "log_likelihood"),
                correction=params.get("correction", "bh"),
                min_frequency=params.get("min_frequency", 1),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                top_n=params.get("top_n", 50),
            )

        if run.run_type == AnalysisRunType.COOCCURRENCE.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).cooccurrence(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                window_size=params.get("window_size", 5),
                top_n=params.get("top_n", 50),
                association_method=params.get("association_method", "pmi"),
                directional=params.get("directional", False),
                min_frequency=params.get("min_frequency", 1),
                min_count=params.get("min_count", 1),
                include_network=params.get("include_network", True),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.SIMILARITY.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).similarity(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                method=params.get("method", "tfidf_cosine"),
                mode=params.get("mode", "pairwise"),
                top_k=params.get("top_k", 20),
                min_score=params.get("min_score"),
                group_by=params.get("group_by"),
                centroid_target=params.get("centroid_target", "between_groups"),
                query_text=params.get("query_text"),
                query_unit_id=params.get("query_unit_id"),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.DUPLICATE_DETECTION.value:
            from backend.modules.text_research.application.quantitative_analysis_service import (
                QuantitativeAnalysisService,
            )

            return await QuantitativeAnalysisService(self.db).duplicate_detection(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                methods=params.get("methods"),
                lexical_threshold=params.get("lexical_threshold", 0.85),
                char_ngram_size=params.get("char_ngram_size", 5),
                use_minhash=params.get("use_minhash", False),
                minhash_num_perm=params.get("minhash_num_perm", 64),
                minhash_shingle_size=params.get("minhash_shingle_size", 3),
                minhash_threshold=params.get("minhash_threshold", 0.8),
                max_pairs=params.get("max_pairs", 1000),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.TOPIC_MODEL.value:
            from backend.modules.text_research.application.topic_model_service import (
                TopicModelService,
            )

            return await TopicModelService(self.db).train(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                algorithm=params.get("algorithm", "lda"),
                n_topics=params.get("n_topics", 5),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                max_iterations=params.get("max_iterations", 25),
                random_seed=params.get("random_seed", 42),
                group_by=params.get("group_by"),
                run_async=run_async,
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.CLASSIFIER_TRAINING.value:
            from backend.modules.text_research.application.classification_service import (
                ClassificationService,
            )

            return await ClassificationService(self.db).train(
                user_id=user_id,
                snapshot_id=params["snapshot_id"],
                algorithm=params.get("algorithm", "logistic_regression"),
                task_type=params.get("task_type"),
                preprocessing_profile_id=params.get("preprocessing_profile_id"),
                vectorizer=params.get("vectorizer", "tfidf"),
                use_word_ngrams=params.get("use_word_ngrams", True),
                ngram_min=params.get("ngram_min", 1),
                ngram_max=params.get("ngram_max", 1),
                use_char_ngrams=params.get("use_char_ngrams", False),
                char_ngram_min=params.get("char_ngram_min", 3),
                char_ngram_max=params.get("char_ngram_max", 5),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=params.get("max_features"),
                class_weight=params.get("class_weight"),
                regularization_c=params.get("regularization_c", 1.0),
                sgd_loss=params.get("sgd_loss", "log_loss"),
                test_size=params.get("test_size", 0.2),
                val_size=params.get("val_size", 0.2),
                random_seed=params.get("random_seed", 42),
                name=params.get("name"),
                run_async=run_async,
            )

        if run.run_type == AnalysisRunType.CLASSIFIER_PREDICTION.value:
            from backend.modules.text_research.application.prediction_service import (
                PredictionService,
            )

            return await PredictionService(self.db).predict(
                params["model_id"],
                user_id=user_id,
                unit_type=params["unit_type"],
                only_unannotated=params.get("only_unannotated", False),
                filters=params.get("filters"),
            )

        if run.run_type == AnalysisRunType.COMPARATIVE_ANALYSIS.value:
            from backend.modules.text_research.application.comparative_analysis_service import (
                ComparativeAnalysisService,
            )

            return await ComparativeAnalysisService(self.db).prevalence_by_metadata(
                run.corpus_id,
                user_id=user_id,
                unit_type=params["unit_type"],
                codebook_id=params["codebook_id"],
                label_ids=params["label_ids"],
                group_by=params["group_by"],
                provenance_mode=params.get("provenance_mode", "human_only"),
                model_id=params.get("model_id"),
                **inner_filters,
            )

        if run.run_type == AnalysisRunType.ROBUSTNESS.value:
            from backend.modules.text_research.application.robustness_service import (
                RobustnessService,
            )

            return await RobustnessService(self.db).run_sweep(
                params["snapshot_id"],
                user_id=user_id,
                algorithm=params.get("algorithm", "logistic_regression"),
                seeds=params.get("seeds"),
                cv_folds=params.get("cv_folds", 5),
                class_weights=params.get("class_weights"),
                test_size=params.get("test_size", 0.25),
                run_async=run_async,
            )

        raise HTTPException(
            status_code=400,
            detail=f"Rerun is not supported for run_type '{run.run_type}'",
        )

    async def cancel_run(self, run_id: str, *, user_id: str) -> AnalysisRun:
        from datetime import UTC, datetime

        from backend.modules.text_research.domain.enums import AnalysisRunStatus

        run = await self.get_run_or_404(run_id, user_id=user_id)
        if run.status not in {
            AnalysisRunStatus.QUEUED.value,
            AnalysisRunStatus.RUNNING.value,
            "pending",
        }:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot cancel run in status '{run.status}'",
            )
        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.CANCELLED.value,
            cancellation_requested=True,
            progress_stage="cancelled",
            completed_at=datetime.now(UTC),
            error_message="Cancelled by user",
        )
        if run.artifact_namespace:
            from backend.modules.text_research.infrastructure.model_storage import ARTIFACT_ROOT

            namespace = ARTIFACT_ROOT / run.artifact_namespace
            if namespace.is_dir():
                import shutil

                shutil.rmtree(namespace)
        await self.db.commit()
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def compare_runs(
        self, run_a_id: str, run_b_id: str, *, user_id: str
    ) -> dict[str, Any]:
        run_a = await self.get_run_or_404(run_a_id, user_id=user_id)
        run_b = await self.get_run_or_404(run_b_id, user_id=user_id)
        params_a = loads(run_a.parameters_json, {}) or {}
        params_b = loads(run_b.parameters_json, {}) or {}
        metrics_a = loads(run_a.metrics_json, {}) or {}
        metrics_b = loads(run_b.metrics_json, {}) or {}

        param_keys = sorted(set(params_a) | set(params_b))
        metric_keys = sorted(set(metrics_a) | set(metrics_b))

        def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
            else:
                out[prefix] = value

        flat_a: dict[str, Any] = {}
        flat_b: dict[str, Any] = {}
        flatten("", params_a, flat_a)
        flatten("", params_b, flat_b)
        all_param_keys = sorted(set(flat_a) | set(flat_b))

        parameter_diff = [
            {
                "parameter": key,
                "run_a": flat_a.get(key),
                "run_b": flat_b.get(key),
                "changed": flat_a.get(key) != flat_b.get(key),
            }
            for key in all_param_keys
        ]
        metric_diff = [
            {
                "metric": key,
                "run_a": metrics_a.get(key),
                "run_b": metrics_b.get(key),
                "changed": metrics_a.get(key) != metrics_b.get(key),
            }
            for key in metric_keys
        ]
        return {
            "run_a": {
                "id": run_a.id,
                "run_type": run_a.run_type,
                "status": run_a.status,
                "created_at": run_a.created_at.isoformat() if run_a.created_at else None,
                "artifact_path": run_a.artifact_path,
                "random_seed": run_a.random_seed,
            },
            "run_b": {
                "id": run_b.id,
                "run_type": run_b.run_type,
                "status": run_b.status,
                "created_at": run_b.created_at.isoformat() if run_b.created_at else None,
                "artifact_path": run_b.artifact_path,
                "random_seed": run_b.random_seed,
            },
            "parameter_diff": parameter_diff,
            "metric_diff": metric_diff,
            "changed_parameters": [row for row in parameter_diff if row["changed"]],
            "changed_metrics": [row for row in metric_diff if row["changed"]],
        }
