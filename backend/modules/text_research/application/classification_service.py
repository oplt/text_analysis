"""Classifier training pipeline: grouped train/val/test split -> fit feature
extractor on TRAIN only -> fit classifier -> evaluate on TEST -> persist
artifacts + `TrainedModel`.

Leakage prevention is enforced end-to-end: the feature extractor is fit
exclusively on the training partition
(`infrastructure.classifiers.fit_text_classifier`) and the split is grouped
by source document (`infrastructure.classifiers.grouped_train_val_test_split`),
so no source document ever contributes text units to more than one
partition.

Task type (binary / multiclass / multilabel) is user/config-driven (see
§27): it is taken from the train request when given, and otherwise inferred
only from unambiguous label shape via `infrastructure.classifiers.infer_task_type`
— never silently forced to multilabel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.domain.enums import (
    AnalysisRunStatus,
    AnalysisRunType,
    ModelLifecycleStatus,
)
from backend.modules.text_research.domain.models import AnalysisRun, TrainedModel, dumps, loads
from backend.modules.text_research.infrastructure import classifiers, model_storage
from backend.modules.text_research.infrastructure.classifiers import (
    FeatureConfig,
    FeatureSelectionConfig,
    extract_linear_coefficients,
    fit_text_classifier,
    flatten_single_label_targets,
    grouped_train_val_test_split,
    infer_task_type,
)
from backend.modules.text_research.infrastructure.error_analysis import classifier_error_report
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts_cached_async,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.provenance import (
    merge_completion_provenance,
    partition_hash,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ClassificationService(ResearchAccessMixin):
    async def train(
        self,
        *,
        user_id: str,
        snapshot_id: str,
        algorithm: str = "logistic_regression",
        task_type: str | None = None,
        preprocessing_profile_id: str | None = None,
        vectorizer: str = "tfidf",
        use_word_ngrams: bool = True,
        ngram_min: int = 1,
        ngram_max: int = 1,
        use_char_ngrams: bool = False,
        char_ngram_min: int = 3,
        char_ngram_max: int = 5,
        min_df: float | int = 1,
        max_df: float | int = 1.0,
        max_features: int | None = None,
        feature_selection_method: str = "none",
        feature_selection_k: int | str = "all",
        feature_selection_percentile: float | None = None,
        class_weight: str | None = None,
        regularization_c: float = 1.0,
        nb_alpha: float = 1.0,
        sgd_loss: str = "log_loss",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_seed: int = 42,
        # §31 hyperparameter tuning: optional small grid/random search
        # evaluated on the VALIDATION set only (or GroupKFold on train+val
        # groups when no validation partition exists) — never on TEST.
        tune_hyperparameters: bool = False,
        hyperparameter_search_type: str = "grid",
        hyperparameter_param_grid: dict[str, list[Any]] | None = None,
        hyperparameter_n_iter: int = 10,
        hyperparameter_scoring: str = "f1_macro",
        # §33 per-class/per-label threshold tuning on VALIDATION only.
        tune_thresholds: bool = True,
        # §35 group-level bootstrap confidence intervals on TEST.
        n_bootstrap: int = 200,
        ci_confidence_level: float = 0.95,
        # §36 calibration: diagnostics always computed when proba is
        # available; recalibration method used only when a VAL set exists.
        calibration_method: str = "sigmoid",
        validation_strategy: str = "holdout",
        nested_cv_outer_splits: int = 5,
        nested_cv_inner_splits: int = 3,
        embedding_provider: str = "hashing",
        threshold_objective: str = "f1",
        threshold_utility_tp: float = 1.0,
        threshold_utility_tn: float = 1.0,
        threshold_utility_fp: float = -1.0,
        threshold_utility_fn: float = -1.0,
        name: str | None = None,
        run_async: bool = True,
    ) -> AnalysisRun:
        snapshot = await self.get_snapshot_or_404(snapshot_id, user_id=user_id)
        corpus = await self.get_corpus_or_404(snapshot.corpus_id, user_id=user_id)

        params = {
            "snapshot_id": snapshot_id,
            "algorithm": algorithm,
            # User/config-driven task type (§27): None means "infer from
            # label shape at training time" — never silently forced.
            "task_type": task_type,
            "preprocessing_profile_id": preprocessing_profile_id,
            "vectorizer": vectorizer,
            "use_word_ngrams": use_word_ngrams,
            "ngram_min": ngram_min,
            "ngram_max": ngram_max,
            "use_char_ngrams": use_char_ngrams,
            "char_ngram_min": char_ngram_min,
            "char_ngram_max": char_ngram_max,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "feature_selection_method": feature_selection_method,
            "feature_selection_k": feature_selection_k,
            "feature_selection_percentile": feature_selection_percentile,
            "class_weight": class_weight,
            "regularization_c": regularization_c,
            "nb_alpha": nb_alpha,
            "sgd_loss": sgd_loss,
            "test_size": test_size,
            "val_size": val_size,
            "random_seed": random_seed,
            "tune_hyperparameters": tune_hyperparameters,
            "hyperparameter_search_type": hyperparameter_search_type,
            "hyperparameter_param_grid": hyperparameter_param_grid,
            "hyperparameter_n_iter": hyperparameter_n_iter,
            "hyperparameter_scoring": hyperparameter_scoring,
            "tune_thresholds": tune_thresholds,
            "n_bootstrap": n_bootstrap,
            "ci_confidence_level": ci_confidence_level,
            "calibration_method": calibration_method,
            "validation_strategy": validation_strategy,
            "nested_cv_outer_splits": nested_cv_outer_splits,
            "nested_cv_inner_splits": nested_cv_inner_splits,
            "embedding_provider": embedding_provider,
            "threshold_objective": threshold_objective,
            "threshold_utility_tp": threshold_utility_tp,
            "threshold_utility_tn": threshold_utility_tn,
            "threshold_utility_fp": threshold_utility_fp,
            "threshold_utility_fn": threshold_utility_fn,
            "name": name,
            "split_strategy": "grouped_by_source_document",
        }
        validation_spec = (
            {
                "strategy": "nested_grouped_cv",
                "outer_splits": nested_cv_outer_splits,
                "inner_splits": nested_cv_inner_splits,
                "random_seed": random_seed,
            }
            if validation_strategy == "nested_grouped_cv"
            else {
                "strategy": "grouped_holdout",
                "test_size": test_size,
                "random_seed": random_seed,
            }
        )
        spec = build_spec_from_request(
            "classification",
            corpus.id,
            snapshot_id=snapshot_id,
            preprocessing_profile_id=preprocessing_profile_id,
            feature={
                "type": vectorizer,
                "ngram_range": (ngram_min, ngram_max),
                "min_df": min_df,
                "max_df": max_df,
                "max_features": max_features,
                "selection": {
                    "method": feature_selection_method,
                    "k": feature_selection_k,
                    "percentile": feature_selection_percentile,
                },
            },
            model={
                "family": algorithm,
                "task_type": task_type,
                "class_weight": class_weight,
                "hyperparameters": {
                    "regularization_c": regularization_c,
                    "nb_alpha": nb_alpha,
                    "sgd_loss": sgd_loss,
                },
            },
            validation=validation_spec,
            random_seed=random_seed,
        )
        params = attach_run_identity(params, spec)
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=AnalysisRunType.CLASSIFIER_TRAINING.value,
                status=AnalysisRunStatus.QUEUED.value,
                parameters_json=dumps(params),
                random_seed=random_seed,
                created_by=user_id,
            )
        )
        await self.db.commit()

        if run_async:
            from backend.modules.text_research.application.execution_service import ExecutionService

            await ExecutionService.submit(
                db=self.db, run=run, operation="classification", user_id=user_id
            )
        else:
            await self.execute_training(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_training(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="building_dataset",
            started_at=_utcnow(),
        )
        await self.db.commit()

        try:
            from backend.modules.text_research.application.run_lifecycle import (
                RunCancelledError,
                ensure_not_cancelled,
            )

            run = await ensure_not_cancelled(self.repo, run)
            snapshot = await self.repo.get_snapshot(params["snapshot_id"])
            if snapshot is None:
                raise ValueError("Training dataset snapshot no longer exists")
            label_names = DatasetBuilderService.label_names(snapshot)
            unit_labels = DatasetBuilderService.unit_labels(snapshot)
            unit_ids = loads(snapshot.unit_ids_json, [])
            units = await self.repo.list_text_units_by_ids(unit_ids)
            # preserve deterministic ordering
            units_by_id = {u.id: u for u in units}
            ordered_units = [units_by_id[uid] for uid in unit_ids if uid in units_by_id]

            texts = [u.text for u in ordered_units]
            groups = [u.corpus_document_id for u in ordered_units]
            y = [unit_labels.get(u.id, []) for u in ordered_units]

            config: dict = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(
                    params["preprocessing_profile_id"]
                )
                if profile is not None:
                    config.update(loads(profile.config_json, {}))
            config.update(
                {
                    "min_df": params["min_df"],
                    "max_df": params["max_df"],
                    "max_features": params["max_features"],
                }
            )

            prepared = await prepare_texts_cached_async(
                texts,
                config,
                corpus_id=run.corpus_id,
                unit_type="dataset_snapshot",
                unit_ids=[u.id for u in ordered_units],
                document_ids=[u.corpus_document_id for u in ordered_units],
                operation_config={"snapshot_id": params["snapshot_id"]},
            )
            prepared_texts = list(prepared.texts_joined)

            # Task type is user/config-driven (§27): honor an explicit
            # request as-is, otherwise infer ONLY from unambiguous label
            # shape. Never silently force multilabel.
            task_type = infer_task_type(y, requested=params.get("task_type"))
            if task_type != "multilabel":
                y = flatten_single_label_targets(y)

            await self.repo.update_run(run, progress_stage="splitting")
            await self.db.commit()
            run = await ensure_not_cancelled(self.repo, run)
            split = grouped_train_val_test_split(
                prepared_texts,
                y,
                groups,
                test_size=params["test_size"],
                val_size=params.get("val_size", 0.2),
                random_seed=params["random_seed"],
                task_type=task_type,
                grouping_variable="corpus_document_id",
            )

            feature_config = FeatureConfig(
                vectorizer=params.get("vectorizer", "tfidf"),
                use_word_ngrams=params.get("use_word_ngrams", True),
                ngram_min=params.get("ngram_min", 1),
                ngram_max=params["ngram_max"],
                use_char_ngrams=params.get("use_char_ngrams", False),
                char_ngram_min=params.get("char_ngram_min", 3),
                char_ngram_max=params.get("char_ngram_max", 5),
                min_df=params["min_df"],
                max_df=params["max_df"],
                max_features=params["max_features"],
            )
            selection_config = FeatureSelectionConfig(
                method=params.get("feature_selection_method", "none"),
                k=params.get("feature_selection_k", "all"),
                percentile=params.get("feature_selection_percentile"),
            )

            # §31 hyperparameter tuning: small grid/random search over
            # C/alpha/max_features, scored on VALIDATION only (or
            # GroupKFold on train+val groups when no validation partition
            # exists). Never touches the held-out TEST partition. The best
            # params found are applied below before the final fit; the
            # complete search configuration + all candidate results are
            # persisted verbatim into the training results JSON.
            hyperparameter_search_results: dict[str, Any] | None = None
            nested_cv_results: dict[str, Any] | None = None
            applied_tuned_params: dict[str, Any] = {}
            resolved_algorithm, feature_family = classifiers.resolve_classifier_algorithm(
                params["algorithm"]
            )
            if params.get("validation_strategy") == "nested_grouped_cv":
                nested_cv_results = classifiers.nested_grouped_cv_evaluation(
                    prepared_texts,
                    y,
                    groups,
                    task_type=task_type,
                    algorithm=params["algorithm"],
                    feature_config=feature_config,
                    preprocessing_config=config,
                    selection_config=selection_config,
                    label_names=label_names if task_type == "multilabel" else None,
                    class_weight=params["class_weight"],
                    C=params["regularization_c"],
                    random_seed=params["random_seed"],
                    outer_splits=params.get("nested_cv_outer_splits", 5),
                    inner_splits=params.get("nested_cv_inner_splits", 3),
                    tune_hyperparameters=params.get("tune_hyperparameters", False),
                    hyperparameter_param_grid=params.get("hyperparameter_param_grid"),
                    hyperparameter_scoring=params.get("hyperparameter_scoring", "f1_macro"),
                    embedding_provider=params.get("embedding_provider", "hashing"),
                )
            if (
                params.get("tune_hyperparameters")
                and params.get("validation_strategy") != "nested_grouped_cv"
            ):
                await self.repo.update_run(run, progress_stage="hyperparameter_search")
                await self.db.commit()
                run = await ensure_not_cancelled(self.repo, run)
                hyperparameter_search_results = classifiers.hyperparameter_search(
                    split["X_train"],
                    split["y_train"],
                    split["groups_train"],
                    split.get("X_val") or None,
                    split.get("y_val") or None,
                    task_type=task_type,
                    algorithm=params["algorithm"],
                    feature_config=feature_config,
                    preprocessing_config=config,
                    selection_config=selection_config,
                    label_names=label_names if task_type == "multilabel" else None,
                    class_weight=params["class_weight"],
                    C=params["regularization_c"],
                    sgd_loss=params.get("sgd_loss", "log_loss"),
                    random_seed=params["random_seed"],
                    param_grid=params.get("hyperparameter_param_grid"),
                    search_type=params.get("hyperparameter_search_type", "grid"),
                    n_iter=params.get("hyperparameter_n_iter", 10),
                    scoring=params.get("hyperparameter_scoring", "f1_macro"),
                )
                best_params = hyperparameter_search_results.get("best_params") or {}
                if "C" in best_params:
                    params["regularization_c"] = best_params["C"]
                    applied_tuned_params["regularization_c"] = best_params["C"]
                if "alpha" in best_params:
                    alpha_value = best_params["alpha"]
                    if params["algorithm"] == "sgd_classifier":
                        if alpha_value:
                            params["regularization_c"] = 1.0 / alpha_value
                            applied_tuned_params["regularization_c"] = params["regularization_c"]
                    else:
                        params["nb_alpha"] = alpha_value
                        applied_tuned_params["nb_alpha"] = alpha_value
                if "max_features" in best_params:
                    feature_config.max_features = best_params["max_features"]
                    config["max_features"] = best_params["max_features"]
                    applied_tuned_params["max_features"] = best_params["max_features"]
                for fc_field in ("ngram_max", "min_df", "max_df"):
                    if fc_field in best_params:
                        setattr(feature_config, fc_field, best_params[fc_field])
                        config[fc_field] = best_params[fc_field]
                        applied_tuned_params[fc_field] = best_params[fc_field]
                if "class_weight" in best_params:
                    params["class_weight"] = best_params["class_weight"]
                    applied_tuned_params["class_weight"] = best_params["class_weight"]

            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            fit_kwargs = {
                "task_type": task_type,
                "label_names": label_names if task_type == "multilabel" else None,
                "class_weight": params["class_weight"],
                "C": params["regularization_c"],
                "random_seed": params["random_seed"],
                "X_val_texts": split.get("X_val") or None,
                "y_val": split.get("y_val") or None,
                "groups_test": split["groups_test"],
                "tune_thresholds": params.get("tune_thresholds", True),
                "threshold_objective": params.get("threshold_objective", "f1"),
                "threshold_utility_tp": params.get("threshold_utility_tp", 1.0),
                "threshold_utility_tn": params.get("threshold_utility_tn", 1.0),
                "threshold_utility_fp": params.get("threshold_utility_fp", -1.0),
                "threshold_utility_fn": params.get("threshold_utility_fn", -1.0),
                "n_bootstrap": params.get("n_bootstrap", 200),
                "ci_confidence_level": params.get("ci_confidence_level", 0.95),
                "calibration_method": params.get("calibration_method", "sigmoid"),
            }
            if feature_family == "embedding":
                fit_result = classifiers.fit_embedding_text_classifier(
                    split["X_train"],
                    split["y_train"],
                    split["X_test"],
                    split["y_test"],
                    task_type,
                    algorithm=params["algorithm"],
                    embedding_provider=params.get("embedding_provider", "hashing"),
                    **fit_kwargs,
                )
            else:
                fit_result = fit_text_classifier(
                    split["X_train"],
                    split["y_train"],
                    split["X_test"],
                    split["y_test"],
                    task_type=task_type,
                    algorithm=resolved_algorithm,
                    feature_config=feature_config,
                    preprocessing_config=config,
                    selection_config=selection_config,
                    sgd_loss=params.get("sgd_loss", "log_loss"),
                    nb_alpha=params.get("nb_alpha", 1.0),
                    **fit_kwargs,
                )

            test_unit_ids = [unit_ids[i] for i in split["test_index"]]
            eval_data = fit_result.get("evaluation") or {}
            metadata_by_unit: dict[str, dict[str, Any]] = {}
            slice_fields = ["organization", "publication_year", "language", "region"]
            test_doc_ids = sorted(set(split["groups_test"]))
            if test_doc_ids:
                docs = await self.repo.list_documents_by_ids(test_doc_ids)
                doc_by_id = {doc.id: doc for doc in docs}
                for uid, doc_id in zip(test_unit_ids, split["groups_test"], strict=True):
                    doc = doc_by_id.get(doc_id)
                    if doc is None:
                        continue
                    metadata_by_unit[str(uid)] = {
                        "document_id": doc_id,
                        "title": doc.title,
                        "organization": doc.organization,
                        "publication_year": doc.publication_year,
                        "language": doc.language,
                        "region": doc.region,
                        "publication_type": doc.publication_type,
                    }

            error_analysis = classifier_error_report(
                unit_ids=test_unit_ids,
                y_true=eval_data.get("y_true", split["y_test"]),
                y_pred=eval_data.get("y_pred", []),
                y_proba=eval_data.get("y_proba"),
                groups=split["groups_test"],
                metadata_by_unit=metadata_by_unit or None,
                slice_fields=slice_fields if metadata_by_unit else None,
                label_names=fit_result["classes"],
            )

            await self.repo.update_run(run, progress_stage="saving")
            await self.db.commit()
            model_artifact_path, model_artifact_metadata = (
                model_storage.save_artifact_with_metadata(
                    fit_result["model"], category="research_classifiers"
                )
            )
            vectorizer_artifact_path, vectorizer_artifact_metadata = (
                model_storage.save_artifact_with_metadata(
                    fit_result["vectorizer"], category="research_vectorizers"
                )
            )

            version = await self.repo.next_model_version(run.corpus_id)
            trained_model = await self.repo.create_model(
                TrainedModel(
                    project_id=run.project_id,
                    corpus_id=run.corpus_id,
                    analysis_run_id=run.id,
                    training_dataset_snapshot_id=snapshot.id,
                    model_family=params["algorithm"],
                    task_type=task_type,
                    # Persist the actual fitted classes (label encoder / MLB
                    # classes_, in prediction-index order) rather than the
                    # raw codebook label set, so downstream consumers
                    # (coefficients, predictions) can decode model output by
                    # position even when the task type is binary/multiclass.
                    label_ids_json=dumps(fit_result["classes"]),
                    feature_config_json=dumps(
                        {
                            **config,
                            "feature_config": fit_result["feature_config"],
                            "feature_selection": fit_result.get("feature_selection"),
                            "feature_space": fit_result.get("feature_space"),
                            "training_snapshot_provenance": {
                                "annotation_campaign_id": snapshot.annotation_campaign_id,
                                "annotation_campaign_snapshot_hash": (
                                    snapshot.annotation_campaign_snapshot_hash
                                ),
                                "adjudication_policy": snapshot.adjudication_policy,
                                "gold_source": snapshot.gold_source,
                            },
                        }
                    ),
                    training_config_json=dumps(params),
                    metrics_json=dumps(fit_result["metrics"]),
                    model_artifact_path=model_artifact_path,
                    vectorizer_artifact_path=vectorizer_artifact_path,
                    version=version,
                    name=params.get("name") or f"model-v{version}",
                    lifecycle_status=ModelLifecycleStatus.CANDIDATE.value,
                    created_by=run.created_by,
                )
            )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(fit_result["metrics"]),
                parameters_json=dumps(
                    merge_completion_provenance(
                        params,
                        corpus_snapshot_id=snapshot.id,
                        corpus_snapshot_hash=prepared.corpus_checksum,
                        corpus_checksum=prepared.corpus_checksum,
                        pipeline_checksum=prepared.pipeline_checksum,
                        model_artifact_checksum=model_artifact_metadata.get("sha256"),
                        output_artifact_checksums=[
                            checksum
                            for checksum in (
                                model_artifact_metadata.get("sha256"),
                                vectorizer_artifact_metadata.get("sha256"),
                            )
                            if checksum
                        ],
                        split_hashes={
                            "train": partition_hash(sorted(set(split["groups_train"]))),
                            "validation": partition_hash(sorted(set(split.get("groups_val", [])))),
                            "test": partition_hash(sorted(set(split["groups_test"]))),
                        },
                        feature_configuration=fit_result.get("feature_config"),
                        feature_selection_configuration=fit_result.get("feature_selection"),
                        algorithm=params.get("algorithm"),
                        hyperparameters=params.get("hyperparameters")
                        or params.get("model_hyperparameters"),
                        validation_strategy=params.get("split_strategy")
                        or "grouped_by_source_document",
                        campaign_id=snapshot.annotation_campaign_id,
                        extra={
                            "training_snapshot_campaign_hash": (
                                snapshot.annotation_campaign_snapshot_hash
                            ),
                            "adjudication_policy": snapshot.adjudication_policy,
                            "gold_source": snapshot.gold_source,
                        },
                    )
                ),
                results_json=dumps(
                    {
                        "trained_model_id": trained_model.id,
                        "task_type": task_type,
                        "n_train": fit_result["n_train"],
                        "n_val": len(split.get("X_val", [])),
                        "n_test": fit_result["n_test"],
                        "vocabulary_size": fit_result["vocabulary_size"],
                        "feature_space": fit_result.get("feature_space"),
                        "feature_selection": fit_result.get("feature_selection"),
                        "scientific_warnings": fit_result.get("scientific_warnings") or [],
                        "classes": fit_result["classes"],
                        "artifact_metadata": {
                            "model": model_artifact_metadata,
                            "vectorizer": vectorizer_artifact_metadata,
                        },
                        "train_groups": sorted(set(split["groups_train"])),
                        "val_groups": sorted(set(split.get("groups_val", []))),
                        "test_groups": sorted(set(split["groups_test"])),
                        "split_notes": split.get("notes", []),
                        "split_feasibility": split.get("split_feasibility"),
                        "split_strategy": split.get("split_strategy"),
                        "split_task_type": split.get("task_type"),
                        "grouping_variable": split.get("grouping_variable"),
                        "stratification_requested": split.get("stratification_requested"),
                        "stratification_applied": split.get("stratification_applied"),
                        "split_reason": split.get("reason"),
                        "corpus_checksum": prepared.corpus_checksum,
                        "pipeline_checksum": prepared.pipeline_checksum,
                        "analysis_spec_hash": params.get("analysis_spec_hash"),
                        # §31: complete search configuration + all
                        # candidate results (persisted verbatim, even when
                        # tuning was not requested — `None` in that case).
                        "hyperparameter_search": hyperparameter_search_results,
                        "nested_grouped_cv": nested_cv_results,
                        "applied_tuned_params": applied_tuned_params,
                        "feature_family": feature_family,
                        "error_analysis": error_analysis,
                    }
                ),
            )
            await self.db.commit()
        except RunCancelledError:
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.CANCELLED.value,
                progress_stage="cancelled",
                completed_at=_utcnow(),
                error_message="Cancelled by user",
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise

        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def list_models(
        self,
        *,
        project_id: str,
        user_id: str,
        corpus_id: str | None = None,
        lifecycle_status: str | None = None,
    ) -> list[TrainedModel]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_models(
            project_id, corpus_id=corpus_id, lifecycle_status=lifecycle_status
        )

    async def get_model(self, model_id: str, *, user_id: str) -> TrainedModel:
        return await self.get_model_or_404(model_id, user_id=user_id)

    async def get_coefficients(
        self, model_id: str, *, user_id: str, top_n: int = 25
    ) -> list[dict[str, Any]]:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        vectorizer = model_storage.load_artifact(model.vectorizer_artifact_path)
        classifier = model_storage.load_artifact(model.model_artifact_path)
        label_names = loads(model.label_ids_json, [])
        return extract_linear_coefficients(classifier, vectorizer, label_names, top_n=top_n)

    async def clone_config(self, model_id: str, *, user_id: str) -> dict[str, Any]:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        return {
            "training_dataset_snapshot_id": model.training_dataset_snapshot_id,
            "algorithm": model.model_family,
            "feature_config": loads(model.feature_config_json, {}),
            "training_config": loads(model.training_config_json, {}),
        }
