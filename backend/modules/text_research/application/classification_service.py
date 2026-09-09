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
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TrainedModel, dumps, loads
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.classifiers import (
    FeatureConfig,
    extract_linear_coefficients,
    fit_text_classifier,
    flatten_single_label_targets,
    grouped_train_val_test_split,
    infer_task_type,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


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
        name: str | None = None,
        run_async: bool = False,
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
            "name": name,
            "split_strategy": "grouped_by_source_document",
        }
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
            from backend.modules.text_research.workers import queue_classifier_training

            queue_classifier_training(run_id=run.id, user_id=user_id)
        else:
            await self.execute_training(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_training(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
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
                texts,
                y,
                groups,
                test_size=params["test_size"],
                val_size=params.get("val_size", 0.2),
                random_seed=params["random_seed"],
            )

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

            # §31 hyperparameter tuning: small grid/random search over
            # C/alpha/max_features, scored on VALIDATION only (or
            # GroupKFold on train+val groups when no validation partition
            # exists). Never touches the held-out TEST partition. The best
            # params found are applied below before the final fit; the
            # complete search configuration + all candidate results are
            # persisted verbatim into the training results JSON.
            hyperparameter_search_results: dict[str, Any] | None = None
            applied_tuned_params: dict[str, Any] = {}
            if params.get("tune_hyperparameters"):
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
            fit_result = fit_text_classifier(
                split["X_train"],
                split["y_train"],
                split["X_test"],
                split["y_test"],
                task_type=task_type,
                algorithm=params["algorithm"],
                feature_config=feature_config,
                preprocessing_config=config,
                label_names=label_names if task_type == "multilabel" else None,
                class_weight=params["class_weight"],
                C=params["regularization_c"],
                random_seed=params["random_seed"],
                sgd_loss=params.get("sgd_loss", "log_loss"),
                nb_alpha=params.get("nb_alpha", 1.0),
                # §33/§35/§36: validation partition (threshold tuning +
                # optional calibration fit) and TEST group ids (bootstrap
                # CIs) — all additive, all no-ops with a persisted note
                # when their prerequisite data is unavailable.
                X_val_texts=split.get("X_val") or None,
                y_val=split.get("y_val") or None,
                groups_test=split["groups_test"],
                tune_thresholds=params.get("tune_thresholds", True),
                n_bootstrap=params.get("n_bootstrap", 200),
                ci_confidence_level=params.get("ci_confidence_level", 0.95),
                calibration_method=params.get("calibration_method", "sigmoid"),
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
                        {**config, "feature_config": fit_result["feature_config"]}
                    ),
                    training_config_json=dumps(params),
                    metrics_json=dumps(fit_result["metrics"]),
                    model_artifact_path=model_artifact_path,
                    vectorizer_artifact_path=vectorizer_artifact_path,
                    version=version,
                    name=params.get("name") or f"model-v{version}",
                    created_by=run.created_by,
                )
            )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(fit_result["metrics"]),
                results_json=dumps(
                    {
                        "trained_model_id": trained_model.id,
                        "task_type": task_type,
                        "n_train": fit_result["n_train"],
                        "n_val": len(split.get("X_val", [])),
                        "n_test": fit_result["n_test"],
                        "vocabulary_size": fit_result["vocabulary_size"],
                        "classes": fit_result["classes"],
                        "artifact_metadata": {
                            "model": model_artifact_metadata,
                            "vectorizer": vectorizer_artifact_metadata,
                        },
                        "train_groups": sorted(set(split["groups_train"])),
                        "val_groups": sorted(set(split.get("groups_val", []))),
                        "test_groups": sorted(set(split["groups_test"])),
                        "split_notes": split.get("notes", []),
                        # §31: complete search configuration + all
                        # candidate results (persisted verbatim, even when
                        # tuning was not requested — `None` in that case).
                        "hyperparameter_search": hyperparameter_search_results,
                        "applied_tuned_params": applied_tuned_params,
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
        self, *, project_id: str, user_id: str, corpus_id: str | None = None
    ) -> list[TrainedModel]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_models(project_id, corpus_id=corpus_id)

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
