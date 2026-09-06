"""Classifier training pipeline: grouped split -> fit TF-IDF on TRAIN only ->
fit classifier -> evaluate on TEST -> persist artifacts + `TrainedModel`.

Leakage prevention is enforced end-to-end: the vectorizer is fit exclusively
on the training partition (`infrastructure.classifiers.fit_tfidf_classifier`)
and the split is grouped by source document
(`infrastructure.classifiers.grouped_train_test_split`), so no source
document ever contributes text units to both partitions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TrainedModel, dumps, loads
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure.classifiers import (
    extract_linear_coefficients,
    fit_tfidf_classifier,
    grouped_train_test_split,
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
        preprocessing_profile_id: str | None = None,
        ngram_max: int = 1,
        min_df: float | int = 1,
        max_df: float | int = 1.0,
        max_features: int | None = None,
        class_weight: str | None = None,
        regularization_c: float = 1.0,
        test_size: float = 0.25,
        random_seed: int = 42,
        name: str | None = None,
        run_async: bool = False,
    ) -> AnalysisRun:
        snapshot = await self.get_snapshot_or_404(snapshot_id, user_id=user_id)
        corpus = await self.get_corpus_or_404(snapshot.corpus_id, user_id=user_id)

        params = {
            "snapshot_id": snapshot_id,
            "algorithm": algorithm,
            "preprocessing_profile_id": preprocessing_profile_id,
            "ngram_max": ngram_max,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "class_weight": class_weight,
            "regularization_c": regularization_c,
            "test_size": test_size,
            "random_seed": random_seed,
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
            run, status=AnalysisRunStatus.RUNNING.value, progress_stage="building_dataset", started_at=_utcnow()
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

            await self.repo.update_run(run, progress_stage="splitting")
            await self.db.commit()
            run = await ensure_not_cancelled(self.repo, run)
            split = grouped_train_test_split(
                texts, y, groups, test_size=params["test_size"], random_seed=params["random_seed"]
            )

            config: dict = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(params["preprocessing_profile_id"])
                if profile is not None:
                    config.update(loads(profile.config_json, {}))
            config.update(
                {
                    "ngram_min": 1,
                    "ngram_max": params["ngram_max"],
                    "min_df": params["min_df"],
                    "max_df": params["max_df"],
                    "max_features": params["max_features"],
                }
            )

            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            fit_result = fit_tfidf_classifier(
                split["X_train"],
                split["y_train"],
                split["X_test"],
                split["y_test"],
                task_type="multilabel",
                algorithm=params["algorithm"],
                preprocessing_config=config,
                label_names=label_names,
                class_weight=params["class_weight"],
                C=params["regularization_c"],
                random_seed=params["random_seed"],
            )

            await self.repo.update_run(run, progress_stage="saving")
            await self.db.commit()
            model_artifact_path, model_artifact_metadata = model_storage.save_artifact_with_metadata(
                fit_result["model"], category="research_classifiers"
            )
            vectorizer_artifact_path, vectorizer_artifact_metadata = model_storage.save_artifact_with_metadata(
                fit_result["vectorizer"], category="research_vectorizers"
            )

            version = await self.repo.next_model_version(run.corpus_id)
            trained_model = await self.repo.create_model(
                TrainedModel(
                    project_id=run.project_id,
                    corpus_id=run.corpus_id,
                    analysis_run_id=run.id,
                    training_dataset_snapshot_id=snapshot.id,
                    model_family=params["algorithm"],
                    task_type="multilabel",
                    label_ids_json=dumps(label_names),
                    feature_config_json=dumps(config),
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
                        "n_train": fit_result["n_train"],
                        "n_test": fit_result["n_test"],
                        "vocabulary_size": fit_result["vocabulary_size"],
                        "classes": fit_result["classes"],
                        "artifact_metadata": {
                            "model": model_artifact_metadata,
                            "vectorizer": vectorizer_artifact_metadata,
                        },
                        "train_groups": sorted(set(split["groups_train"])),
                        "test_groups": sorted(set(split["groups_test"])),
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
                run, status=AnalysisRunStatus.FAILED.value, completed_at=_utcnow(), error_message=str(exc)
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
