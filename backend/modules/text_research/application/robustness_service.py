"""Robustness testing: repeated-seed stability, grouped cross-validation,
preprocessing sensitivity, class-weight sensitivity, leave-one-organization-
out, and temporal holdout — all computed against a frozen
`TrainingDatasetSnapshot`, never against mutable annotations.

All robustness runs execute as a single `AnalysisRun` (optionally dispatched
to Celery for larger sweeps) so results are persisted and auditable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.classifiers import (
    fit_tfidf_classifier,
    grouped_train_test_split,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _train_eval(
    texts: list[str],
    y: list[list[str]],
    groups: list[str],
    *,
    label_names: list[str],
    config: dict,
    algorithm: str,
    class_weight: str | None,
    regularization_c: float,
    random_seed: int,
    test_size: float,
) -> dict[str, Any]:
    split = grouped_train_test_split(texts, y, groups, test_size=test_size, random_seed=random_seed)
    result = fit_tfidf_classifier(
        split["X_train"],
        split["y_train"],
        split["X_test"],
        split["y_test"],
        task_type="multilabel",
        algorithm=algorithm,
        preprocessing_config=config,
        label_names=label_names,
        class_weight=class_weight,
        C=regularization_c,
        random_seed=random_seed,
    )
    return result["metrics"]


def _safe_train_eval(
    texts: list[str],
    y: list[list[str]],
    groups: list[str],
    *,
    label_names: list[str],
    config: dict,
    algorithm: str,
    class_weight: str | None,
    regularization_c: float,
    random_seed: int,
    test_size: float,
) -> dict[str, Any]:
    try:
        metrics = _train_eval(
            texts,
            y,
            groups,
            label_names=label_names,
            config=config,
            algorithm=algorithm,
            class_weight=class_weight,
            regularization_c=regularization_c,
            random_seed=random_seed,
            test_size=test_size,
        )
        return {"status": "ok", "macro_f1": metrics["f1_macro"], "metrics": metrics}
    except Exception as exc:  # noqa: BLE001 — per-check resilience
        return {"status": "not_evaluable", "reason": str(exc), "macro_f1": None}


def _safe_fit(
    x_train: list[str],
    y_train: list[list[str]],
    x_test: list[str],
    y_test: list[list[str]],
    *,
    label_names: list[str],
    config: dict,
    algorithm: str,
    class_weight: str | None,
    regularization_c: float,
    random_seed: int,
) -> dict[str, Any]:
    if not x_train or not x_test:
        return {
            "status": "not_evaluable",
            "reason": "Empty train or test split",
            "macro_f1": None,
        }
    try:
        fit_result = fit_tfidf_classifier(
            x_train,
            y_train,
            x_test,
            y_test,
            task_type="multilabel",
            algorithm=algorithm,
            preprocessing_config=config,
            label_names=label_names,
            class_weight=class_weight,
            C=regularization_c,
            random_seed=random_seed,
        )
        metrics = fit_result["metrics"]
        return {
            "status": "ok",
            "macro_f1": metrics["f1_macro"],
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001 — per-check resilience
        return {"status": "not_evaluable", "reason": str(exc), "macro_f1": None}


class RobustnessService(ResearchAccessMixin):
    async def _load_snapshot_data(self, snapshot_id: str, *, user_id: str):
        snapshot = await self.get_snapshot_or_404(snapshot_id, user_id=user_id)
        label_names = DatasetBuilderService.label_names(snapshot)
        unit_labels = DatasetBuilderService.unit_labels(snapshot)
        unit_ids = loads(snapshot.unit_ids_json, [])
        units = await self.repo.list_text_units_by_ids(unit_ids)
        units_by_id = {u.id: u for u in units}
        ordered_units = [units_by_id[uid] for uid in unit_ids if uid in units_by_id]
        documents = await self.repo.list_documents(snapshot.corpus_id)
        documents_by_id = {d.id: d for d in documents}
        return snapshot, label_names, unit_labels, ordered_units, documents_by_id

    async def run_sweep(
        self,
        snapshot_id: str,
        *,
        user_id: str,
        algorithm: str = "logistic_regression",
        seeds: list[int] | None = None,
        cv_folds: int = 5,
        class_weights: list[str | None] | None = None,
        test_size: float = 0.25,
        run_async: bool = False,
    ) -> AnalysisRun:
        snapshot = await self.get_snapshot_or_404(snapshot_id, user_id=user_id)
        params = {
            "snapshot_id": snapshot_id,
            "algorithm": algorithm,
            "seeds": seeds or [1, 11, 21, 42, 84],
            "cv_folds": cv_folds,
            "class_weights": class_weights if class_weights is not None else [None, "balanced"],
            "test_size": test_size,
        }
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=snapshot.project_id,
                corpus_id=snapshot.corpus_id,
                run_type=AnalysisRunType.ROBUSTNESS.value,
                status=AnalysisRunStatus.QUEUED.value,
                parameters_json=dumps(params),
                created_by=user_id,
            )
        )
        await self.db.commit()

        if run_async:
            from backend.modules.text_research.workers import queue_robustness_sweep

            queue_robustness_sweep(run_id=run.id, user_id=user_id)
        else:
            await self.execute_sweep(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_sweep(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        params = loads(run.parameters_json, {})

        await self.repo.update_run(run, status=AnalysisRunStatus.RUNNING.value, started_at=_utcnow())
        await self.db.commit()

        try:
            _snapshot, label_names, unit_labels, units, documents_by_id = await self._load_snapshot_data(
                params["snapshot_id"], user_id=run.created_by
            )
            texts = [u.text for u in units]
            y = [unit_labels.get(u.id, []) for u in units]
            groups = [u.corpus_document_id for u in units]
            base_config = PreprocessingConfig().to_dict()

            results: dict[str, Any] = {}

            # 1. Repeated-seed stability
            seed_runs = []
            for seed in params["seeds"]:
                outcome = _safe_train_eval(
                    texts, y, groups,
                    label_names=label_names, config=base_config, algorithm=params["algorithm"],
                    class_weight=None, regularization_c=1.0, random_seed=seed, test_size=params["test_size"],
                )
                seed_runs.append({"seed": seed, **outcome})
            seed_f1s = [r["macro_f1"] for r in seed_runs if r.get("status") == "ok" and r.get("macro_f1") is not None]
            results["seed_stability"] = {
                "runs": seed_runs,
                "mean_macro_f1": float(np.mean(seed_f1s)) if seed_f1s else None,
                "std_macro_f1": float(np.std(seed_f1s)) if seed_f1s else None,
                "min_macro_f1": float(np.min(seed_f1s)) if seed_f1s else None,
                "max_macro_f1": float(np.max(seed_f1s)) if seed_f1s else None,
            }

            # 2. Grouped cross-validation
            n_unique_groups = len(set(groups))
            fold_count = max(2, min(params["cv_folds"], n_unique_groups))
            fold_results = []
            if n_unique_groups < 2:
                fold_results.append(
                    {
                        "fold": 0,
                        "status": "not_evaluable",
                        "reason": "Insufficient groups for grouped cross-validation",
                        "macro_f1": None,
                    }
                )
            else:
                gkf = GroupKFold(n_splits=fold_count)
                for fold_idx, (train_idx, test_idx) in enumerate(
                    gkf.split(np.zeros(len(texts)), groups=groups)
                ):
                    outcome = _safe_fit(
                        [texts[i] for i in train_idx],
                        [y[i] for i in train_idx],
                        [texts[i] for i in test_idx],
                        [y[i] for i in test_idx],
                        label_names=label_names,
                        config=base_config,
                        algorithm=params["algorithm"],
                        class_weight=None,
                        regularization_c=1.0,
                        random_seed=42,
                    )
                    fold_results.append({"fold": fold_idx, **outcome})
            fold_f1s = [r["macro_f1"] for r in fold_results if r.get("status") == "ok" and r.get("macro_f1") is not None]
            results["group_cross_validation"] = {
                "folds": fold_results,
                "mean_macro_f1": float(np.mean(fold_f1s)) if fold_f1s else None,
                "std_macro_f1": float(np.std(fold_f1s)) if fold_f1s else None,
            }

            # 3. Preprocessing sensitivity
            preprocessing_variants = [
                {"label": "unigrams", "overrides": {"ngram_max": 1}},
                {"label": "unigrams+bigrams", "overrides": {"ngram_max": 2}},
                {"label": "stopwords_removed", "overrides": {"remove_stopwords": True}},
            ]
            preprocessing_rows = []
            for variant in preprocessing_variants:
                config = {**base_config, **variant["overrides"]}
                outcome = _safe_train_eval(
                    texts, y, groups,
                    label_names=label_names, config=config, algorithm=params["algorithm"],
                    class_weight=None, regularization_c=1.0, random_seed=42, test_size=params["test_size"],
                )
                preprocessing_rows.append({"variant": variant["label"], **outcome})
            results["preprocessing_sensitivity"] = preprocessing_rows

            # 4. Class-weight sensitivity
            class_weight_rows = []
            for class_weight in params["class_weights"]:
                outcome = _safe_train_eval(
                    texts, y, groups,
                    label_names=label_names, config=base_config, algorithm=params["algorithm"],
                    class_weight=class_weight, regularization_c=1.0, random_seed=42, test_size=params["test_size"],
                )
                class_weight_rows.append({"class_weight": class_weight or "none", **outcome})
            results["class_weight_sensitivity"] = class_weight_rows

            # 5. Leave-one-organization-out
            org_by_doc = {
                doc_id: (doc.organization or "unspecified") for doc_id, doc in documents_by_id.items()
            }
            organizations = sorted({org_by_doc.get(g, "unspecified") for g in groups})
            org_rows = []
            if len(organizations) < 2:
                org_rows.append(
                    {
                        "held_out_organization": None,
                        "status": "not_evaluable",
                        "reason": "Need at least two organizations",
                        "macro_f1": None,
                    }
                )
            else:
                for held_out_org in organizations:
                    train_idx = [i for i, g in enumerate(groups) if org_by_doc.get(g, "unspecified") != held_out_org]
                    test_idx = [i for i, g in enumerate(groups) if org_by_doc.get(g, "unspecified") == held_out_org]
                    if not train_idx or not test_idx:
                        org_rows.append(
                            {
                                "held_out_organization": held_out_org,
                                "test_size": len(test_idx),
                                "status": "not_evaluable",
                                "reason": "Empty train or test data for organization holdout",
                                "macro_f1": None,
                            }
                        )
                        continue
                    outcome = _safe_fit(
                        [texts[i] for i in train_idx],
                        [y[i] for i in train_idx],
                        [texts[i] for i in test_idx],
                        [y[i] for i in test_idx],
                        label_names=label_names,
                        config=base_config,
                        algorithm=params["algorithm"],
                        class_weight=None,
                        regularization_c=1.0,
                        random_seed=42,
                    )
                    org_rows.append(
                        {
                            "held_out_organization": held_out_org,
                            "test_size": len(test_idx),
                            **outcome,
                        }
                    )
            results["leave_one_organization_out"] = org_rows

            # 6. Temporal holdout
            year_by_doc = {doc_id: doc.publication_year for doc_id, doc in documents_by_id.items()}
            years = sorted({year_by_doc.get(g) for g in groups if year_by_doc.get(g) is not None})
            temporal_rows = []
            if len(years) < 2:
                temporal_rows.append(
                    {
                        "status": "not_evaluable",
                        "reason": "Need publication years spanning at least two periods",
                        "macro_f1": None,
                    }
                )
            else:
                split_year = years[len(years) // 2]
                train_idx = [
                    i for i, g in enumerate(groups)
                    if year_by_doc.get(g) is not None and year_by_doc[g] <= split_year
                ]
                test_idx = [
                    i for i, g in enumerate(groups)
                    if year_by_doc.get(g) is not None and year_by_doc[g] > split_year
                ]
                if not train_idx or not test_idx:
                    temporal_rows.append(
                        {
                            "split_year": split_year,
                            "train_size": len(train_idx),
                            "test_size": len(test_idx),
                            "status": "not_evaluable",
                            "reason": "Empty train or test period",
                            "macro_f1": None,
                        }
                    )
                else:
                    outcome = _safe_fit(
                        [texts[i] for i in train_idx],
                        [y[i] for i in train_idx],
                        [texts[i] for i in test_idx],
                        [y[i] for i in test_idx],
                        label_names=label_names,
                        config=base_config,
                        algorithm=params["algorithm"],
                        class_weight=None,
                        regularization_c=1.0,
                        random_seed=42,
                    )
                    temporal_rows.append(
                        {
                            "split_year": split_year,
                            "train_size": len(train_idx),
                            "test_size": len(test_idx),
                            **outcome,
                        }
                    )
            results["temporal_holdout"] = temporal_rows

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "mean_seed_macro_f1": results["seed_stability"]["mean_macro_f1"],
                        "mean_cv_macro_f1": results["group_cross_validation"]["mean_macro_f1"],
                    }
                ),
                results_json=dumps(results),
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
