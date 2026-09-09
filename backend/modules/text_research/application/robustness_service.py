"""Robustness testing: repeated-seed stability, grouped cross-validation,
preprocessing sensitivity, class-weight sensitivity, leave-one-group-out,
generic transfer tests, and temporal holdout — all computed against a frozen
`TrainingDatasetSnapshot`, never against mutable annotations.

Group/temporal fields are USER CONFIGURABLE (§38/§39): callers pass
``group_field`` (default ``"organization"``, but any facet field on
``CorpusDocument`` — including custom ``metadata_json`` keys — works) and
``temporal_field`` (default ``"publication_year"``). Nothing here hardcodes
a specific grouping dimension; see
:mod:`backend.modules.text_research.infrastructure.validation_splits` for the
pure split logic.

All robustness runs execute as a single `AnalysisRun` (optionally dispatched
to Celery for larger sweeps) so results are persisted and auditable.

Limitation: leave-one-group-out sweeps one fit per distinct group value, so
very high-cardinality fields are capped via ``max_groups`` to bound cost.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.application.dataset_builder_service import DatasetBuilderService
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, dumps, loads
from backend.modules.text_research.infrastructure.classifiers import (
    fit_tfidf_classifier,
    grouped_train_test_split,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.validation_splits import (
    class_prevalence,
    expanding_window_splits,
    leave_one_group_out,
    temporal_holdout_split,
    transfer_split,
)


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
        group_field: str = "organization",
        max_groups: int | None = 25,
        temporal_field: str = "publication_year",
        temporal_windows: bool = False,
        transfer_field: str | None = None,
        transfer_train_values: list[str] | None = None,
        transfer_test_values: list[str] | None = None,
        run_async: bool = True,
    ) -> AnalysisRun:
        snapshot = await self.get_snapshot_or_404(snapshot_id, user_id=user_id)
        params = {
            "snapshot_id": snapshot_id,
            "algorithm": algorithm,
            "seeds": seeds or [1, 11, 21, 42, 84],
            "cv_folds": cv_folds,
            "class_weights": class_weights if class_weights is not None else [None, "balanced"],
            "test_size": test_size,
            "group_field": group_field,
            "max_groups": max_groups,
            "temporal_field": temporal_field,
            "temporal_windows": temporal_windows,
            "transfer_field": transfer_field,
            "transfer_train_values": transfer_train_values,
            "transfer_test_values": transfer_test_values,
        }
        spec = build_spec_from_request(
            "classification",
            snapshot.corpus_id,
            snapshot_id=snapshot_id,
            model={"family": algorithm},
            validation={"strategy": "grouped_cv", "group_field": group_field},
            analysis_parameters={"mode": "robustness", "cv_folds": cv_folds},
        )
        params = attach_run_identity(params, spec)
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
            from backend.modules.text_research.application.execution_service import ExecutionService

            await ExecutionService.submit(
                db=self.db, run=run, operation="robustness", user_id=user_id
            )
        else:
            await self.execute_sweep(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_sweep(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run, status=AnalysisRunStatus.RUNNING.value, started_at=_utcnow()
        )
        await self.db.commit()

        try:
            (
                _snapshot,
                label_names,
                unit_labels,
                units,
                documents_by_id,
            ) = await self._load_snapshot_data(params["snapshot_id"], user_id=run.created_by)
            texts = [u.text for u in units]
            y = [unit_labels.get(u.id, []) for u in units]
            groups = [u.corpus_document_id for u in units]
            base_config = PreprocessingConfig().to_dict()

            results: dict[str, Any] = {}

            # 1. Repeated-seed stability
            seed_runs = []
            for seed in params["seeds"]:
                outcome = _safe_train_eval(
                    texts,
                    y,
                    groups,
                    label_names=label_names,
                    config=base_config,
                    algorithm=params["algorithm"],
                    class_weight=None,
                    regularization_c=1.0,
                    random_seed=seed,
                    test_size=params["test_size"],
                )
                seed_runs.append({"seed": seed, **outcome})
            seed_f1s = [
                r["macro_f1"]
                for r in seed_runs
                if r.get("status") == "ok" and r.get("macro_f1") is not None
            ]
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
            fold_f1s = [
                r["macro_f1"]
                for r in fold_results
                if r.get("status") == "ok" and r.get("macro_f1") is not None
            ]
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
                    texts,
                    y,
                    groups,
                    label_names=label_names,
                    config=config,
                    algorithm=params["algorithm"],
                    class_weight=None,
                    regularization_c=1.0,
                    random_seed=42,
                    test_size=params["test_size"],
                )
                preprocessing_rows.append({"variant": variant["label"], **outcome})
            results["preprocessing_sensitivity"] = preprocessing_rows

            # 4. Class-weight sensitivity
            class_weight_rows = []
            for class_weight in params["class_weights"]:
                outcome = _safe_train_eval(
                    texts,
                    y,
                    groups,
                    label_names=label_names,
                    config=base_config,
                    algorithm=params["algorithm"],
                    class_weight=class_weight,
                    regularization_c=1.0,
                    random_seed=42,
                    test_size=params["test_size"],
                )
                class_weight_rows.append({"class_weight": class_weight or "none", **outcome})
            results["class_weight_sensitivity"] = class_weight_rows

            # 5. Leave-one-group-out (§38: group_field is user-configurable,
            # not hardcoded to "organization"; works for any CorpusDocument
            # facet field or custom metadata_json key).
            group_field = params["group_field"]
            group_values = [
                documents_by_id[g].get_field_value(group_field) if g in documents_by_id else None
                for g in groups
            ]
            group_splits = leave_one_group_out(group_values, max_groups=params.get("max_groups"))
            group_rows: list[dict[str, Any]] = []
            if not group_splits:
                group_rows.append(
                    {
                        "held_out_value": None,
                        "status": "not_evaluable",
                        "reason": (
                            f"Need at least two distinct values for group_field={group_field!r}"
                        ),
                        "macro_f1": None,
                    }
                )
            else:
                for split in group_splits:
                    if not split.train_index or not split.test_index:
                        group_rows.append(
                            {
                                "held_out_value": split.held_out_value,
                                "test_size": len(split.test_index),
                                "status": "not_evaluable",
                                "reason": (
                                    f"Empty train or test data for "
                                    f"{group_field}={split.held_out_value!r}"
                                ),
                                "macro_f1": None,
                            }
                        )
                        continue
                    outcome = _safe_fit(
                        [texts[i] for i in split.train_index],
                        [y[i] for i in split.train_index],
                        [texts[i] for i in split.test_index],
                        [y[i] for i in split.test_index],
                        label_names=label_names,
                        config=base_config,
                        algorithm=params["algorithm"],
                        class_weight=None,
                        regularization_c=1.0,
                        random_seed=42,
                    )
                    group_rows.append(
                        {
                            "held_out_value": split.held_out_value,
                            "train_size": len(split.train_index),
                            "test_size": len(split.test_index),
                            "class_prevalence_test": class_prevalence(
                                [y[i] for i in split.test_index]
                            ),
                            **outcome,
                        }
                    )
            results["leave_one_group_out"] = {"group_field": group_field, "runs": group_rows}

            # 5b. Generic transfer test (§38): train where field ∈ A, test
            # where field ∈ B, both user-defined value sets.
            transfer_field = params.get("transfer_field")
            transfer_result: dict[str, Any] | None = None
            if transfer_field:
                transfer_values = [
                    documents_by_id[g].get_field_value(transfer_field)
                    if g in documents_by_id
                    else None
                    for g in groups
                ]
                try:
                    split = transfer_split(
                        transfer_values,
                        train_values=params.get("transfer_train_values") or [],
                        test_values=params.get("transfer_test_values") or [],
                    )
                except ValueError as exc:
                    transfer_result = {
                        "field": transfer_field,
                        "status": "not_evaluable",
                        "reason": str(exc),
                    }
                else:
                    if not split.train_index or not split.test_index:
                        transfer_result = {
                            "field": transfer_field,
                            "train_values": split.train_values,
                            "test_values": split.test_values,
                            "status": "not_evaluable",
                            "reason": "Empty train or test data for the requested transfer filters",
                        }
                    else:
                        outcome = _safe_fit(
                            [texts[i] for i in split.train_index],
                            [y[i] for i in split.train_index],
                            [texts[i] for i in split.test_index],
                            [y[i] for i in split.test_index],
                            label_names=label_names,
                            config=base_config,
                            algorithm=params["algorithm"],
                            class_weight=None,
                            regularization_c=1.0,
                            random_seed=42,
                        )
                        transfer_result = {
                            "field": transfer_field,
                            "train_values": split.train_values,
                            "test_values": split.test_values,
                            "train_size": len(split.train_index),
                            "test_size": len(split.test_index),
                            "class_prevalence_test": class_prevalence(
                                [y[i] for i in split.test_index]
                            ),
                            **outcome,
                        }
            results["transfer_test"] = transfer_result

            # 6. Temporal holdout (§39: temporal_field is user-selected, not
            # hardcoded to publication_year) plus optional expanding-window
            # validation when the field has enough distinct periods.
            temporal_field = params["temporal_field"]
            temporal_values = [
                documents_by_id[g].get_field_value(temporal_field) if g in documents_by_id else None
                for g in groups
            ]
            holdout = temporal_holdout_split(temporal_values)
            if not holdout.train_index or not holdout.test_index:
                temporal_rows = [
                    {
                        "field": temporal_field,
                        "status": "not_evaluable",
                        "reason": (
                            f"Need temporal_field={temporal_field!r} spanning at least two periods"
                        ),
                        "macro_f1": None,
                    }
                ]
            else:
                outcome = _safe_fit(
                    [texts[i] for i in holdout.train_index],
                    [y[i] for i in holdout.train_index],
                    [texts[i] for i in holdout.test_index],
                    [y[i] for i in holdout.test_index],
                    label_names=label_names,
                    config=base_config,
                    algorithm=params["algorithm"],
                    class_weight=None,
                    regularization_c=1.0,
                    random_seed=42,
                )
                temporal_rows = [
                    {
                        "field": temporal_field,
                        "train_period": holdout.train_period,
                        "test_period": holdout.test_period,
                        "train_size": len(holdout.train_index),
                        "test_size": len(holdout.test_index),
                        "class_prevalence_test": class_prevalence(
                            [y[i] for i in holdout.test_index]
                        ),
                        **outcome,
                    }
                ]
            results["temporal_holdout"] = {"field": temporal_field, "runs": temporal_rows}

            expanding_rows: list[dict[str, Any]] = []
            if params.get("temporal_windows"):
                windows = expanding_window_splits(temporal_values)
                if not windows:
                    expanding_rows.append(
                        {
                            "status": "not_evaluable",
                            "reason": (
                                f"temporal_field={temporal_field!r} needs "
                                ">=3 distinct periods for expanding-window validation"
                            ),
                        }
                    )
                else:
                    for window in windows:
                        outcome = _safe_fit(
                            [texts[i] for i in window.train_index],
                            [y[i] for i in window.train_index],
                            [texts[i] for i in window.test_index],
                            [y[i] for i in window.test_index],
                            label_names=label_names,
                            config=base_config,
                            algorithm=params["algorithm"],
                            class_weight=None,
                            regularization_c=1.0,
                            random_seed=42,
                        )
                        expanding_rows.append(
                            {
                                "train_period": window.train_period,
                                "test_period": window.test_period,
                                "train_size": len(window.train_index),
                                "test_size": len(window.test_index),
                                **outcome,
                            }
                        )
            results["temporal_expanding_window"] = {
                "field": temporal_field,
                "enabled": bool(params.get("temporal_windows")),
                "runs": expanding_rows,
            }

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
