"""Grouped split planning with optional stratified group holdout.

Uses ``StratifiedGroupKFold`` when every class appears in at least two
distinct groups; otherwise falls back to ``GroupShuffleSplit`` (same
leakage-safe group boundary as :func:`classifiers.grouped_train_val_test_split`).

Multilabel targets never use stratified grouped splitting — each row is a
label set (unhashable / not a single class). Leakage-safe ``GroupShuffleSplit``
is used instead, with explicit provenance that stratification was requested
but not applied.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RECOMMENDED_STRATEGIES = ("stratified_group", "group_shuffle", "group_shuffle_multilabel")


@dataclass
class SplitFeasibility:
    """Feasibility assessment for stratified grouped splitting."""

    n_groups: int
    class_counts: dict[Any, int]
    feasible_stratified_group: bool
    reason: str
    recommended_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_multilabel_labels(labels: list[Any]) -> bool:
    """True when any target row is a list/tuple/set (multilabel encoding)."""
    return any(isinstance(label, (list, tuple, set)) for label in labels)


def evaluate_stratified_group_feasibility(
    labels: list[Any],
    groups: list[Any],
    *,
    min_groups_per_class: int = 2,
    task_type: str | None = None,
) -> SplitFeasibility:
    """Return whether stratified grouped splits are feasible for ``labels``/``groups``."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    n_groups = len(set(groups))
    resolved_task = task_type
    if resolved_task is None and _is_multilabel_labels(labels):
        resolved_task = "multilabel"

    if resolved_task == "multilabel" or _is_multilabel_labels(labels):
        return SplitFeasibility(
            n_groups=n_groups,
            class_counts={},
            feasible_stratified_group=False,
            reason=(
                "standard stratified grouped splitting is not valid for multilabel targets"
            ),
            recommended_strategy="group_shuffle_multilabel",
        )

    class_counts = dict(Counter(labels))
    groups_by_class: dict[Any, set[Any]] = defaultdict(set)
    for label, group in zip(labels, groups, strict=True):
        groups_by_class[label].add(group)

    if n_groups < 2:
        return SplitFeasibility(
            n_groups=n_groups,
            class_counts=class_counts,
            feasible_stratified_group=False,
            reason="fewer than 2 distinct groups",
            recommended_strategy="group_shuffle",
        )

    for label, group_set in sorted(groups_by_class.items(), key=lambda item: str(item[0])):
        n_class_groups = len(group_set)
        if n_class_groups < min_groups_per_class:
            return SplitFeasibility(
                n_groups=n_groups,
                class_counts=class_counts,
                feasible_stratified_group=False,
                reason=(
                    f"class {label!r} appears in only {n_class_groups} group"
                    f"{'s' if n_class_groups != 1 else ''}; "
                    f"stratified grouped split requires at least {min_groups_per_class} "
                    "groups per class"
                ),
                recommended_strategy="group_shuffle",
            )

    return SplitFeasibility(
        n_groups=n_groups,
        class_counts=class_counts,
        feasible_stratified_group=True,
        reason="each class appears in at least 2 distinct groups",
        recommended_strategy="stratified_group",
    )


def _pick_stratified_fold(
    labels: list[Any],
    groups: list[Any],
    *,
    n_splits: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    indices = np.arange(len(labels))
    sgkf = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_seed)
    folds = list(sgkf.split(indices, labels, groups))
    fold_idx = int(random_seed) % len(folds)
    return folds[fold_idx]


def _group_shuffle_holdout(
    pool_indices: np.ndarray,
    pool_groups: list[Any],
    *,
    holdout_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    if holdout_size <= 0 or len(pool_indices) == 0:
        return pool_indices, np.array([], dtype=int)
    if len(set(pool_groups)) < 2:
        return pool_indices, np.array([], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_seed)
    rel_train, rel_holdout = next(splitter.split(pool_indices, groups=pool_groups))
    return pool_indices[rel_train], pool_indices[rel_holdout]


def _assert_disjoint_group_partitions(
    groups: list[Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}
    if (
        not train_groups.isdisjoint(test_groups)
        or not train_groups.isdisjoint(val_groups)
        or not val_groups.isdisjoint(test_groups)
    ):
        raise AssertionError("Grouped split produced overlapping groups across partitions")


def plan_grouped_splits(
    labels: list[Any],
    groups: list[Any],
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    prefer_stratified: bool = True,
    task_type: str | None = None,
    grouping_variable: str = "corpus_document_id",
) -> dict[str, Any]:
    """Plan train/validation/test index partitions with no group leakage.

    Returns index lists compatible with
    :func:`classifiers.grouped_train_val_test_split`.

    For ``task_type="multilabel"`` (or multilabel-shaped labels), always uses
    leakage-safe ``GroupShuffleSplit`` and records that stratification was not
    applied.
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    n_unique_groups = len(set(groups))
    if n_unique_groups < 2:
        raise ValueError("plan_grouped_splits requires at least 2 distinct groups")

    indices = np.arange(len(labels))
    feasibility = evaluate_stratified_group_feasibility(
        labels, groups, task_type=task_type
    )
    is_multilabel = (
        task_type == "multilabel"
        or feasibility.recommended_strategy == "group_shuffle_multilabel"
    )
    stratification_requested = prefer_stratified
    use_stratified = (
        prefer_stratified and feasibility.feasible_stratified_group and not is_multilabel
    )
    notes: list[str] = []
    if is_multilabel:
        strategy = "group_shuffle_multilabel"
        stratification_applied = False
        notes.append(feasibility.reason)
    elif use_stratified:
        strategy = "stratified_group"
        stratification_applied = True
    else:
        strategy = "group_shuffle"
        stratification_applied = False
        if prefer_stratified and not feasibility.feasible_stratified_group:
            notes.append(
                "Stratified grouped split unavailable; "
                f"falling back to GroupShuffleSplit ({feasibility.reason})."
            )

    if use_stratified:
        n_splits_test = max(2, int(round(1.0 / test_size)) if test_size > 0 else 2)
        train_val_idx, test_idx = _pick_stratified_fold(
            labels, groups, n_splits=n_splits_test, random_seed=random_seed
        )
    else:
        train_val_idx, test_idx = _group_shuffle_holdout(
            indices, groups, holdout_size=test_size, random_seed=random_seed
        )

    train_idx = train_val_idx
    val_idx = np.array([], dtype=int)
    if val_size and val_size > 0 and len(train_val_idx):
        train_val_labels = [labels[i] for i in train_val_idx]
        train_val_groups = [groups[i] for i in train_val_idx]
        n_remaining_groups = len(set(train_val_groups))
        if n_remaining_groups >= 2:
            inner_feasibility = evaluate_stratified_group_feasibility(
                train_val_labels, train_val_groups, task_type=task_type
            )
            inner_stratified = (
                use_stratified
                and inner_feasibility.feasible_stratified_group
                and not is_multilabel
            )
            try:
                if inner_stratified:
                    n_splits_val = max(2, int(round(1.0 / val_size)))
                    rel_train, rel_val = _pick_stratified_fold(
                        train_val_labels,
                        train_val_groups,
                        n_splits=n_splits_val,
                        random_seed=random_seed + 1,
                    )
                    train_idx = train_val_idx[rel_train]
                    val_idx = train_val_idx[rel_val]
                else:
                    rel_train, rel_val = _group_shuffle_holdout(
                        train_val_idx,
                        train_val_groups,
                        holdout_size=val_size,
                        random_seed=random_seed + 1,
                    )
                    train_idx = rel_train
                    val_idx = rel_val
            except ValueError:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=int)
                notes.append(
                    "Validation split skipped: too few groups remained after "
                    "the test split to form a non-empty train/validation pair."
                )
        else:
            notes.append(
                "Validation split skipped: fewer than 2 distinct groups remained "
                "after the test split."
            )

    _assert_disjoint_group_partitions(groups, train_idx, val_idx, test_idx)

    return {
        "train_index": train_idx.tolist(),
        "val_index": val_idx.tolist(),
        "test_index": test_idx.tolist(),
        "strategy": strategy,
        "split_strategy": strategy,
        "task_type": task_type or ("multilabel" if is_multilabel else "single_label"),
        "grouping_variable": grouping_variable,
        "stratification_requested": stratification_requested,
        "stratification_applied": stratification_applied,
        "reason": feasibility.reason if not stratification_applied else None,
        "feasibility": feasibility.to_dict(),
        "notes": notes,
    }


def nested_grouped_cv_indices(
    labels: list[Any],
    groups: list[Any],
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_seed: int = 42,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    """Nested grouped CV index plan: outer train/test + inner train/val folds."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    indices = np.arange(len(labels))
    feasibility = evaluate_stratified_group_feasibility(
        labels, groups, task_type=task_type
    )
    outer_results: list[dict[str, Any]] = []
    is_multilabel = (
        task_type == "multilabel"
        or feasibility.recommended_strategy == "group_shuffle_multilabel"
    )

    if feasibility.feasible_stratified_group and not is_multilabel:
        from sklearn.model_selection import StratifiedGroupKFold

        outer_cv = StratifiedGroupKFold(
            n_splits=max(2, outer_splits), shuffle=True, random_state=random_seed
        )
        outer_iterator: Iterator[tuple[np.ndarray, np.ndarray]] = outer_cv.split(
            indices, labels, groups
        )
        outer_strategy = "stratified_group"
    else:
        from sklearn.model_selection import GroupKFold

        outer_cv = GroupKFold(n_splits=max(2, min(outer_splits, len(set(groups)))))
        outer_iterator = outer_cv.split(indices, groups=groups)
        outer_strategy = "group_kfold_multilabel" if is_multilabel else "group_kfold"

    for outer_train, outer_test in outer_iterator:
        outer_train_labels = [labels[i] for i in outer_train]
        outer_train_groups = [groups[i] for i in outer_train]
        inner_folds: list[dict[str, list[int]]] = []

        if len(set(outer_train_groups)) >= 2:
            inner_feasibility = evaluate_stratified_group_feasibility(
                outer_train_labels, outer_train_groups, task_type=task_type
            )
            if inner_feasibility.feasible_stratified_group and not is_multilabel:
                from sklearn.model_selection import StratifiedGroupKFold

                inner_cv = StratifiedGroupKFold(
                    n_splits=max(2, inner_splits),
                    shuffle=True,
                    random_state=random_seed,
                )
                inner_iterator = inner_cv.split(outer_train, outer_train_labels, outer_train_groups)
            else:
                from sklearn.model_selection import GroupKFold

                inner_cv = GroupKFold(
                    n_splits=max(2, min(inner_splits, len(set(outer_train_groups))))
                )
                inner_iterator = inner_cv.split(outer_train, groups=outer_train_groups)

            for inner_train_rel, inner_val_rel in inner_iterator:
                inner_folds.append(
                    {
                        "train": outer_train[inner_train_rel].tolist(),
                        "val": outer_train[inner_val_rel].tolist(),
                    }
                )

        outer_results.append(
            {
                "outer_train": outer_train.tolist(),
                "outer_test": outer_test.tolist(),
                "inner_folds": inner_folds,
                "outer_strategy": outer_strategy,
            }
        )

    return outer_results


class SplitPlanner:
    """Grouped split planner with optional stratified group holdout."""

    evaluate_feasibility = staticmethod(evaluate_stratified_group_feasibility)
    plan_grouped_splits = staticmethod(plan_grouped_splits)
    nested_grouped_cv_indices = staticmethod(nested_grouped_cv_indices)
