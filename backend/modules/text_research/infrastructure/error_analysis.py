"""Classifier error analysis helpers for text research training reports."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skmetrics

from backend.modules.text_research.infrastructure.classifiers import ABSTENTION_MARKER, _to_native


def _infer_task_type(y_true: np.ndarray, label_names: list[Any] | None) -> str:
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        return "multilabel"
    n_classes = len(label_names) if label_names else len(np.unique(y_true))
    return "binary" if n_classes == 2 else "multiclass"


def _sample_confidence(y_proba: np.ndarray | None, task_type: str, index: int) -> float | None:
    if y_proba is None:
        return None
    proba = np.asarray(y_proba)
    if task_type == "binary":
        if proba.ndim == 2:
            return float(max(proba[index, 0], proba[index, 1]))
        return float(max(proba[index], 1.0 - proba[index]))
    if task_type == "multilabel":
        return float(np.max(proba[index]))
    return float(np.max(proba[index]))


def _is_valid_prediction(pred: Any) -> bool:
    if pred is None:
        return False
    if isinstance(pred, str):
        return pred != ABSTENTION_MARKER
    return True


def _performance_dict(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, Any]:
    zd = {"zero_division": 0}
    if task_type == "multilabel":
        return {
            "n_samples": int(len(y_true)),
            "subset_accuracy": float(skmetrics.accuracy_score(y_true, y_pred)),
            "hamming_loss": float(skmetrics.hamming_loss(y_true, y_pred)),
            "f1_macro": float(skmetrics.f1_score(y_true, y_pred, average="macro", **zd)),
            "precision_macro": float(
                skmetrics.precision_score(y_true, y_pred, average="macro", **zd)
            ),
            "recall_macro": float(skmetrics.recall_score(y_true, y_pred, average="macro", **zd)),
        }
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(skmetrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skmetrics.balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(skmetrics.f1_score(y_true, y_pred, average="macro", **zd)),
        "precision_macro": float(skmetrics.precision_score(y_true, y_pred, average="macro", **zd)),
        "recall_macro": float(skmetrics.recall_score(y_true, y_pred, average="macro", **zd)),
    }


def _false_positives_negatives(
    unit_ids: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    label_names: list[Any] | None,
) -> dict[str, Any]:
    if task_type == "multilabel":
        labels = [str(label) for label in (label_names or range(y_true.shape[1]))]
        out: dict[str, Any] = {}
        for i, label in enumerate(labels):
            fp_ids = [
                str(unit_ids[j])
                for j in range(len(unit_ids))
                if y_true[j, i] == 0 and y_pred[j, i] == 1
            ]
            fn_ids = [
                str(unit_ids[j])
                for j in range(len(unit_ids))
                if y_true[j, i] == 1 and y_pred[j, i] == 0
            ]
            out[label] = {"false_positives": fp_ids, "false_negatives": fn_ids}
        return out

    false_positives = [
        str(unit_ids[i]) for i in range(len(unit_ids)) if y_true[i] == 0 and y_pred[i] == 1
    ]
    false_negatives = [
        str(unit_ids[i]) for i in range(len(unit_ids)) if y_true[i] == 1 and y_pred[i] == 0
    ]
    if task_type == "multiclass":
        misclassified = [
            {
                "unit_id": str(unit_ids[i]),
                "true_label": int(y_true[i]),
                "predicted_label": int(y_pred[i]),
            }
            for i in range(len(unit_ids))
            if y_true[i] != y_pred[i]
        ]
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "misclassified": misclassified,
        }
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def _performance_by_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[Any] | None,
    task_type: str,
) -> dict[str, Any]:
    if task_type == "multilabel":
        labels = [str(label) for label in (label_names or range(y_true.shape[1]))]
        return {
            label: {
                "precision": float(
                    skmetrics.precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                ),
                "recall": float(
                    skmetrics.recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                ),
                "f1": float(skmetrics.f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "support": int(y_true[:, i].sum()),
            }
            for i, label in enumerate(labels)
        }

    labels = list(range(len(label_names or np.unique(y_true))))
    target_names = [str(label) for label in (label_names or labels)]
    report = skmetrics.classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {name: report[name] for name in target_names if name in report}


def _performance_by_metadata_slice(
    unit_ids: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_by_unit: dict[str, dict],
    slice_fields: list[str],
    task_type: str,
) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for field in slice_fields:
        field_slices: dict[str, Any] = {}
        values_by_index: dict[str, list[int]] = {}
        for idx, unit_id in enumerate(unit_ids):
            meta = metadata_by_unit.get(str(unit_id), {})
            value = meta.get(field)
            key = str(value) if value is not None else "__missing__"
            values_by_index.setdefault(key, []).append(idx)
        for value, indices in values_by_index.items():
            idx_arr = np.asarray(indices, dtype=int)
            field_slices[value] = _performance_dict(y_true[idx_arr], y_pred[idx_arr], task_type)
        slices[field] = field_slices
    return slices


def _performance_by_document(
    unit_ids: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[Any],
    task_type: str,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(str(group), []).append(idx)
    return {
        group: _performance_dict(
            y_true[np.asarray(indices, dtype=int)],
            y_pred[np.asarray(indices, dtype=int)],
            task_type,
        )
        for group, indices in by_group.items()
    }


def classifier_error_report(
    *,
    unit_ids: list[Any],
    y_true: np.ndarray | list[Any],
    y_pred: np.ndarray | list[Any],
    y_proba: np.ndarray | list[Any] | None = None,
    groups: list[Any] | None = None,
    metadata_by_unit: dict[str, dict] | None = None,
    slice_fields: list[str] | None = None,
    label_names: list[Any] | None = None,
    top_k_uncertain: int = 50,
) -> dict[str, Any]:
    """Build a structured classifier error-analysis report for one evaluation set."""
    if len(unit_ids) != len(y_true) or len(unit_ids) != len(y_pred):
        raise ValueError("unit_ids, y_true, and y_pred must have the same length")
    if groups is not None and len(groups) != len(unit_ids):
        raise ValueError("groups must have the same length as unit_ids when provided")

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    valid_mask = np.array([_is_valid_prediction(p) for p in y_pred_arr], dtype=bool)
    if valid_mask.any():
        y_true_eval = y_true_arr[valid_mask]
        y_pred_eval = y_pred_arr[valid_mask]
        unit_ids_eval = [unit_ids[i] for i, ok in enumerate(valid_mask) if ok]
    else:
        y_true_eval = y_true_arr
        y_pred_eval = y_pred_arr
        unit_ids_eval = list(unit_ids)

    task_type = _infer_task_type(y_true_eval, label_names)
    report: dict[str, Any] = {
        "task_type": task_type,
        "n_samples": len(unit_ids),
        "n_valid_predictions": int(valid_mask.sum()) if valid_mask.any() else len(unit_ids),
        "errors": _false_positives_negatives(
            unit_ids_eval, y_true_eval, y_pred_eval, task_type, label_names
        ),
        "performance_by_label": _performance_by_label(
            y_true_eval, y_pred_eval, label_names, task_type
        ),
    }

    if task_type in ("binary", "multiclass"):
        labels = list(range(len(label_names or np.unique(y_true_eval))))
        cm = skmetrics.confusion_matrix(y_true_eval, y_pred_eval, labels=labels)
        report["confusion_matrix"] = cm.tolist()
        report["confusion_matrix_labels"] = [str(label) for label in (label_names or labels)]
    elif task_type == "multilabel":
        report["confusion_matrix"] = skmetrics.multilabel_confusion_matrix(
            y_true_eval, y_pred_eval
        ).tolist()

    uncertain_cases: list[dict[str, Any]] = []
    high_confidence_errors: list[dict[str, Any]] = []
    if y_proba is not None:
        proba = np.asarray(y_proba)
        for i in range(len(unit_ids)):
            confidence = _sample_confidence(proba, task_type, i)
            if confidence is None:
                continue
            pred = y_pred_arr[i]
            true = y_true_arr[i]
            is_error = (
                not np.array_equal(true, pred)
                if task_type == "multilabel"
                else (_is_valid_prediction(pred) and true != pred)
            )
            entry = {
                "unit_id": str(unit_ids[i]),
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "y_true": _to_native(true),
                "y_pred": _to_native(pred),
            }
            if groups is not None:
                entry["group"] = str(groups[i])
            uncertain_cases.append(entry)
            if is_error and confidence >= 0.75:
                high_confidence_errors.append(entry)

        uncertain_cases.sort(key=lambda row: row["uncertainty"], reverse=True)
        high_confidence_errors.sort(key=lambda row: -row["confidence"])

    report["most_uncertain_cases"] = uncertain_cases[:top_k_uncertain]
    report["high_confidence_errors"] = high_confidence_errors[:top_k_uncertain]

    if groups is not None:
        groups_eval = (
            [groups[i] for i in range(len(groups)) if valid_mask[i]]
            if valid_mask.any()
            else list(groups)
        )
        report["performance_by_document"] = _performance_by_document(
            unit_ids_eval, y_true_eval, y_pred_eval, groups_eval, task_type
        )

    if metadata_by_unit and slice_fields:
        report["performance_by_metadata_slice"] = _performance_by_metadata_slice(
            unit_ids_eval,
            y_true_eval,
            y_pred_eval,
            metadata_by_unit,
            slice_fields,
            task_type,
        )

    return _to_native(report)
