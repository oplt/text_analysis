"""TF-IDF + linear classifier engines for Policy Text Lab.

Implements the leakage-safe classification pipeline:

    Frozen Training Dataset -> Grouped Split -> Fit TF-IDF on TRAIN only ->
    Fit classifier -> Transform TEST with the TRAIN vectorizer -> Predict ->
    Metrics -> Persist.

Critical invariant: the TF-IDF vectorizer is always fit exclusively on
training texts; test texts are only ever ``.transform()``-ed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skmetrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from backend.modules.text_research.infrastructure.model_storage import load_joblib, save_joblib
from backend.modules.text_research.infrastructure.preprocessing import build_tfidf_vectorizer

TASK_TYPES = ("binary", "multiclass", "multilabel")
ALGORITHMS = ("logistic_regression", "linear_svm")


def _to_native(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_native(v) for v in value]
    return value


def grouped_train_test_split(
    X_texts: list[str],
    y: list[Any],
    groups: list[Any],
    test_size: float = 0.25,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Split by ``groups`` (e.g. source document id) using GroupShuffleSplit.

    Guarantees no group appears in both train and test (leakage prevention
    for sentence/paragraph units drawn from the same source document).
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not (len(X_texts) == len(y) == len(groups)):
        raise ValueError("X_texts, y, and groups must have the same length")

    n_unique_groups = len(set(groups))
    if n_unique_groups < 2:
        raise ValueError("grouped_train_test_split requires at least 2 distinct groups")

    indices = np.arange(len(X_texts))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(indices, groups=groups))

    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    if not train_groups.isdisjoint(test_groups):  # pragma: no cover - defensive
        raise AssertionError("GroupShuffleSplit produced overlapping groups between train/test")

    def _select(seq: list[Any], idx: np.ndarray) -> list[Any]:
        return [seq[i] for i in idx]

    return {
        "train_index": train_idx.tolist(),
        "test_index": test_idx.tolist(),
        "X_train": _select(X_texts, train_idx),
        "X_test": _select(X_texts, test_idx),
        "y_train": _select(y, train_idx),
        "y_test": _select(y, test_idx),
        "groups_train": _select(groups, train_idx),
        "groups_test": _select(groups, test_idx),
    }


def _build_base_estimator(
    algorithm: str,
    random_seed: int,
    class_weight: str | dict | None,
    C: float,
) -> Any:
    if algorithm == "logistic_regression":
        return LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=random_seed,
            class_weight=class_weight,
        )
    if algorithm == "linear_svm":
        return LinearSVC(
            C=C,
            max_iter=10000,
            random_state=random_seed,
            class_weight=class_weight,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm!r}; expected one of {ALGORITHMS}")


def _build_model(
    algorithm: str,
    task_type: str,
    random_seed: int,
    class_weight: str | dict | None,
    C: float,
) -> Any:
    base = _build_base_estimator(algorithm, random_seed, class_weight, C)
    if task_type == "multilabel":
        return OneVsRestClassifier(base)
    return base


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[Any],
    task_type: str,
    model: Any,
    X_test_vec: Any,
) -> dict[str, Any]:
    zd = {"zero_division": 0}
    metrics_out: dict[str, Any] = {
        "accuracy": skmetrics.accuracy_score(y_true, y_pred),
        "precision_macro": skmetrics.precision_score(y_true, y_pred, average="macro", **zd),
        "recall_macro": skmetrics.recall_score(y_true, y_pred, average="macro", **zd),
        "f1_macro": skmetrics.f1_score(y_true, y_pred, average="macro", **zd),
        "precision_micro": skmetrics.precision_score(y_true, y_pred, average="micro", **zd),
        "recall_micro": skmetrics.recall_score(y_true, y_pred, average="micro", **zd),
        "f1_micro": skmetrics.f1_score(y_true, y_pred, average="micro", **zd),
        "precision_weighted": skmetrics.precision_score(y_true, y_pred, average="weighted", **zd),
        "recall_weighted": skmetrics.recall_score(y_true, y_pred, average="weighted", **zd),
        "f1_weighted": skmetrics.f1_score(y_true, y_pred, average="weighted", **zd),
    }

    if task_type == "multilabel":
        precisions = skmetrics.precision_score(y_true, y_pred, average=None, **zd)
        recalls = skmetrics.recall_score(y_true, y_pred, average=None, **zd)
        f1s = skmetrics.f1_score(y_true, y_pred, average=None, **zd)
        supports = np.asarray(y_true).sum(axis=0)
        per_class = {
            str(cls): {
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1": float(f1s[i]),
                "support": int(supports[i]),
            }
            for i, cls in enumerate(classes)
        }
        metrics_out["per_class"] = per_class
    else:
        target_names = [str(c) for c in classes]
        report = skmetrics.classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        metrics_out["per_class"] = {name: report[name] for name in target_names if name in report}
        metrics_out["confusion_matrix"] = skmetrics.confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes)))
        ).tolist()
        metrics_out["confusion_matrix_labels"] = target_names

    if task_type == "multilabel":
        metrics_out["multilabel_confusion_matrices"] = {
            str(cls): matrix.tolist()
            for cls, matrix in zip(
                classes,
                skmetrics.multilabel_confusion_matrix(y_true, y_pred),
                strict=True,
            )
        }

    if task_type == "binary" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test_vec)[:, 1]
            if len(set(np.asarray(y_true).tolist())) > 1:
                metrics_out["roc_auc"] = skmetrics.roc_auc_score(y_true, proba)
        except (ValueError, IndexError):
            pass

    return metrics_out


def fit_tfidf_classifier(
    X_train_texts: list[str],
    y_train: list[Any],
    X_test_texts: list[str],
    y_test: list[Any],
    task_type: str,
    algorithm: str = "logistic_regression",
    preprocessing_config: dict[str, Any] | None = None,
    vectorizer_kwargs: dict[str, Any] | None = None,
    label_names: list[str] | None = None,
    class_weight: str | dict | None = None,
    C: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit a TF-IDF + linear classifier, respecting the train/test boundary.

    The vectorizer is fit ONLY on ``X_train_texts``; ``X_test_texts`` is only
    ever transformed with that already-fitted vectorizer.

    Returns a dict with the fitted ``vectorizer``, ``model``, label
    encoders, ``classes``, and computed ``metrics``.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type!r}; expected one of {TASK_TYPES}")

    vectorizer = build_tfidf_vectorizer(preprocessing_config)
    if vectorizer_kwargs:
        vectorizer.set_params(**vectorizer_kwargs)

    # CRITICAL: fit on train only, transform test with the train-fitted vectorizer.
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)

    label_encoder: LabelEncoder | None = None
    mlb: MultiLabelBinarizer | None = None

    if task_type == "multilabel":
        mlb = MultiLabelBinarizer(classes=label_names)
        y_train_enc = mlb.fit_transform(y_train)
        y_test_enc = mlb.transform(y_test)
        classes = list(mlb.classes_)
    else:
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
        classes = list(label_encoder.classes_)

    model = _build_model(algorithm, task_type, random_seed, class_weight, C)
    model.fit(X_train_vec, y_train_enc)
    y_pred = model.predict(X_test_vec)

    metrics_out = _classification_metrics(y_test_enc, y_pred, classes, task_type, model, X_test_vec)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "label_encoder": label_encoder,
        "multilabel_binarizer": mlb,
        "classes": [str(c) for c in classes],
        "metrics": _to_native(metrics_out),
        "algorithm": algorithm,
        "task_type": task_type,
        "n_train": len(X_train_texts),
        "n_test": len(X_test_texts),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def extract_linear_coefficients(
    model: Any,
    vectorizer: Any,
    label_names: list[str],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Explainability: feature/label/coefficient/direction/rank for linear models.

    Works for ``LogisticRegression``, ``LinearSVC``, and
    ``OneVsRestClassifier`` wrapping either (all expose ``coef_``).
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise ValueError("Model does not expose linear coefficients (no coef_ attribute)")

    feature_names = list(vectorizer.get_feature_names_out())
    coef = np.asarray(coef)

    if coef.shape[0] == 1 and len(label_names) == 2:
        # Binary classifiers store a single row representing the positive
        # (second/index-1) class relative to the negative class.
        label_rows = [(label_names[1], coef[0])]
    else:
        label_rows = list(zip(label_names, coef, strict=False))

    results: list[dict[str, Any]] = []
    for label, row in label_rows:
        order = sorted(range(len(row)), key=lambda i: -float(row[i]))
        for rank, feature_idx in enumerate(order, start=1):
            value = float(row[feature_idx])
            if value > 0:
                direction = "positive"
            elif value < 0:
                direction = "negative"
            else:
                direction = "neutral"
            results.append(
                {
                    "feature": feature_names[feature_idx],
                    "label": label,
                    "coefficient": value,
                    "direction": direction,
                    "rank": rank,
                }
            )

    if top_n is None:
        return results

    trimmed: list[dict[str, Any]] = []
    for label, _ in label_rows:
        label_rows_out = [r for r in results if r["label"] == label]
        top_positive = [r for r in label_rows_out if r["coefficient"] > 0][:top_n]
        top_negative = [r for r in label_rows_out if r["coefficient"] < 0][-top_n:]
        trimmed.extend(top_positive + top_negative)
    return trimmed


def predict_with_uncertainty(
    model: Any,
    vectorizer: Any,
    texts: list[str],
    task_type: str,
) -> list[dict[str, Any]]:
    """Predict on new texts (transform-only) with an uncertainty score.

    Every score uses the same invariant: ``0`` is maximally certain and ``1``
    is maximally uncertain. This lets active-learning queries rank scores in
    descending order independent of classifier type.

    - binary: normalized distance from certainty around ``0.5`` (falls back
      to a decision-function-based proxy for non-probabilistic models such as
      LinearSVC).
    - multiclass: top-1/top-2 margin uncertainty.
    - multilabel: mean normalized binary entropy across labels.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type!r}; expected one of {TASK_TYPES}")

    X_vec = vectorizer.transform(texts)
    predictions = model.predict(X_vec)
    results: list[dict[str, Any]] = []

    if task_type == "binary":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)
            for i in range(len(texts)):
                p_positive = float(proba[i][-1])
                results.append(
                    {
                        "prediction": _to_native(predictions[i]),
                        "probability": p_positive,
                        "uncertainty": 1.0 - (2.0 * abs(p_positive - 0.5)),
                    }
                )
        else:
            scores = np.asarray(model.decision_function(X_vec)).reshape(-1)
            for i in range(len(texts)):
                score = float(scores[i])
                results.append(
                    {
                        "prediction": _to_native(predictions[i]),
                        "decision_score": score,
                        "uncertainty": 1.0 / (1.0 + abs(score)),
                    }
                )
        return results

    if task_type == "multiclass":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)
            for i in range(len(texts)):
                p = np.clip(proba[i], 1e-12, 1.0)
                entropy = float(-np.sum(p * np.log(p)))
                sorted_p = np.sort(proba[i])[::-1]
                margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
                results.append(
                    {
                        "prediction": _to_native(predictions[i]),
                        "probabilities": _to_native(proba[i]),
                        "entropy": entropy,
                        "margin": margin,
                        "uncertainty": 1.0 - margin,
                    }
                )
        else:
            scores = np.asarray(model.decision_function(X_vec))
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            for i in range(len(texts)):
                sorted_scores = np.sort(scores[i])[::-1]
                margin = (
                    float(sorted_scores[0] - sorted_scores[1])
                    if len(sorted_scores) > 1
                    else float("inf")
                )
                results.append(
                    {
                        "prediction": _to_native(predictions[i]),
                        "decision_scores": _to_native(scores[i]),
                        "uncertainty": 1.0 / (1.0 + margin),
                    }
                )
        return results

    # multilabel
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)
        for i in range(len(texts)):
            p = np.clip(np.asarray(proba[i]).reshape(-1), 1e-12, 1.0 - 1e-12)
            binary_entropy = float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))
            results.append(
                {
                    "prediction": _to_native(np.asarray(predictions[i]).reshape(-1)),
                    "probabilities": _to_native(p),
                    "entropy": binary_entropy,
                    "uncertainty": binary_entropy / float(np.log(2)),
                }
            )
    else:
        decision_scores = np.asarray(model.decision_function(X_vec))
        if decision_scores.ndim == 1:
            decision_scores = decision_scores.reshape(-1, 1)
        for i in range(len(texts)):
            scores = decision_scores[i]
            results.append(
                {
                    "prediction": _to_native(np.asarray(predictions[i]).reshape(-1)),
                    "decision_scores": _to_native(scores),
                    "uncertainty": float(np.mean(1.0 / (1.0 + np.abs(scores)))),
                }
            )
    return results


def save_classifier(vectorizer: Any, model: Any, vectorizer_path: str, model_path: str) -> None:
    """Persist a fitted vectorizer + classifier as separate joblib artifacts."""
    save_joblib(vectorizer, vectorizer_path)
    save_joblib(model, model_path)


def load_classifier(vectorizer_path: str, model_path: str) -> tuple[Any, Any]:
    """Load a previously persisted vectorizer + classifier pair."""
    return load_joblib(vectorizer_path), load_joblib(model_path)
