"""Sparse text classifier engines for text research.

Implements the leakage-safe classification pipeline:

    Frozen Training Dataset -> Grouped Split (train/val/test) -> Fit feature
    extractor on TRAIN only -> Fit classifier -> Transform VAL/TEST with the
    TRAIN-fitted extractor -> Predict -> Metrics -> Persist.

Critical invariant: the vectorizer is always fit exclusively on training
texts; validation/test texts are only ever ``.transform()``-ed.

Task types (binary / multiclass / multilabel) are never silently forced.
:func:`infer_task_type` only infers a task type from label shape when the
caller does not explicitly request one, and only when that shape is
unambiguous (see §27).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np
from sklearn import metrics as skmetrics
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold, ParameterGrid, ParameterSampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from backend.modules.text_research.infrastructure.model_storage import load_joblib, save_joblib
from backend.modules.text_research.infrastructure.preprocessing import (
    build_char_count_vectorizer,
    build_char_tfidf_vectorizer,
    build_count_vectorizer,
    build_tfidf_vectorizer,
)

TASK_TYPES = ("binary", "multiclass", "multilabel")
ALGORITHMS = (
    "logistic_regression",
    "linear_svm",
    "multinomial_nb",
    "complement_nb",
    "sgd_classifier",
)
# Algorithms whose scikit-learn estimator has no native `class_weight`
# constructor parameter. When a caller requests class weighting for one of
# these, we translate it into per-sample weights at `.fit()` time instead.
_NO_CLASS_WEIGHT_PARAM_ALGORITHMS = frozenset({"multinomial_nb", "complement_nb"})
_SGD_LOSSES = ("log_loss", "hinge")
VECTORIZER_TYPES = ("count", "tfidf")


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


def grouped_train_val_test_split(
    X_texts: list[str],
    y: list[Any],
    groups: list[Any],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Split by ``groups`` into train/validation/test with no group leakage.

    ``test_size`` is the fraction of ALL distinct groups held out for the
    final TEST partition (evaluated once, for primary/reported metrics).
    ``val_size`` is the fraction of the REMAINING (non-test) groups held out
    for a VALIDATION partition (intended for threshold tuning / model
    selection later — never for final reported metrics).

    Guarantees no group appears in more than one partition. If too few
    groups remain after the test split to carve out a non-trivial
    validation partition, validation is skipped (empty) rather than raising,
    and a human-readable note is returned under ``"notes"``.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not (len(X_texts) == len(y) == len(groups)):
        raise ValueError("X_texts, y, and groups must have the same length")

    n_unique_groups = len(set(groups))
    if n_unique_groups < 2:
        raise ValueError("grouped_train_val_test_split requires at least 2 distinct groups")

    indices = np.arange(len(X_texts))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(test_splitter.split(indices, groups=groups))

    def _select(seq: list[Any], idx: np.ndarray) -> list[Any]:
        return [seq[i] for i in idx]

    notes: list[str] = []
    train_val_groups = _select(groups, train_val_idx)
    n_remaining_groups = len(set(train_val_groups))

    val_idx: np.ndarray = np.array([], dtype=int)
    train_idx: np.ndarray = train_val_idx
    if val_size and val_size > 0:
        if n_remaining_groups >= 2:
            val_splitter = GroupShuffleSplit(
                n_splits=1, test_size=val_size, random_state=random_seed
            )
            try:
                rel_train_idx, rel_val_idx = next(
                    val_splitter.split(train_val_idx, groups=train_val_groups)
                )
                train_idx = train_val_idx[rel_train_idx]
                val_idx = train_val_idx[rel_val_idx]
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

    train_groups_final = {groups[i] for i in train_idx}
    val_groups_final = {groups[i] for i in val_idx}
    test_groups_final = {groups[i] for i in test_idx}
    if (
        not train_groups_final.isdisjoint(test_groups_final)
        or not train_groups_final.isdisjoint(val_groups_final)
        or not val_groups_final.isdisjoint(test_groups_final)
    ):  # pragma: no cover - defensive
        raise AssertionError("Grouped split produced overlapping groups across partitions")

    return {
        "train_index": train_idx.tolist(),
        "val_index": val_idx.tolist(),
        "test_index": test_idx.tolist(),
        "X_train": _select(X_texts, train_idx),
        "X_val": _select(X_texts, val_idx),
        "X_test": _select(X_texts, test_idx),
        "y_train": _select(y, train_idx),
        "y_val": _select(y, val_idx),
        "y_test": _select(y, test_idx),
        "groups_train": _select(groups, train_idx),
        "groups_val": _select(groups, val_idx),
        "groups_test": _select(groups, test_idx),
        "notes": notes,
    }


def infer_task_type(
    y: list[Any],
    requested: str | None = None,
    n_classes: int | None = None,
) -> str:
    """Resolve the effective classification task type (see §27).

    - If ``requested`` is provided (user/config-driven), it is always
      honored as-is — this function never overrides an explicit choice.
    - Otherwise, the task type is inferred ONLY from unambiguous label
      shape: if every example carries exactly one active label, the gold is
      single-label (``"binary"`` for exactly two observed classes, else
      ``"multiclass"``). If ANY example carries zero or more than one active
      label, the gold is treated as ``"multilabel"``.

    This never *silently* forces multilabel: single-label gold (list-like
    with exactly one active label per example, or already-scalar labels)
    always resolves to binary/multiclass.
    """
    if requested:
        normalized = str(requested).strip().lower()
        if normalized not in TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {requested!r}; expected one of {TASK_TYPES}")
        return normalized

    def _label_count(labels: Any) -> int:
        if isinstance(labels, list | tuple | set):
            return len(labels)
        return 1  # already a scalar label

    is_multilabel = any(_label_count(labels) != 1 for labels in y)
    if is_multilabel:
        return "multilabel"

    if n_classes is None:
        scalars = {labels[0] if isinstance(labels, list | tuple | set) else labels for labels in y}
        n_classes = len(scalars)
    return "binary" if n_classes <= 2 else "multiclass"


def flatten_single_label_targets(y: list[Any]) -> list[Any]:
    """Flatten list-of-lists gold labels into scalars for binary/multiclass tasks.

    Raises ``ValueError`` if any example is ambiguous (zero or more than one
    active label) — flattening such an example would silently misrepresent
    multilabel gold as single-label, which this module never does.
    """
    flat: list[Any] = []
    for i, labels in enumerate(y):
        if not isinstance(labels, list | tuple | set):
            flat.append(labels)
            continue
        if len(labels) != 1:
            raise ValueError(
                f"Cannot flatten example at index {i} with {len(labels)} active "
                "label(s) into a single-label target; use task_type='multilabel' "
                "for this data instead."
            )
        flat.append(next(iter(labels)))
    return flat


@dataclass
class FeatureConfig:
    """Configurable sparse feature representation for text classification (§29).

    Supports word n-grams and/or character n-grams, each as either raw
    counts or TF-IDF weighting. When both word and character n-grams are
    enabled, the two feature spaces are combined via
    ``sklearn.pipeline.FeatureUnion`` (fit on TRAIN only, like any other
    vectorizer in this module).
    """

    vectorizer: str = "tfidf"  # "count" | "tfidf"
    use_word_ngrams: bool = True
    ngram_min: int = 1
    ngram_max: int = 1
    use_char_ngrams: bool = False
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    min_df: int | float = 1
    max_df: int | float = 1.0
    max_features: int | None = None

    def __post_init__(self) -> None:
        if self.vectorizer not in VECTORIZER_TYPES:
            raise ValueError(
                f"vectorizer must be one of {VECTORIZER_TYPES}, got {self.vectorizer!r}"
            )
        if not self.use_word_ngrams and not self.use_char_ngrams:
            raise ValueError(
                "FeatureConfig requires at least one of use_word_ngrams or use_char_ngrams"
            )
        if self.use_word_ngrams and self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min must be <= ngram_max")
        if self.use_char_ngrams and self.char_ngram_min > self.char_ngram_max:
            raise ValueError("char_ngram_min must be <= char_ngram_max")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | FeatureConfig | None) -> FeatureConfig:
        if isinstance(data, FeatureConfig):
            return data
        if not data:
            return cls()
        known = {f.name for f in dataclass_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def build_feature_extractor(
    feature_config: FeatureConfig | dict[str, Any] | None,
    preprocessing_config: dict[str, Any] | None = None,
) -> Any:
    """Build the (word and/or char n-gram) vectorizer described by ``feature_config``.

    Returns a single vectorizer when only one feature family is enabled, or
    a ``FeatureUnion`` combining both when word AND character n-grams are
    both enabled. The returned object exposes the standard
    ``fit_transform`` / ``transform`` / ``get_feature_names_out`` API.
    """
    fc = FeatureConfig.from_dict(feature_config)
    is_count = fc.vectorizer == "count"
    word_builder = build_count_vectorizer if is_count else build_tfidf_vectorizer
    char_builder = build_char_count_vectorizer if is_count else build_char_tfidf_vectorizer

    transformers: list[tuple[str, Any]] = []
    if fc.use_word_ngrams:
        word_cfg = dict(preprocessing_config or {})
        word_cfg["ngram_min"] = fc.ngram_min
        word_cfg["ngram_max"] = fc.ngram_max
        word_cfg["min_df"] = fc.min_df
        word_cfg["max_df"] = fc.max_df
        word_cfg["max_features"] = fc.max_features
        transformers.append(("word", word_builder(word_cfg)))
    if fc.use_char_ngrams:
        transformers.append(
            (
                "char",
                char_builder(
                    preprocessing_config,
                    ngram_min=fc.char_ngram_min,
                    ngram_max=fc.char_ngram_max,
                    min_df=fc.min_df,
                    max_df=fc.max_df,
                    max_features=fc.max_features,
                ),
            )
        )

    if not transformers:  # pragma: no cover - guarded by FeatureConfig.__post_init__
        raise ValueError("FeatureConfig must enable at least one feature family")
    if len(transformers) == 1:
        return transformers[0][1]
    return FeatureUnion(transformers)


def _build_base_estimator(
    algorithm: str,
    random_seed: int,
    class_weight: str | dict | None,
    C: float,
    sgd_loss: str = "log_loss",
    nb_alpha: float = 1.0,
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
    if algorithm == "multinomial_nb":
        return MultinomialNB(alpha=nb_alpha)
    if algorithm == "complement_nb":
        return ComplementNB(alpha=nb_alpha)
    if algorithm == "sgd_classifier":
        loss = sgd_loss if sgd_loss in _SGD_LOSSES else "log_loss"
        alpha = 1.0 / C if C and C > 0 else 1.0
        return SGDClassifier(
            loss=loss,
            alpha=alpha,
            max_iter=2000,
            tol=1e-3,
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
    sgd_loss: str = "log_loss",
    nb_alpha: float = 1.0,
) -> Any:
    # MultinomialNB/ComplementNB have no `class_weight` constructor param;
    # weighting for those is applied via `sample_weight` at fit time instead
    # (see `_maybe_sample_weight`), so never pass class_weight to their ctor.
    if algorithm in _NO_CLASS_WEIGHT_PARAM_ALGORITHMS:
        effective_class_weight = None
    else:
        effective_class_weight = class_weight
    base = _build_base_estimator(
        algorithm, random_seed, effective_class_weight, C, sgd_loss=sgd_loss, nb_alpha=nb_alpha
    )
    if task_type == "multilabel":
        return OneVsRestClassifier(base)
    return base


def _maybe_sample_weight(
    algorithm: str,
    class_weight: str | dict | None,
    task_type: str,
    y_train_enc: Any,
) -> np.ndarray | None:
    """Translate ``class_weight`` into per-sample weights for algorithms
    whose estimator has no native ``class_weight`` parameter (NB models).

    Not applied for multilabel (the one-hot target matrix has no single
    well-defined "class" per sample); NB + multilabel + class_weight simply
    trains unweighted in that combination.
    """
    if not class_weight or algorithm not in _NO_CLASS_WEIGHT_PARAM_ALGORITHMS:
        return None
    if task_type == "multilabel":
        return None
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight(class_weight, y_train_enc)


def _predict_proba_matrix(model: Any, X: Any) -> np.ndarray | None:
    """Best-effort ``predict_proba``; returns ``None`` if unavailable/unsupported."""
    if X is None or not hasattr(model, "predict_proba"):
        return None
    try:
        return np.asarray(model.predict_proba(X))
    except (ValueError, AttributeError, NotImplementedError, IndexError):
        return None


# ---------------------------------------------------------------------------
# §33 Per-class / per-label threshold tuning
# ---------------------------------------------------------------------------

_THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 19)


def optimize_thresholds(
    y_val_true_enc: np.ndarray,
    y_val_proba: np.ndarray,
    task_type: str,
    classes: list[Any],
    objective: str = "f1",
) -> dict[str, Any]:
    """Optimize decision thresholds on VALIDATION predictions only (§33).

    - ``binary``: a single scalar threshold on the positive-class
      probability that maximizes F1 on the validation labels.
    - ``multilabel``: one threshold per label, each independently
      maximizing that label's F1 on the validation labels.
    - ``multiclass``: not supported (no single meaningful binary decision
      threshold per class without renormalization); returns a note instead.

    Never uses test labels — this function only ever sees whatever
    ``y_val_true_enc``/``y_val_proba`` the caller passes in, and the caller
    (``_fit_classifier_and_evaluate``) only ever passes validation data.
    """
    if objective != "f1":
        raise ValueError(f"Unsupported threshold objective: {objective!r}; expected 'f1'")

    def _best_threshold(y_true_col: np.ndarray, proba_col: np.ndarray) -> tuple[float, float]:
        best_t, best_score = 0.5, -1.0
        for t in _THRESHOLD_CANDIDATES:
            pred = (proba_col >= t).astype(int)
            score = float(skmetrics.f1_score(y_true_col, pred, zero_division=0))
            if score > best_score:
                best_t, best_score = float(t), score
        return best_t, best_score

    if task_type == "binary":
        proba_positive = y_val_proba[:, 1] if y_val_proba.ndim == 2 else y_val_proba
        threshold, score = _best_threshold(np.asarray(y_val_true_enc), proba_positive)
        return {
            "task_type": "binary",
            "objective": objective,
            "threshold": threshold,
            "val_f1": score,
            "default_threshold": 0.5,
        }

    if task_type == "multilabel":
        y_true_arr = np.asarray(y_val_true_enc)
        thresholds: dict[str, float] = {}
        val_scores: dict[str, float] = {}
        for i, label in enumerate(classes):
            t, score = _best_threshold(y_true_arr[:, i], y_val_proba[:, i])
            thresholds[str(label)] = t
            val_scores[str(label)] = score
        return {
            "task_type": "multilabel",
            "objective": objective,
            "thresholds": thresholds,
            "val_f1": val_scores,
            "default_threshold": 0.5,
        }

    return {
        "task_type": task_type,
        "objective": objective,
        "note": (
            "Threshold tuning is only supported for binary/multilabel tasks "
            "(§33); skipped for multiclass."
        ),
    }


def apply_thresholds(
    y_proba: np.ndarray | None,
    task_type: str,
    threshold_result: dict[str, Any] | None,
    default_pred: np.ndarray,
    classes: list[Any],
) -> np.ndarray:
    """Apply validation-tuned thresholds to new probability predictions (§33).

    Falls back to ``default_pred`` (the classifier's native ``.predict()``
    output) whenever thresholds were not tuned (no validation set, no
    ``predict_proba``, or an unsupported/multiclass task type).
    """
    if not threshold_result or y_proba is None:
        return default_pred
    if task_type == "binary" and "threshold" in threshold_result:
        proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        return (proba_positive >= threshold_result["threshold"]).astype(int)
    if task_type == "multilabel" and "thresholds" in threshold_result:
        thresholds = threshold_result["thresholds"]
        out = np.zeros_like(default_pred)
        for i, label in enumerate(classes):
            t = thresholds.get(str(label), 0.5)
            out[:, i] = (y_proba[:, i] >= t).astype(int)
        return out
    return default_pred


# ---------------------------------------------------------------------------
# §35 Confidence intervals (group-level bootstrap)
# ---------------------------------------------------------------------------


def _score_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float | None:
    try:
        if metric_name == "f1_macro":
            return float(skmetrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "f1_micro":
            return float(skmetrics.f1_score(y_true, y_pred, average="micro", zero_division=0))
        if metric_name == "f1_weighted":
            return float(skmetrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if metric_name == "accuracy":
            return float(skmetrics.accuracy_score(y_true, y_pred))
        if metric_name == "balanced_accuracy":
            return float(skmetrics.balanced_accuracy_score(y_true, y_pred))
        if metric_name == "precision_macro":
            return float(skmetrics.precision_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "recall_macro":
            return float(skmetrics.recall_score(y_true, y_pred, average="macro", zero_division=0))
        return None
    except ValueError:
        return None


def bootstrap_group_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[Any],
    metrics: tuple[str, ...] = ("f1_macro",),
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-level bootstrap confidence intervals for key metrics (§35).

    Resamples whole *groups* (source documents) with replacement — never
    individual text units — because units drawn from the same document are
    not statistically independent observations. This must always be called
    with TEST-partition predictions/labels/groups.
    """
    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(groups)
    unique_groups = np.unique(groups_arr)
    n_groups = len(unique_groups)

    point_estimates = {m: _score_metric(y_true, y_pred, m) for m in metrics}

    samples: dict[str, list[float]] = {m: [] for m in metrics}
    if n_groups >= 2:
        # Precompute each group's row indices once.
        group_indices = {g: np.flatnonzero(groups_arr == g) for g in unique_groups}
        for _ in range(max(0, n_bootstrap)):
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_indices[g] for g in sampled_groups])
            y_t_sample = y_true[idx]
            y_p_sample = y_pred[idx]
            for m in metrics:
                score = _score_metric(y_t_sample, y_p_sample, m)
                if score is not None:
                    samples[m].append(score)

    alpha = 1.0 - confidence_level
    out: dict[str, Any] = {}
    for m in metrics:
        values = samples[m]
        if len(values) < 2:
            out[m] = {
                "estimate": point_estimates[m],
                "lower": None,
                "upper": None,
                "confidence_level": confidence_level,
                "n_bootstrap": len(values),
                "seed": seed,
                "note": "Too few groups (or bootstrap samples) for a stable CI.",
            }
            continue
        out[m] = {
            "estimate": point_estimates[m],
            "lower": float(np.percentile(values, 100 * alpha / 2)),
            "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
            "confidence_level": confidence_level,
            "n_bootstrap": len(values),
            "seed": seed,
        }
    return out


# ---------------------------------------------------------------------------
# §36 Model calibration
# ---------------------------------------------------------------------------


def _binned_ece(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, Any]]]:
    """Equal-width-binned Expected Calibration Error + reliability curve points."""
    confidences = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n == 0:
        return 0.0, []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    curve: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (
            (confidences >= lo) & (confidences <= hi)
            if i == n_bins - 1
            else (confidences >= lo) & (confidences < hi)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
        curve.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_predicted": bin_confidence,
                "empirical_frequency": bin_accuracy,
                "count": count,
            }
        )
    return float(ece), curve


def compute_calibration_summary(
    y_true_enc: Any,
    y_proba: np.ndarray,
    task_type: str,
    classes: list[Any],
    n_bins: int = 10,
) -> dict[str, Any]:
    """ECE + reliability curve diagnostics (§36) for a given prediction set.

    Intended to be called on TEST predictions (primary) or VALIDATION
    predictions — this function is a pure diagnostic and does not fit or
    mutate anything, so either partition is safe to pass in.
    """
    if task_type == "binary":
        proba_positive = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        correct = (np.asarray(y_true_enc) == 1).astype(float)
        ece, curve = _binned_ece(proba_positive, correct, n_bins=n_bins)
        return {
            "task_type": "binary",
            "ece": ece,
            "n_bins": n_bins,
            "reliability_curve": curve,
            "brier_score": float(skmetrics.brier_score_loss(y_true_enc, proba_positive)),
        }

    if task_type == "multilabel":
        y_true_arr = np.asarray(y_true_enc)
        per_label: dict[str, Any] = {}
        eces = []
        for i, label in enumerate(classes):
            ece_i, curve_i = _binned_ece(y_proba[:, i], y_true_arr[:, i], n_bins=n_bins)
            per_label[str(label)] = {"ece": ece_i, "reliability_curve": curve_i}
            eces.append(ece_i)
        return {
            "task_type": "multilabel",
            "ece": _mean_or_none(eces),
            "n_bins": n_bins,
            "per_label": per_label,
        }

    # multiclass: "top-label" calibration diagnostic — confidence is the
    # maximum predicted probability, correctness is whether the argmax
    # prediction matches gold. This is the standard multiclass ECE
    # generalization used in the calibration literature.
    confidences = y_proba.max(axis=1)
    predicted = y_proba.argmax(axis=1)
    correct = (predicted == np.asarray(y_true_enc)).astype(float)
    ece, curve = _binned_ece(confidences, correct, n_bins=n_bins)
    return {"task_type": "multiclass", "ece": ece, "n_bins": n_bins, "reliability_curve": curve}


def fit_probability_calibrator(
    val_true_binary: np.ndarray, val_proba_column: np.ndarray, method: str = "sigmoid"
) -> Any:
    """Fit a 1-D probability calibrator on VALIDATION predictions only (§36).

    ``method="sigmoid"`` fits Platt scaling (a 1-feature logistic
    regression mapping raw probability -> calibrated probability);
    ``method="isotonic"`` fits a monotonic isotonic regression. Returns
    ``None`` when the validation fold has only one observed class (a
    calibrator cannot be fit; the caller should skip recalibration).
    """
    val_true_binary = np.asarray(val_true_binary)
    if len(set(val_true_binary.tolist())) < 2:
        return None
    if method == "isotonic":
        reg = IsotonicRegression(out_of_bounds="clip")
        reg.fit(val_proba_column, val_true_binary)
        return reg
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(val_proba_column).reshape(-1, 1), val_true_binary)
    return lr


def apply_probability_calibrator(calibrator: Any, proba_column: np.ndarray, method: str) -> np.ndarray:
    if calibrator is None:
        return np.asarray(proba_column)
    if method == "isotonic":
        return np.asarray(calibrator.predict(proba_column))
    return np.asarray(calibrator.predict_proba(np.asarray(proba_column).reshape(-1, 1))[:, 1])


def calibrate_and_reevaluate(
    val_true_enc: Any,
    val_proba: np.ndarray | None,
    test_true_enc: Any,
    test_proba: np.ndarray,
    task_type: str,
    classes: list[Any],
    method: str = "sigmoid",
) -> dict[str, Any] | None:
    """Optional Platt/isotonic recalibration (§36): fit on VAL only, apply to TEST.

    For multiclass/multilabel, calibration is fit independently per class /
    per label (one-vs-rest binary calibration) — the standard approach when
    generalizing Platt/isotonic scaling beyond binary classification.
    Returns ``None`` when no validation set/probabilities are available
    (calibration fitting is skipped entirely rather than falling back to
    fitting on test data).
    """
    if val_proba is None or val_true_enc is None or len(np.asarray(val_true_enc)) == 0:
        return None
    if method not in ("sigmoid", "isotonic"):
        method = "sigmoid"

    if task_type == "binary":
        val_pos = val_proba[:, 1] if val_proba.ndim == 2 else val_proba
        test_pos = test_proba[:, 1] if test_proba.ndim == 2 else test_proba
        calibrator = fit_probability_calibrator(np.asarray(val_true_enc), val_pos, method=method)
        before = compute_calibration_summary(test_true_enc, test_proba, task_type, classes)
        if calibrator is None:
            return {
                "method": method,
                "before": before,
                "after": None,
                "note": "Validation fold has a single class; recalibration skipped.",
            }
        calibrated_pos = apply_probability_calibrator(calibrator, test_pos, method)
        calibrated_proba = np.stack([1.0 - calibrated_pos, calibrated_pos], axis=1)
        after = compute_calibration_summary(test_true_enc, calibrated_proba, task_type, classes)
        return {"method": method, "before": before, "after": after}

    if task_type == "multilabel":
        val_true_arr = np.asarray(val_true_enc)
        test_true_arr = np.asarray(test_true_enc)
        calibrated_test_proba = np.array(test_proba, copy=True, dtype=float)
        notes: dict[str, str] = {}
        for i, label in enumerate(classes):
            calibrator = fit_probability_calibrator(val_true_arr[:, i], val_proba[:, i], method=method)
            if calibrator is None:
                notes[str(label)] = "single-class validation fold; not recalibrated"
                continue
            calibrated_test_proba[:, i] = apply_probability_calibrator(
                calibrator, test_proba[:, i], method
            )
        before = compute_calibration_summary(test_true_arr, test_proba, task_type, classes)
        after = compute_calibration_summary(test_true_arr, calibrated_test_proba, task_type, classes)
        result: dict[str, Any] = {"method": method, "before": before, "after": after}
        if notes:
            result["notes"] = notes
        return result

    # multiclass: per-class one-vs-rest Platt/isotonic calibration. Columns
    # are calibrated independently and are NOT renormalized to sum to 1 —
    # this is a diagnostic recalibration, not a replacement decision rule.
    val_true_arr = np.asarray(val_true_enc)
    test_true_arr = np.asarray(test_true_enc)
    calibrated_test_proba = np.array(test_proba, copy=True, dtype=float)
    for i in range(len(classes)):
        val_bin = (val_true_arr == i).astype(int)
        calibrator = fit_probability_calibrator(val_bin, val_proba[:, i], method=method)
        if calibrator is None:
            continue
        calibrated_test_proba[:, i] = apply_probability_calibrator(calibrator, test_proba[:, i], method)
    before = compute_calibration_summary(test_true_arr, test_proba, task_type, classes)
    after = compute_calibration_summary(test_true_arr, calibrated_test_proba, task_type, classes)
    return {
        "method": method,
        "before": before,
        "after": after,
        "note": "Per-class one-vs-rest calibration; probabilities are not renormalized.",
    }


# ---------------------------------------------------------------------------
# §31 Hyperparameter tuning
# ---------------------------------------------------------------------------


def default_hyperparameter_grid(algorithm: str) -> dict[str, list[Any]]:
    """Small, scientifically-motivated default search grids (§31).

    Intentionally tiny by default ("avoid excessive search spaces") — a
    light sweep over the single most impactful regularization/smoothing
    knob per algorithm family. Callers can widen this via ``param_grid``.
    """
    if algorithm in ("logistic_regression", "linear_svm"):
        return {"C": [0.1, 1.0, 10.0]}
    if algorithm == "sgd_classifier":
        return {"alpha": [0.0001, 0.001, 0.01]}
    if algorithm in ("multinomial_nb", "complement_nb"):
        return {"alpha": [0.1, 0.5, 1.0]}
    return {}


def hyperparameter_search(
    X_train_texts: list[str],
    y_train: list[Any],
    groups_train: list[Any],
    X_val_texts: list[str] | None,
    y_val: list[Any] | None,
    *,
    task_type: str,
    algorithm: str,
    feature_config: FeatureConfig | dict[str, Any] | None,
    preprocessing_config: dict[str, Any] | None,
    label_names: list[str] | None,
    class_weight: str | dict | None,
    C: float,
    sgd_loss: str = "log_loss",
    random_seed: int = 42,
    param_grid: dict[str, list[Any]] | None = None,
    search_type: str = "grid",
    n_iter: int = 10,
    scoring: str = "f1_macro",
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Hyperparameter tuning (§31).

    Evaluates each candidate on the VALIDATION set only when one is
    available (``X_val_texts``/``y_val`` non-empty); otherwise falls back to
    ``GroupKFold`` cross-validation over the train+val groups. The held-out
    TEST partition is never touched by this function — callers must not
    pass test data in here.

    Grid values may target either the classifier (``C`` for
    logistic_regression/linear_svm, ``alpha`` for sgd_classifier/NB
    smoothing, ``class_weight``) or the feature extractor (any
    :class:`FeatureConfig` field, e.g. ``max_features``, ``min_df``,
    ``max_df``, ``ngram_max``). Unrecognized keys are ignored.
    """
    fc_base = FeatureConfig.from_dict(feature_config)
    fc_field_names = {f.name for f in dataclass_fields(FeatureConfig)}

    grid = dict(default_hyperparameter_grid(algorithm))
    if param_grid:
        grid.update(param_grid)

    if not grid:
        combos: list[dict[str, Any]] = [{}]
    else:
        combos = list(ParameterGrid(grid))
        if search_type == "random" and len(combos) > max(1, n_iter):
            combos = list(
                ParameterSampler(grid, n_iter=n_iter, random_state=random_seed)
            )

    def _score_split(
        x_tr: list[str], y_tr: list[Any], x_ev: list[str], y_ev: list[Any], combo: dict[str, Any]
    ) -> float | None:
        fc_kwargs = fc_base.to_dict()
        for key, value in combo.items():
            if key in fc_field_names:
                fc_kwargs[key] = value
        try:
            fc_variant = FeatureConfig.from_dict(fc_kwargs)
            vec = build_feature_extractor(fc_variant, preprocessing_config)
            combo_class_weight = combo.get("class_weight", class_weight)
            combo_c = combo.get("C", C)
            combo_alpha = combo.get("alpha")
            if combo_alpha is not None:
                if algorithm == "sgd_classifier":
                    combo_c = 1.0 / combo_alpha if combo_alpha else combo_c
                    nb_alpha = 1.0
                else:
                    nb_alpha = combo_alpha
            else:
                nb_alpha = 1.0

            X_tr_vec = vec.fit_transform(x_tr)
            X_ev_vec = vec.transform(x_ev)

            if task_type == "multilabel":
                mlb = MultiLabelBinarizer(classes=label_names)
                y_tr_enc = mlb.fit_transform(y_tr)
                y_ev_enc = mlb.transform(y_ev)
            else:
                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                if not set(y_ev).issubset(set(le.classes_)):
                    return None
                y_ev_enc = le.transform(y_ev)

            model = _build_model(
                algorithm,
                task_type,
                random_seed,
                combo_class_weight,
                combo_c,
                sgd_loss=sgd_loss,
                nb_alpha=nb_alpha,
            )
            sample_weight = _maybe_sample_weight(algorithm, combo_class_weight, task_type, y_tr_enc)
            if sample_weight is not None:
                model.fit(X_tr_vec, y_tr_enc, sample_weight=sample_weight)
            else:
                model.fit(X_tr_vec, y_tr_enc)
            y_pred = model.predict(X_ev_vec)
            return _score_metric(y_ev_enc, y_pred, scoring)
        except (ValueError, IndexError):
            return None

    results: list[dict[str, Any]] = []

    if X_val_texts:
        method_used = "validation_holdout"
        for combo in combos:
            score = _score_split(X_train_texts, y_train, X_val_texts, y_val or [], combo)
            results.append({"params": _to_native(combo), "score": score})
    else:
        n_groups = len(set(groups_train))
        if n_groups < 2:
            return {
                "enabled": True,
                "method": "skipped",
                "search_type": search_type,
                "scoring": scoring,
                "param_grid": _to_native(grid),
                "results": [],
                "best_params": {},
                "best_score": None,
                "note": "Hyperparameter search skipped: fewer than 2 distinct groups available.",
            }
        method_used = "group_kfold_train_val"
        folds = max(2, min(cv_folds, n_groups))
        gkf = GroupKFold(n_splits=folds)
        idx_arr = np.arange(len(X_train_texts))
        for combo in combos:
            fold_scores: list[float] = []
            for tr_idx, ev_idx in gkf.split(idx_arr, groups=groups_train):
                score = _score_split(
                    [X_train_texts[i] for i in tr_idx],
                    [y_train[i] for i in tr_idx],
                    [X_train_texts[i] for i in ev_idx],
                    [y_train[i] for i in ev_idx],
                    combo,
                )
                if score is not None:
                    fold_scores.append(score)
            results.append(
                {
                    "params": _to_native(combo),
                    "score": _mean_or_none(fold_scores),
                    "n_folds_evaluated": len(fold_scores),
                }
            )

    valid_results = [r for r in results if r["score"] is not None]
    if valid_results:
        best = max(valid_results, key=lambda r: r["score"])
        best_params, best_score = best["params"], best["score"]
    else:
        best_params, best_score = {}, None

    return {
        "enabled": True,
        "method": method_used,
        "search_type": search_type,
        "scoring": scoring,
        "param_grid": _to_native(grid),
        "results": results,
        "best_params": best_params,
        "best_score": best_score,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[Any],
    task_type: str,
    model: Any,
    X_test_vec: Any,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification metrics (§34).

    Beyond the original accuracy/precision/recall/F1 (macro/micro/weighted)
    and per-class/confusion-matrix reporting, this additionally computes —
    only where statistically defined for the current ``task_type`` —
    balanced accuracy, Matthews correlation coefficient, log loss, Brier
    score, PR-AUC (binary), and Hamming loss / subset accuracy (multilabel).
    Metrics that are undefined for the current task (e.g. balanced accuracy
    for multilabel, PR-AUC for multiclass) are simply omitted rather than
    returned as misleading placeholder values.
    """
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
        # Multilabel-specific metrics (§34): Hamming loss (fraction of
        # individual label mismatches) and subset accuracy (exact match of
        # the full label set per example — `accuracy_score` on a 2D
        # indicator matrix already computes this, but expose it explicitly
        # under its own name to avoid ambiguity with `accuracy` elsewhere).
        metrics_out["hamming_loss"] = float(skmetrics.hamming_loss(y_true, y_pred))
        metrics_out["subset_accuracy"] = float(skmetrics.accuracy_score(y_true, y_pred))
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
        # Balanced accuracy / MCC are only well-defined for single-label
        # (binary/multiclass) predictions — omitted for multilabel.
        metrics_out["balanced_accuracy"] = float(skmetrics.balanced_accuracy_score(y_true, y_pred))
        try:
            metrics_out["mcc"] = float(skmetrics.matthews_corrcoef(y_true, y_pred))
        except ValueError:
            metrics_out["mcc"] = None

    if task_type == "multilabel":
        metrics_out["multilabel_confusion_matrices"] = {
            str(cls): matrix.tolist()
            for cls, matrix in zip(
                classes,
                skmetrics.multilabel_confusion_matrix(y_true, y_pred),
                strict=True,
            )
        }

    # Probability-dependent metrics (§34): reuse an already-computed `proba`
    # matrix when the caller has one (e.g. from threshold tuning), otherwise
    # compute it lazily here. Never raises: any of these are simply omitted
    # when the model has no calibrated probability output.
    proba = y_proba
    if proba is None and hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X_test_vec))
        except (ValueError, AttributeError, IndexError):
            proba = None
    elif proba is not None:
        proba = np.asarray(proba)

    if proba is not None and task_type == "binary":
        try:
            proba_positive = proba[:, 1] if proba.ndim == 2 else proba
            if len(set(np.asarray(y_true).tolist())) > 1:
                metrics_out["roc_auc"] = float(skmetrics.roc_auc_score(y_true, proba_positive))
                metrics_out["pr_auc"] = float(
                    skmetrics.average_precision_score(y_true, proba_positive)
                )
        except (ValueError, IndexError):
            pass

    if proba is not None:
        # Log loss: full-support labels are pinned via `labels=` so metrics
        # remain defined even when a fold happens to observe only one class.
        try:
            if task_type == "multilabel":
                y_true_arr = np.asarray(y_true)
                per_label_ll = [
                    float(skmetrics.log_loss(y_true_arr[:, i], proba[:, i], labels=[0, 1]))
                    for i in range(proba.shape[1])
                ]
                metrics_out["log_loss"] = _mean_or_none(per_label_ll)
            else:
                metrics_out["log_loss"] = float(
                    skmetrics.log_loss(y_true, proba, labels=list(range(len(classes))))
                )
        except ValueError:
            metrics_out["log_loss"] = None

        # Brier score: binary uses the standard single-probability formula;
        # multiclass uses the multi-class generalization (mean squared error
        # between the one-hot gold vector and the full probability vector);
        # multilabel averages the per-label binary Brier score.
        try:
            if task_type == "binary":
                proba_positive = proba[:, 1] if proba.ndim == 2 else proba
                metrics_out["brier_score"] = float(
                    skmetrics.brier_score_loss(y_true, proba_positive)
                )
            elif task_type == "multiclass":
                y_true_onehot = np.zeros_like(proba, dtype=float)
                y_true_onehot[np.arange(len(y_true)), np.asarray(y_true)] = 1.0
                metrics_out["brier_score"] = float(
                    np.mean(np.sum((proba - y_true_onehot) ** 2, axis=1))
                )
            else:
                y_true_arr = np.asarray(y_true)
                per_label_brier = [
                    float(skmetrics.brier_score_loss(y_true_arr[:, i], proba[:, i]))
                    for i in range(proba.shape[1])
                ]
                metrics_out["brier_score"] = _mean_or_none(per_label_brier)
        except ValueError:
            metrics_out["brier_score"] = None

    return metrics_out


def _fit_classifier_and_evaluate(
    vectorizer: Any,
    X_train_texts: list[str],
    y_train: list[Any],
    X_test_texts: list[str],
    y_test: list[Any],
    task_type: str,
    algorithm: str,
    label_names: list[str] | None,
    class_weight: str | dict | None,
    C: float,
    random_seed: int,
    sgd_loss: str,
    nb_alpha: float = 1.0,
    X_val_texts: list[str] | None = None,
    y_val: list[Any] | None = None,
    groups_test: list[Any] | None = None,
    tune_thresholds: bool = True,
    threshold_objective: str = "f1",
    n_bootstrap: int = 200,
    ci_confidence_level: float = 0.95,
    ci_metrics: tuple[str, ...] = ("f1_macro",),
    calibration_method: str = "sigmoid",
) -> dict[str, Any]:
    """Shared fit/evaluate core for :func:`fit_tfidf_classifier` and
    :func:`fit_text_classifier`. ``vectorizer`` is any already-constructed
    (unfitted) vectorizer/FeatureUnion exposing ``fit_transform``/``transform``.

    Optional additive stages (all leakage-safe, all off/no-op when their
    prerequisite data is absent):

    - §33 threshold tuning: tuned on ``X_val_texts``/``y_val`` only, then
      applied when scoring ``X_test_texts``/``y_test``.
    - §35 bootstrap CIs: computed over TEST predictions, resampled at the
      ``groups_test`` (source-document) level.
    - §36 calibration: ECE/reliability diagnostics on TEST, plus an
      optional Platt/isotonic recalibration fit on VAL and evaluated on
      TEST.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type!r}; expected one of {TASK_TYPES}")

    # CRITICAL: fit on train only, transform val/test with the train-fitted vectorizer.
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)
    X_val_vec = vectorizer.transform(X_val_texts) if X_val_texts else None

    label_encoder: LabelEncoder | None = None
    mlb: MultiLabelBinarizer | None = None

    if task_type == "multilabel":
        mlb = MultiLabelBinarizer(classes=label_names)
        y_train_enc = mlb.fit_transform(y_train)
        y_test_enc = mlb.transform(y_test)
        y_val_enc = mlb.transform(y_val) if y_val else None
        classes = list(mlb.classes_)
    else:
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
        y_val_enc = label_encoder.transform(y_val) if y_val else None
        classes = list(label_encoder.classes_)

    model = _build_model(
        algorithm, task_type, random_seed, class_weight, C, sgd_loss=sgd_loss, nb_alpha=nb_alpha
    )
    sample_weight = _maybe_sample_weight(algorithm, class_weight, task_type, y_train_enc)
    if sample_weight is not None:
        model.fit(X_train_vec, y_train_enc, sample_weight=sample_weight)
    else:
        model.fit(X_train_vec, y_train_enc)
    y_pred_default = model.predict(X_test_vec)

    y_proba_test = _predict_proba_matrix(model, X_test_vec)
    y_proba_val = _predict_proba_matrix(model, X_val_vec)

    # §33: tune thresholds on VAL only, apply to TEST.
    threshold_result: dict[str, Any] | None = None
    y_pred_final = y_pred_default
    if tune_thresholds:
        if task_type not in ("binary", "multilabel"):
            threshold_result = {
                "task_type": task_type,
                "note": "Threshold tuning not applicable to multiclass tasks (§33).",
            }
        elif y_val_enc is None:
            threshold_result = {
                "task_type": task_type,
                "objective": threshold_objective,
                "note": "Threshold tuning skipped: no validation set available.",
            }
        elif y_proba_val is None:
            threshold_result = {
                "task_type": task_type,
                "objective": threshold_objective,
                "note": "Threshold tuning skipped: model has no predict_proba.",
            }
        else:
            threshold_result = optimize_thresholds(
                y_val_enc, y_proba_val, task_type, classes, objective=threshold_objective
            )
            if y_proba_test is not None:
                y_pred_final = apply_thresholds(
                    y_proba_test, task_type, threshold_result, y_pred_default, classes
                )

    metrics_out = _classification_metrics(
        y_test_enc, y_pred_final, classes, task_type, model, X_test_vec, y_proba=y_proba_test
    )
    if threshold_result is not None:
        metrics_out["thresholds"] = threshold_result

    # §35: group-level bootstrap CIs on TEST predictions.
    if groups_test:
        metrics_out["confidence_intervals"] = bootstrap_group_metric_ci(
            y_test_enc,
            y_pred_final,
            groups_test,
            metrics=ci_metrics,
            n_bootstrap=n_bootstrap,
            confidence_level=ci_confidence_level,
            seed=random_seed,
        )
    else:
        metrics_out["confidence_intervals"] = {
            "note": "Bootstrap CI skipped: no group ids provided for the test partition."
        }

    # §36: calibration diagnostics (+ optional VAL-fit recalibration evaluated on TEST).
    if y_proba_test is not None:
        calibration_summary = compute_calibration_summary(y_test_enc, y_proba_test, task_type, classes)
        recalibration = None
        if y_val_enc is not None and y_proba_val is not None:
            recalibration = calibrate_and_reevaluate(
                y_val_enc,
                y_proba_val,
                y_test_enc,
                y_proba_test,
                task_type,
                classes,
                method=calibration_method,
            )
        metrics_out["calibration"] = {
            "diagnostics": calibration_summary,
            "recalibration": recalibration,
        }
    else:
        metrics_out["calibration"] = {"note": "Calibration skipped: model has no predict_proba."}

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
        "n_val": len(X_val_texts) if X_val_texts else 0,
        "n_test": len(X_test_texts),
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
    }


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
    sgd_loss: str = "log_loss",
    nb_alpha: float = 1.0,
    X_val_texts: list[str] | None = None,
    y_val: list[Any] | None = None,
    groups_test: list[Any] | None = None,
    tune_thresholds: bool = True,
    threshold_objective: str = "f1",
    n_bootstrap: int = 200,
    ci_confidence_level: float = 0.95,
    ci_metrics: tuple[str, ...] = ("f1_macro",),
    calibration_method: str = "sigmoid",
) -> dict[str, Any]:
    """Fit a TF-IDF (word n-grams only) + classifier, respecting the train/test boundary.

    Kept for backward compatibility and simple word-TF-IDF use cases; prefer
    :func:`fit_text_classifier` for configurable feature representations
    (count vs. TF-IDF, word and/or character n-grams; see ``FeatureConfig``).

    The vectorizer is fit ONLY on ``X_train_texts``; ``X_test_texts``/
    ``X_val_texts`` are only ever transformed with that already-fitted
    vectorizer. See :func:`_fit_classifier_and_evaluate` for the optional
    threshold-tuning/CI/calibration stages (§33/§35/§36).

    Returns a dict with the fitted ``vectorizer``, ``model``, label
    encoders, ``classes``, and computed ``metrics``.
    """
    vectorizer = build_tfidf_vectorizer(preprocessing_config)
    if vectorizer_kwargs:
        vectorizer.set_params(**vectorizer_kwargs)

    return _fit_classifier_and_evaluate(
        vectorizer,
        X_train_texts,
        y_train,
        X_test_texts,
        y_test,
        task_type,
        algorithm,
        label_names,
        class_weight,
        C,
        random_seed,
        sgd_loss,
        nb_alpha=nb_alpha,
        X_val_texts=X_val_texts,
        y_val=y_val,
        groups_test=groups_test,
        tune_thresholds=tune_thresholds,
        threshold_objective=threshold_objective,
        n_bootstrap=n_bootstrap,
        ci_confidence_level=ci_confidence_level,
        ci_metrics=ci_metrics,
        calibration_method=calibration_method,
    )


def fit_text_classifier(
    X_train_texts: list[str],
    y_train: list[Any],
    X_test_texts: list[str],
    y_test: list[Any],
    task_type: str,
    algorithm: str = "logistic_regression",
    feature_config: FeatureConfig | dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    label_names: list[str] | None = None,
    class_weight: str | dict | None = None,
    C: float = 1.0,
    random_seed: int = 42,
    sgd_loss: str = "log_loss",
    nb_alpha: float = 1.0,
    X_val_texts: list[str] | None = None,
    y_val: list[Any] | None = None,
    groups_test: list[Any] | None = None,
    tune_thresholds: bool = True,
    threshold_objective: str = "f1",
    n_bootstrap: int = 200,
    ci_confidence_level: float = 0.95,
    ci_metrics: tuple[str, ...] = ("f1_macro",),
    calibration_method: str = "sigmoid",
) -> dict[str, Any]:
    """Fit a configurable-feature classifier, respecting the train/test boundary (§29).

    ``feature_config`` (a :class:`FeatureConfig` or an equivalent dict)
    selects count vs. TF-IDF weighting and word and/or character n-grams
    (combined via ``FeatureUnion`` when both are enabled). The resulting
    feature extractor is fit ONLY on ``X_train_texts``; ``X_test_texts``/
    ``X_val_texts`` are only ever transformed with that already-fitted
    extractor.

    Optional additive parameters implement §31/§33/§35/§36 — all no-ops
    (with a persisted explanatory note) when their prerequisite data is
    unavailable (e.g. no validation partition):

    - ``nb_alpha``: Naive Bayes (multinomial/complement) smoothing parameter.
    - ``X_val_texts``/``y_val``: validation partition used ONLY for
      threshold tuning (§33) and optional calibration fitting (§36) — never
      for fitting the feature extractor or the classifier itself.
    - ``groups_test``: source-document ids for the TEST partition, used for
      group-level bootstrap CIs (§35).
    - ``tune_thresholds``/``threshold_objective``: §33 threshold tuning.
    - ``n_bootstrap``/``ci_confidence_level``/``ci_metrics``: §35 bootstrap CIs.
    - ``calibration_method``: ``"sigmoid"`` (Platt) or ``"isotonic"`` for
      the optional §36 recalibration step.
    """
    fc = FeatureConfig.from_dict(feature_config)
    vectorizer = build_feature_extractor(fc, preprocessing_config)

    result = _fit_classifier_and_evaluate(
        vectorizer,
        X_train_texts,
        y_train,
        X_test_texts,
        y_test,
        task_type,
        algorithm,
        label_names,
        class_weight,
        C,
        random_seed,
        sgd_loss,
        nb_alpha=nb_alpha,
        X_val_texts=X_val_texts,
        y_val=y_val,
        groups_test=groups_test,
        tune_thresholds=tune_thresholds,
        threshold_objective=threshold_objective,
        n_bootstrap=n_bootstrap,
        ci_confidence_level=ci_confidence_level,
        ci_metrics=ci_metrics,
        calibration_method=calibration_method,
    )
    result["feature_config"] = fc.to_dict()
    return result


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
    label_names: list[str] | None = None,
    thresholds: dict[str, Any] | None = None,
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

    ``thresholds`` (the persisted ``metrics["thresholds"]`` output of
    :func:`optimize_thresholds`/training — see §33) is applied here so the
    same validation-tuned decision boundary used when scoring TEST is also
    used at inference time, instead of the classifier's raw 0.5 default.
    Ignored when absent, when the model has no ``predict_proba``, or for
    multiclass tasks (no single tuned threshold applies).
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type!r}; expected one of {TASK_TYPES}")

    X_vec = vectorizer.transform(texts)
    predictions = model.predict(X_vec)
    results: list[dict[str, Any]] = []

    if task_type == "binary":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_vec)
            threshold = (
                thresholds.get("threshold") if isinstance(thresholds, dict) else None
            )
            for i in range(len(texts)):
                p_positive = float(proba[i][-1])
                prediction = (
                    (1 if p_positive >= threshold else 0)
                    if threshold is not None
                    else _to_native(predictions[i])
                )
                results.append(
                    {
                        "prediction": prediction,
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
        label_thresholds = (
            thresholds.get("thresholds") if isinstance(thresholds, dict) else None
        )
        for i in range(len(texts)):
            p = np.clip(np.asarray(proba[i]).reshape(-1), 1e-12, 1.0 - 1e-12)
            binary_entropy = float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))
            if label_thresholds and label_names:
                prediction = np.array(
                    [
                        1 if p[j] >= label_thresholds.get(str(label_names[j]), 0.5) else 0
                        for j in range(len(p))
                    ]
                )
            else:
                prediction = np.asarray(predictions[i]).reshape(-1)
            results.append(
                {
                    "prediction": _to_native(prediction),
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
