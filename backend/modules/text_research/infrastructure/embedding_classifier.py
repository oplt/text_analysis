"""Dense-embedding classifiers for research baselines.

Sits between TF-IDF sparse baselines and transformer embeddings: fit a
sklearn linear model directly on precomputed dense vectors (e.g. from
:class:`~backend.modules.text_research.infrastructure.embeddings.HashingEmbeddingProvider`
or an optional sentence-transformer backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ALGORITHMS = ("logistic_regression", "linear_svm")


def fit_embedding_classifier(
    embeddings: np.ndarray | list[list[float]],
    y: list[Any],
    algorithm: str = "logistic_regression",
    *,
    class_weight: str | dict | None = None,
    C: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit a linear classifier on dense embedding vectors."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; expected one of {ALGORITHMS}")

    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array of shape (n_samples, n_features)")
    if len(y) != X.shape[0]:
        raise ValueError("embeddings and y must have the same number of samples")

    if algorithm == "logistic_regression":
        model: LogisticRegression | LinearSVC = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=random_seed,
            max_iter=1000,
        )
    else:
        model = LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=random_seed,
        )

    model.fit(X, y)
    predictions = model.predict(X)

    return {
        "algorithm": algorithm,
        "model": model,
        "predictions": predictions.tolist(),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }
