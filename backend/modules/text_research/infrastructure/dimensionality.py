"""Dimensionality reduction for visual exploration (§44).

TruncatedSVD (sparse-safe, works directly on a TF-IDF matrix) and PCA
(dense features) project a feature matrix down to 2 or 3 coordinates per
unit, for scatter-plot exploration in the frontend.

Limitation (documented, not hidden): UMAP/t-SNE are **not implemented**.
Neither ``umap-learn`` nor a project-vetted t-SNE workflow is a current
dependency (see ``backend/pyproject.toml``); adding either is a product
decision, not a silent substitution. TruncatedSVD/PCA coordinates are
linear projections — treat them (and any future UMAP/t-SNE coordinates) as
exploratory visualization aids only, never as inferential statistics (§44).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

METHODS = ("svd", "pca")


def _to_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def reduce_dimensions(
    matrix: Any,
    unit_ids: list[str],
    *,
    method: str = "svd",
    n_components: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Project ``matrix`` (sparse or dense) to ``n_components`` coordinates.

    ``method="svd"`` (default) — TruncatedSVD, safe on sparse TF-IDF
    matrices without densifying them; the natural choice for lexical
    features (§44).

    ``method="pca"`` — requires a dense matrix (e.g. embeddings or an
    already-reduced representation); densifies sparse input, so callers
    with large vocabularies should prefer ``svd``.
    """
    algo = method.lower()
    if algo not in METHODS:
        raise ValueError(f"Unsupported dimensionality reduction method {method!r}; expected one of {METHODS}")
    if matrix.shape[0] != len(unit_ids):
        raise ValueError("matrix row count must match len(unit_ids)")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for visual exploration")

    n_samples, n_features = matrix.shape
    effective_components = max(1, min(n_components, n_features - 1 if algo == "svd" else n_features, n_samples - 1 if n_samples > 1 else 1))

    if algo == "svd":
        model = TruncatedSVD(n_components=effective_components, random_state=random_seed)
        coords = model.fit_transform(matrix)
        explained = model.explained_variance_ratio_
    else:
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        model = PCA(n_components=effective_components, random_state=random_seed)
        coords = model.fit_transform(dense)
        explained = model.explained_variance_ratio_

    # Pad to requested dimensionality with zeros if the corpus is too small
    # to support the full n_components (never silently return fewer axes
    # than requested — the frontend expects a fixed shape).
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])

    axis_names = [f"axis_{i + 1}" for i in range(n_components)]
    points = [
        {"unit_id": unit_ids[i], **{axis_names[j]: float(coords[i, j]) for j in range(n_components)}}
        for i in range(n_samples)
    ]

    return {
        "method": algo,
        "n_components": n_components,
        "effective_components": int(effective_components),
        "axis_names": axis_names,
        "points": points,
        "explained_variance_ratio": _to_native(explained),
        "total_explained_variance": float(np.sum(explained)),
        "note": (
            "Linear projection for exploratory visualization only; do not treat axis "
            "positions or distances as inferential statistics. UMAP/t-SNE are not "
            "implemented (not project dependencies)."
        ),
    }
