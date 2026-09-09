"""Unsupervised document clustering over sparse lexical features (§43).

KMeans / MiniBatchKMeans on TF-IDF, optionally preceded by TruncatedSVD
(latent-semantic) dimensionality reduction for speed/noise-reduction on
large vocabularies. Cluster *labels* are always plain integers (``0``,
``1``, ...); this module never assigns a human-meaningful name to a
cluster — that stays a user decision (per the platform's "do not interpret
group meaning automatically" principle).

Limitation (documented, not hidden): embedding-based clustering (KMeans on
dense semantic embeddings, or HDBSCAN) is NOT implemented here. HDBSCAN is
not a project dependency, and semantic-embedding clustering depends on
:mod:`infrastructure.embeddings`, which itself degrades to "unavailable"
without a configured embedding provider (see §45). Callers who already have
embeddings can still use :func:`kmeans_cluster` directly on a dense
embedding matrix — sparse TF-IDF is simply the default, dependency-light
path required by §43.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from backend.modules.text_research.infrastructure.preprocessing import build_tfidf_vectorizer

ALGORITHMS = ("kmeans", "minibatch_kmeans")


def _to_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_native(v) for v in value]
    return value


def build_tfidf_matrix(
    texts: list[str], config: dict[str, Any] | None = None
) -> tuple[Any, list[str]]:
    """Fit a TF-IDF matrix over ``texts``; returns (sparse matrix, feature_names)."""
    vectorizer = build_tfidf_vectorizer(config)
    matrix = vectorizer.fit_transform(texts)
    return matrix, list(vectorizer.get_feature_names_out())


def kmeans_cluster(
    matrix: Any,
    *,
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    random_seed: int = 42,
    n_init: int | str = 10,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Run KMeans/MiniBatchKMeans on a feature matrix (sparse or dense).

    Works on any 2D feature matrix (TF-IDF, TruncatedSVD output, or
    externally supplied embeddings) — this function does not know or care
    where the features came from.
    """
    algo = algorithm.lower()
    if algo not in ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm {algorithm!r}; expected one of {ALGORITHMS}")

    n_samples = matrix.shape[0]
    effective_k = max(1, min(n_clusters, n_samples))

    if algo == "kmeans":
        model = KMeans(
            n_clusters=effective_k, random_state=random_seed, n_init=n_init, max_iter=max_iter
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=effective_k, random_state=random_seed, n_init=n_init, max_iter=max_iter
        )
    labels = model.fit_predict(matrix)
    inertia = float(model.inertia_) if hasattr(model, "inertia_") else None

    silhouette: float | None = None
    if 1 < effective_k < n_samples:
        from sklearn.metrics import silhouette_score

        try:
            silhouette = float(silhouette_score(matrix, labels))
        except ValueError:
            silhouette = None

    return {
        "algorithm": algo,
        "n_clusters": effective_k,
        "requested_n_clusters": n_clusters,
        "labels": _to_native(labels),
        "inertia": inertia,
        "silhouette_score": silhouette,
        "silhouette_note": (
            "Requires 2 <= n_clusters < n_samples; null otherwise (not a failure, just "
            "not defined for degenerate cluster counts)."
        ),
        "model": model,
        "random_seed": random_seed,
    }


def top_terms_per_cluster(
    matrix: Any,
    labels: list[int],
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Characteristic terms per cluster: mean TF-IDF weight within the cluster.

    Only meaningful when ``matrix`` is directly interpretable in the
    ``feature_names`` space (i.e. the raw TF-IDF matrix, not a
    TruncatedSVD-reduced one). Callers that reduced dimensionality before
    clustering should pass the *original* TF-IDF matrix here for term
    extraction.
    """
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    labels_arr = np.asarray(labels)
    result: dict[str, list[dict[str, Any]]] = {}
    for cluster_id in sorted(set(labels_arr.tolist())):
        mask = labels_arr == cluster_id
        cluster_rows = matrix[mask]
        mean_weights = np.asarray(cluster_rows.mean(axis=0)).ravel()
        top_indices = np.argsort(-mean_weights)[:top_n]
        result[str(cluster_id)] = [
            {"term": feature_names[idx], "mean_weight": float(mean_weights[idx])}
            for idx in top_indices
            if mean_weights[idx] > 0
        ]
    return result


def cluster_sizes(labels: list[int]) -> dict[str, int]:
    labels_arr = np.asarray(labels)
    return {str(cid): int((labels_arr == cid).sum()) for cid in sorted(set(labels_arr.tolist()))}


def representative_units(
    matrix: Any,
    labels: list[int],
    unit_ids: list[str],
    centers: Any,
    *,
    top_n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Units closest to each cluster centroid (cosine distance on TF-IDF-like space)."""
    from sklearn.metrics.pairwise import cosine_distances

    labels_arr = np.asarray(labels)
    result: dict[str, list[dict[str, Any]]] = {}
    for cluster_id in sorted(set(labels_arr.tolist())):
        mask = np.where(labels_arr == cluster_id)[0]
        if len(mask) == 0:
            result[str(cluster_id)] = []
            continue
        cluster_rows = matrix[mask]
        center = centers[cluster_id].reshape(1, -1)
        distances = cosine_distances(cluster_rows, center).ravel()
        order = np.argsort(distances)[:top_n]
        result[str(cluster_id)] = [
            {"unit_id": unit_ids[mask[i]], "distance_to_centroid": float(distances[i])}
            for i in order
        ]
    return result


def run_clustering(
    texts: list[str],
    unit_ids: list[str],
    *,
    n_clusters: int = 5,
    algorithm: str = "kmeans",
    config: dict[str, Any] | None = None,
    use_svd: bool = False,
    svd_components: int = 50,
    random_seed: int = 42,
    top_n_terms: int = 10,
) -> dict[str, Any]:
    """End-to-end TF-IDF clustering pipeline (§43): fit → cluster → describe.

    When ``use_svd`` is set, clustering runs on a TruncatedSVD projection of
    the TF-IDF matrix (faster, denoised) but characteristic terms are still
    computed against the original TF-IDF space so results stay interpretable
    in vocabulary terms.
    """
    if len(texts) != len(unit_ids):
        raise ValueError("texts and unit_ids must have the same length")
    if not texts:
        raise ValueError("texts must be non-empty to run clustering")

    tfidf_matrix, feature_names = build_tfidf_matrix(texts, config)
    if tfidf_matrix.shape[1] == 0:
        raise ValueError("Vocabulary is empty after preprocessing; cannot cluster")

    cluster_matrix = tfidf_matrix
    svd_info: dict[str, Any] | None = None
    if use_svd:
        effective_components = max(1, min(svd_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1))
        svd = TruncatedSVD(n_components=effective_components, random_state=random_seed)
        cluster_matrix = svd.fit_transform(tfidf_matrix)
        svd_info = {
            "n_components": effective_components,
            "explained_variance_ratio": _to_native(svd.explained_variance_ratio_),
            "total_explained_variance": float(svd.explained_variance_ratio_.sum()),
        }

    clustering = kmeans_cluster(
        cluster_matrix, n_clusters=n_clusters, algorithm=algorithm, random_seed=random_seed
    )
    labels = clustering["labels"]

    centers = clustering["model"].cluster_centers_
    if use_svd:
        # Recompute centroids in original TF-IDF space (mean of member rows)
        # for representative-unit distance so results stay in an
        # interpretable, vocabulary-anchored space.
        tfidf_dense_centers = np.zeros((clustering["n_clusters"], tfidf_matrix.shape[1]))
        labels_arr = np.asarray(labels)
        for cid in range(clustering["n_clusters"]):
            mask = labels_arr == cid
            if mask.any():
                tfidf_dense_centers[cid] = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        centers_for_terms = tfidf_dense_centers
    else:
        centers_for_terms = centers

    return {
        "algorithm": clustering["algorithm"],
        "n_clusters": clustering["n_clusters"],
        "requested_n_clusters": n_clusters,
        "random_seed": random_seed,
        "unit_ids": list(unit_ids),
        "labels": labels,
        "cluster_sizes": cluster_sizes(labels),
        "top_terms": top_terms_per_cluster(tfidf_matrix, labels, feature_names, top_n=top_n_terms),
        "representative_units": representative_units(
            tfidf_matrix, labels, unit_ids, centers_for_terms, top_n=3
        ),
        "inertia": clustering["inertia"],
        "silhouette_score": clustering["silhouette_score"],
        "dimensionality_reduction": {"used_svd": use_svd, **(svd_info or {})},
        "note": (
            "Cluster indices are arbitrary and carry no inherent meaning; label them "
            "yourself after inspecting top_terms/representative_units."
        ),
        "tfidf_matrix": tfidf_matrix,
        "feature_names": feature_names,
    }
