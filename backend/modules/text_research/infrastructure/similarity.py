"""Text similarity engines for research text analysis.

Two lexical methods are always available and require no external services:

* ``tfidf_cosine`` — cosine similarity on TF-IDF vectors (scikit-learn)
* ``jaccard`` — Jaccard similarity on token *sets*

A third, optional method is supported for callers that already have
embedding vectors from an existing platform embedding provider:

* ``embedding_cosine`` — cosine similarity on caller-supplied dense vectors

Embeddings are **never** computed inside this module and never silently
substituted for the lexical methods. If ``embedding_cosine`` is requested
without vectors, a clear ``ValueError`` is raised — callers must supply
embeddings explicitly (e.g. from an existing embedding provider) or use
``tfidf_cosine``/``jaccard`` instead. This keeps semantic similarity an
optional analysis family, never forced into the standard path.

Four similarity *modes* are supported, all implemented on top of the same
three methods (nothing here hardcodes research concepts — "document" vs
"unit" is purely a matter of what ``ids``/``tokenized`` the caller passes in):

* ``pairwise`` — document-to-document / unit-to-unit similarity within one set
* ``query`` — query-to-document(s) similarity
* ``group_centroid`` — similarity between group centroids, or of each item to
  its own group's centroid (caller-selected ``target``)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity

SIMILARITY_METHODS: frozenset[str] = frozenset({"tfidf_cosine", "jaccard", "embedding_cosine"})
SIMILARITY_MODES: frozenset[str] = frozenset({"pairwise", "query", "group_centroid"})
CENTROID_TARGETS: frozenset[str] = frozenset({"between_groups", "item_to_own_group"})

_EMBEDDING_ERROR = (
    "embedding_cosine requires precomputed embedding vectors. This platform does "
    "not compute embeddings inside the similarity engine and never forces "
    "embeddings into the standard (lexical) similarity path. Supply `embeddings` "
    "explicitly (e.g. from an existing embedding provider), or use "
    "method='tfidf_cosine' or method='jaccard' instead."
)


def normalize_similarity_method(method: str | None) -> str:
    key = (method or "tfidf_cosine").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cosine": "tfidf_cosine",
        "tfidf": "tfidf_cosine",
        "tf_idf_cosine": "tfidf_cosine",
        "tf_idf": "tfidf_cosine",
        "jaccard_similarity": "jaccard",
        "token_jaccard": "jaccard",
        "embedding": "embedding_cosine",
        "embeddings": "embedding_cosine",
        "vector": "embedding_cosine",
        "vector_cosine": "embedding_cosine",
        "semantic": "embedding_cosine",
    }
    canonical = aliases.get(key, key)
    if canonical not in SIMILARITY_METHODS:
        raise ValueError(
            f"Unsupported similarity method {method!r}; expected one of "
            f"{sorted(SIMILARITY_METHODS)}"
        )
    return canonical


def normalize_similarity_mode(mode: str | None) -> str:
    key = (mode or "pairwise").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "document_to_document": "pairwise",
        "doc_to_doc": "pairwise",
        "unit_to_unit": "pairwise",
        "documents": "pairwise",
        "units": "pairwise",
        "query_to_document": "query",
        "query_to_documents": "query",
        "centroid": "group_centroid",
        "group": "group_centroid",
        "centroids": "group_centroid",
    }
    canonical = aliases.get(key, key)
    if canonical not in SIMILARITY_MODES:
        raise ValueError(
            f"Unsupported similarity mode {mode!r}; expected one of {sorted(SIMILARITY_MODES)}"
        )
    return canonical


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two token *sets* (not multisets)."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _build_tfidf_matrix(
    tokenized: list[list[str]], **tfidf_kwargs: Any
) -> tuple[Any, TfidfVectorizer]:
    """Fit a TF-IDF matrix over already-tokenized documents.

    Uses the same tokenizer-bypass convention as the DFM builder: documents
    are pre-tokenized upstream (:mod:`preprocessing`), so the vectorizer just
    splits on whitespace and does not re-tokenize or re-lowercase.
    """
    base_kwargs: dict[str, Any] = {
        "tokenizer": str.split,
        "preprocessor": lambda doc: doc,
        "lowercase": False,
        "token_pattern": None,
    }
    base_kwargs.update(tfidf_kwargs)
    vectorizer = TfidfVectorizer(**base_kwargs)
    documents = [" ".join(tokens) for tokens in tokenized]
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def _as_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def build_similarity_matrix(
    method: str,
    *,
    tokenized: list[list[str]] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
    tfidf_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a dense N×N similarity matrix for the requested ``method``.

    Returns ``(matrix, metadata)``. ``metadata`` always includes the
    canonical ``method`` name plus method-specific provenance (vocabulary
    size, embedding dimensionality, ...).
    """
    canonical = normalize_similarity_method(method)

    if canonical == "tfidf_cosine":
        if not tokenized:
            raise ValueError("tfidf_cosine requires 'tokenized' documents")
        matrix, vectorizer = _build_tfidf_matrix(tokenized, **(tfidf_kwargs or {}))
        sim = _sk_cosine_similarity(matrix)
        return sim, {
            "method": canonical,
            "vocabulary_size": len(vectorizer.get_feature_names_out()),
        }

    if canonical == "jaccard":
        if tokenized is None:
            raise ValueError("jaccard requires 'tokenized' documents")
        sets = [set(tokens) for tokens in tokenized]
        n = len(sets)
        sim = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                score = jaccard_similarity(sets[i], sets[j])
                sim[i, j] = sim[j, i] = score
        return sim, {"method": canonical}

    if canonical == "embedding_cosine":
        if not embeddings:
            raise ValueError(_EMBEDDING_ERROR)
        arr = np.asarray(embeddings, dtype=float)
        if arr.ndim != 2:
            raise ValueError("embeddings must be a 2D array of shape (n_items, dim)")
        sim = _sk_cosine_similarity(arr)
        return sim, {"method": canonical, "embedding_dim": int(arr.shape[1])}

    raise ValueError(f"Unsupported similarity method {method!r}")  # pragma: no cover


def pairwise_similarity(
    ids: Sequence[str],
    *,
    method: str = "tfidf_cosine",
    tokenized: list[list[str]] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    include_self: bool = False,
    tfidf_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Document-to-document / unit-to-unit similarity within one item set.

    ``ids`` may be document ids, text-unit ids, or any caller-chosen key —
    this module has no notion of "document" vs "unit"; that distinction is
    entirely up to what the caller passes in.
    """
    n = len(ids)
    if n < 2:
        raise ValueError("pairwise similarity requires at least 2 items")

    matrix, meta = build_similarity_matrix(
        method, tokenized=tokenized, embeddings=embeddings, tfidf_kwargs=tfidf_kwargs
    )
    if matrix.shape[0] != n or matrix.shape[1] != n:
        raise ValueError("similarity matrix size does not match the number of ids")

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        j_start = i if include_self else i + 1
        for j in range(j_start, n):
            score = float(matrix[i, j])
            if min_score is not None and score < min_score:
                continue
            pairs.append({"source_id": ids[i], "target_id": ids[j], "score": score})
    pairs.sort(key=lambda row: -row["score"])
    total_pairs = len(pairs)
    limited = pairs if top_k is None else pairs[: max(0, int(top_k))]

    return {
        "mode": "pairwise",
        "item_count": n,
        "pairs_tested": total_pairs,
        "pairs_returned": len(limited),
        "pairs": limited,
        **meta,
    }


def query_similarity(
    query_id: str,
    ids: Sequence[str],
    *,
    method: str = "tfidf_cosine",
    query_tokens: list[str] | None = None,
    tokenized: list[list[str]] | None = None,
    query_embedding: Sequence[float] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    tfidf_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Query-to-document similarity: rank ``ids`` by similarity to a query."""
    canonical = normalize_similarity_method(method)
    n = len(ids)
    if n < 1:
        raise ValueError("query similarity requires at least one candidate item")

    if canonical == "tfidf_cosine":
        if tokenized is None or query_tokens is None:
            raise ValueError(
                "tfidf_cosine query similarity requires 'tokenized' documents and 'query_tokens'"
            )
        combined = [*tokenized, query_tokens]
        matrix, vectorizer = _build_tfidf_matrix(combined, **(tfidf_kwargs or {}))
        sims = _sk_cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        meta: dict[str, Any] = {
            "method": canonical,
            "vocabulary_size": len(vectorizer.get_feature_names_out()),
        }
    elif canonical == "jaccard":
        if tokenized is None or query_tokens is None:
            raise ValueError(
                "jaccard query similarity requires 'tokenized' documents and 'query_tokens'"
            )
        query_set = set(query_tokens)
        sims = np.array([jaccard_similarity(query_set, set(tokens)) for tokens in tokenized])
        meta = {"method": canonical}
    elif canonical == "embedding_cosine":
        if not embeddings or query_embedding is None:
            raise ValueError(_EMBEDDING_ERROR)
        arr = np.asarray(embeddings, dtype=float)
        query_arr = np.asarray(query_embedding, dtype=float).reshape(1, -1)
        sims = _sk_cosine_similarity(query_arr, arr).ravel()
        meta = {"method": canonical, "embedding_dim": int(arr.shape[1])}
    else:
        raise ValueError(f"Unsupported similarity method {method!r}")  # pragma: no cover

    if len(sims) != n:
        raise ValueError("similarity scores size does not match the number of ids")

    rows: list[dict[str, Any]] = []
    for target_id, score in zip(ids, sims, strict=True):
        s = float(score)
        if min_score is not None and s < min_score:
            continue
        rows.append({"source_id": query_id, "target_id": target_id, "score": s})
    rows.sort(key=lambda row: -row["score"])
    limited = rows if top_k is None else rows[: max(0, int(top_k))]

    return {
        "mode": "query",
        "query_id": query_id,
        "item_count": n,
        "pairs_tested": n,
        "pairs_returned": len(limited),
        "pairs": limited,
        **meta,
    }


def group_centroid_similarity(
    ids: Sequence[str],
    group_keys: Sequence[str],
    *,
    method: str = "tfidf_cosine",
    tokenized: list[list[str]] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
    target: str = "between_groups",
    top_k: int | None = None,
    min_score: float | None = None,
    tfidf_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Group-centroid similarity.

    ``group_keys`` are entirely caller-supplied (any metadata field the
    caller chooses) — nothing here hardcodes a grouping variable.

    ``target``:

    * ``between_groups`` (default) — pairwise similarity between group centroids
    * ``item_to_own_group`` — similarity of each item to its *own* group's
      centroid (useful for spotting items that drift from their group)
    """
    if len(ids) != len(group_keys):
        raise ValueError("ids and group_keys must align 1:1")
    canonical = normalize_similarity_method(method)
    target_mode = (target or "between_groups").strip().lower()
    if target_mode not in CENTROID_TARGETS:
        raise ValueError(f"target must be one of {sorted(CENTROID_TARGETS)}")

    groups = sorted({str(key) for key in group_keys})
    if target_mode == "between_groups" and len(groups) < 2:
        raise ValueError("group_centroid similarity (between_groups) requires at least 2 groups")
    if not groups:
        raise ValueError("group_centroid similarity requires at least 1 group")

    if canonical == "jaccard":
        if tokenized is None:
            raise ValueError("jaccard group_centroid similarity requires 'tokenized' documents")
        return _group_centroid_jaccard(
            ids, group_keys, tokenized, groups, target_mode, top_k=top_k, min_score=min_score
        )

    if canonical == "tfidf_cosine":
        if not tokenized:
            raise ValueError(
                "tfidf_cosine group_centroid similarity requires 'tokenized' documents"
            )
        matrix, vectorizer = _build_tfidf_matrix(tokenized, **(tfidf_kwargs or {}))
        meta = {"vocabulary_size": len(vectorizer.get_feature_names_out())}
        vectors = matrix
    elif canonical == "embedding_cosine":
        if not embeddings:
            raise ValueError(_EMBEDDING_ERROR)
        vectors = np.asarray(embeddings, dtype=float)
        meta = {"embedding_dim": int(vectors.shape[1])}
    else:
        raise ValueError(f"Unsupported similarity method {method!r}")  # pragma: no cover

    return _group_centroid_numeric(
        ids,
        group_keys,
        vectors,
        groups,
        target_mode,
        canonical,
        meta,
        top_k=top_k,
        min_score=min_score,
    )


def _group_centroid_jaccard(
    ids: Sequence[str],
    group_keys: Sequence[str],
    tokenized: list[list[str]],
    groups: list[str],
    target_mode: str,
    *,
    top_k: int | None,
    min_score: float | None,
) -> dict[str, Any]:
    # There is no numeric "mean" for token sets — the group "centroid" is the
    # pooled vocabulary (profile) shared by every member of the group.
    profiles: dict[str, set[str]] = {g: set() for g in groups}
    for tokens, key in zip(tokenized, group_keys, strict=True):
        profiles[str(key)].update(tokens)

    if target_mode == "between_groups":
        pairs = []
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                score = jaccard_similarity(profiles[group_a], profiles[group_b])
                if min_score is not None and score < min_score:
                    continue
                pairs.append({"source_id": group_a, "target_id": group_b, "score": score})
        pairs.sort(key=lambda row: -row["score"])
        limited = pairs if top_k is None else pairs[: max(0, int(top_k))]
        return {
            "mode": "group_centroid",
            "target": target_mode,
            "method": "jaccard",
            "groups": groups,
            "pairs_returned": len(limited),
            "pairs": limited,
        }

    rows = []
    for item_id, tokens, key in zip(ids, tokenized, group_keys, strict=True):
        score = jaccard_similarity(set(tokens), profiles[str(key)])
        if min_score is not None and score < min_score:
            continue
        rows.append({"item_id": item_id, "group": str(key), "score": score})
    rows.sort(key=lambda row: row["score"])  # ascending: low cohesion first
    return {
        "mode": "group_centroid",
        "target": target_mode,
        "method": "jaccard",
        "groups": groups,
        "items_returned": len(rows),
        "items": rows,
    }


def _group_centroid_numeric(
    ids: Sequence[str],
    group_keys: Sequence[str],
    vectors: Any,
    groups: list[str],
    target_mode: str,
    canonical: str,
    meta: dict[str, Any],
    *,
    top_k: int | None,
    min_score: float | None,
) -> dict[str, Any]:
    centroids: dict[str, np.ndarray] = {}
    for group in groups:
        idx = [i for i, key in enumerate(group_keys) if str(key) == group]
        rows_g = vectors[idx]
        mean_vec = np.asarray(rows_g.mean(axis=0)).ravel()
        centroids[group] = mean_vec

    if target_mode == "between_groups":
        pairs = []
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                score = float(
                    _sk_cosine_similarity(
                        centroids[group_a].reshape(1, -1), centroids[group_b].reshape(1, -1)
                    )[0, 0]
                )
                if min_score is not None and score < min_score:
                    continue
                pairs.append({"source_id": group_a, "target_id": group_b, "score": score})
        pairs.sort(key=lambda row: -row["score"])
        limited = pairs if top_k is None else pairs[: max(0, int(top_k))]
        return {
            "mode": "group_centroid",
            "target": target_mode,
            "method": canonical,
            "groups": groups,
            "pairs_returned": len(limited),
            "pairs": limited,
            **meta,
        }

    centroid_matrix = np.stack([centroids[g] for g in groups])
    item_matrix = _as_dense(vectors)
    sims = _sk_cosine_similarity(item_matrix, centroid_matrix)
    group_index = {g: idx for idx, g in enumerate(groups)}

    rows = []
    for row_idx, (item_id, key) in enumerate(zip(ids, group_keys, strict=True)):
        score = float(sims[row_idx, group_index[str(key)]])
        if min_score is not None and score < min_score:
            continue
        rows.append({"item_id": item_id, "group": str(key), "score": score})
    rows.sort(key=lambda row: row["score"])  # ascending: low cohesion first
    return {
        "mode": "group_centroid",
        "target": target_mode,
        "method": canonical,
        "groups": groups,
        "items_returned": len(rows),
        "items": rows,
        **meta,
    }


def describe_similarity_capabilities() -> dict[str, Any]:
    return {
        "methods": sorted(SIMILARITY_METHODS),
        "modes": sorted(SIMILARITY_MODES),
        "centroid_targets": sorted(CENTROID_TARGETS),
        "embedding_policy": (
            "embedding_cosine requires caller-supplied vectors; embeddings are never "
            "computed here and never forced into the default (lexical) similarity path"
        ),
        "grouping": "caller-supplied ids/group_keys only (no hardcoded categories)",
        "notes": [
            "'pairwise' covers both document-to-document and unit-to-unit similarity — "
            "the granularity is whatever ids/tokenized the caller passes in.",
            "'jaccard' group centroids are pooled token-set profiles, not numeric means.",
        ],
    }
