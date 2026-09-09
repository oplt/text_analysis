"""Feature weighting schemes — separate from tokenization / preprocessing.

Build a raw count matrix first, then apply a :class:`WeightingScheme`. This keeps
reproducible weighting provenance independent of tokenizer choices.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from scipy import sparse

WEIGHTING_NAMES: frozenset[str] = frozenset(
    {
        "count",
        "binary",
        "tf",
        "term_frequency",
        "tfidf",
        "tf-idf",
        "sublinear_tf",
        "sublinear",
        "log_count",
        "logcount",
        "bm25",
    }
)

_CANONICAL = {
    "term_frequency": "tf",
    "tf-idf": "tfidf",
    "sublinear": "sublinear_tf",
    "sublinear_tfidf": "sublinear_tf",
    "logcount": "log_count",
}


@dataclass(frozen=True, slots=True)
class WeightingScheme:
    """Serializable feature-weighting configuration."""

    name: str
    # BM25 parameters (Robertson / Sparkes defaults commonly used in IR).
    k1: float = 1.5
    b: float = 0.75
    # TF-IDF: use smoothed IDF log((N+1)/(df+1))+1 style when True.
    smooth_idf: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_name"] = self.name
        return payload


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolve_weighting_scheme(
    name: str | WeightingScheme | None,
    *,
    k1: float | None = None,
    b: float | None = None,
    smooth_idf: bool | None = None,
) -> WeightingScheme:
    """Resolve a weighting name / partial config into a :class:`WeightingScheme`."""
    if isinstance(name, WeightingScheme):
        scheme = name
        return WeightingScheme(
            name=scheme.name,
            k1=scheme.k1 if k1 is None else float(k1),
            b=scheme.b if b is None else float(b),
            smooth_idf=scheme.smooth_idf if smooth_idf is None else bool(smooth_idf),
        )

    key = "count" if name is None else str(name).strip().lower().replace(" ", "_")
    canonical = _CANONICAL.get(key, key)
    if canonical not in {
        "count",
        "binary",
        "tf",
        "tfidf",
        "sublinear_tf",
        "log_count",
        "bm25",
    }:
        raise ValueError(
            f"Unsupported weighting {name!r}; expected one of "
            f"count, binary, tf, tfidf, sublinear_tf, log_count, bm25"
        )
    return WeightingScheme(
        name=canonical,
        k1=1.5 if k1 is None else float(k1),
        b=0.75 if b is None else float(b),
        smooth_idf=True if smooth_idf is None else bool(smooth_idf),
    )


def describe_weighting(scheme: WeightingScheme | str | None) -> dict[str, Any]:
    """Recordable provenance for a weighting scheme (never claims unused params)."""
    resolved = resolve_weighting_scheme(scheme)
    payload = {
        "weighting": resolved.name,
        "weighting_implementation": "text_research.weighting",
        "weighting_separate_from_tokenization": True,
        "parameters": {},
        "package_versions": {
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "scikit-learn": _pkg_version("scikit-learn"),
        },
    }
    if resolved.name in {"tfidf", "sublinear_tf"}:
        payload["parameters"]["smooth_idf"] = resolved.smooth_idf
    if resolved.name == "sublinear_tf":
        payload["parameters"]["sublinear_tf"] = True
    if resolved.name == "bm25":
        payload["parameters"]["k1"] = resolved.k1
        payload["parameters"]["b"] = resolved.b
    return payload


def list_weighting_schemes() -> list[dict[str, Any]]:
    """Catalog of supported weighting schemes for API / docs."""
    return [
        {
            "name": "count",
            "description": "Raw term counts (identity weighting).",
            "optional": False,
        },
        {
            "name": "binary",
            "description": "Presence/absence (1 if count > 0).",
            "optional": False,
        },
        {
            "name": "tf",
            "description": "Within-document L1-normalized term frequency.",
            "optional": False,
        },
        {
            "name": "tfidf",
            "description": "Term frequency–inverse document frequency.",
            "optional": False,
        },
        {
            "name": "sublinear_tf",
            "description": "TF-IDF with sublinear TF (1 + log tf).",
            "optional": False,
        },
        {
            "name": "log_count",
            "description": "log(1 + count) per cell.",
            "optional": True,
        },
        {
            "name": "bm25",
            "description": "Okapi BM25 with configurable k1 and b.",
            "optional": True,
            "parameters": {"k1": 1.5, "b": 0.75},
        },
    ]


def _document_lengths(count_matrix: sparse.spmatrix) -> np.ndarray:
    return np.asarray(count_matrix.sum(axis=1)).ravel().astype(float)


def _document_frequencies(count_matrix: sparse.spmatrix) -> np.ndarray:
    # Number of documents with a nonzero count for each feature.
    binary = count_matrix.copy()
    binary.data = np.ones_like(binary.data)
    return np.asarray(binary.sum(axis=0)).ravel().astype(float)


def _idf_vector(n_docs: int, dfs: np.ndarray, *, smooth: bool) -> np.ndarray:
    if smooth:
        # sklearn-compatible smooth IDF: log((N+1)/(df+1)) + 1
        return np.log((n_docs + 1.0) / (dfs + 1.0)) + 1.0
    # Classic: log(N / df) with df floor at 1
    return np.log(n_docs / np.maximum(dfs, 1.0))


def apply_weighting(
    count_matrix: sparse.spmatrix,
    scheme: WeightingScheme | str | None,
) -> sparse.csr_matrix:
    """Apply ``scheme`` to a **count** matrix. Does not tokenize or alter vocabulary.

    ``count_matrix`` must be non-negative document×feature counts (CSR/CSC/COO OK).
    """
    resolved = resolve_weighting_scheme(scheme)
    matrix = sparse.csr_matrix(count_matrix, dtype=float)
    n_docs, n_features = matrix.shape

    if resolved.name == "count":
        return matrix

    if resolved.name == "binary":
        out = matrix.copy()
        out.data = np.ones_like(out.data)
        return out

    if resolved.name == "log_count":
        out = matrix.copy()
        out.data = np.log1p(out.data)
        return out

    if resolved.name == "tf":
        # Row-wise L1 normalize.
        from sklearn.preprocessing import normalize as sk_normalize

        return sk_normalize(matrix, norm="l1", axis=1)

    dfs = _document_frequencies(matrix)
    idf = _idf_vector(n_docs, dfs, smooth=resolved.smooth_idf)

    if resolved.name in {"tfidf", "sublinear_tf"}:
        out = matrix.copy().tocsr()
        if resolved.name == "sublinear_tf":
            out.data = 1.0 + np.log(out.data)
        # Multiply columns by IDF.
        out = out.multiply(idf)
        return sparse.csr_matrix(out)

    if resolved.name == "bm25":
        doc_len = _document_lengths(matrix)
        avgdl = float(doc_len.mean()) if n_docs else 0.0
        # BM25 IDF variant: log(1 + (N - df + 0.5)/(df + 0.5))
        bm25_idf = np.log(1.0 + (n_docs - dfs + 0.5) / (dfs + 0.5))
        k1 = resolved.k1
        b = resolved.b

        coo = matrix.tocoo()
        data = np.empty_like(coo.data, dtype=float)
        for i, (row, col, tf) in enumerate(zip(coo.row, coo.col, coo.data, strict=True)):
            dl = doc_len[row]
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 0.0))
            data[i] = bm25_idf[col] * ((tf * (k1 + 1.0)) / denom) if denom else 0.0
        return sparse.csr_matrix((data, (coo.row, coo.col)), shape=matrix.shape)

    raise ValueError(f"Unsupported weighting {resolved.name!r}")  # pragma: no cover


# Back-compat alias used by DFM layer.
def normalize_weighting_name(name: str) -> str:
    return resolve_weighting_scheme(name).name
