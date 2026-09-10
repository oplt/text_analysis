"""Deterministic summaries for two independently persisted engine runs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _canonical(results: dict[str, Any]) -> dict[str, Any]:
    nested = results.get("analysis_result")
    if isinstance(nested, dict) and isinstance(nested.get("results"), dict):
        return nested["results"]
    return results


def _rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _kwic_occurrence(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("text_unit_id"),
        row.get("token_start"),
        row.get("token_end"),
        row.get("keyword"),
    )


def _dfm_payload(results: dict[str, Any]) -> dict[str, Any]:
    nested = results.get("dfm") if isinstance(results.get("dfm"), dict) else None
    return nested if nested is not None else results


def _coo_triples(payload: dict[str, Any]) -> list[tuple[int, int, float]]:
    sparse = payload.get("sparse_coo")
    if isinstance(sparse, dict) and sparse.get("rows") is not None:
        rows = sparse.get("rows") or []
        cols = sparse.get("cols") or []
        values = sparse.get("values") or []
    else:
        sparse = payload.get("sparse") if isinstance(payload.get("sparse"), dict) else {}
        rows = sparse.get("row") or []
        cols = sparse.get("col") or []
        values = sparse.get("data") or []
    triples: list[tuple[int, int, float]] = []
    for row, col, value in zip(rows, cols, values, strict=False):
        triples.append((int(row), int(col), float(value)))
    return triples


def _cell_map(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Map sparse cells to (unit_id, feature_name) → value for cross-engine compare."""
    features = [str(name) for name in (payload.get("feature_names") or [])]
    unit_ids = [str(uid) for uid in (payload.get("unit_ids") or [])]
    cells: dict[tuple[str, str], float] = {}
    for row, col, value in _coo_triples(payload):
        unit = unit_ids[row] if 0 <= row < len(unit_ids) else str(row)
        feature = features[col] if 0 <= col < len(features) else str(col)
        cells[(unit, feature)] = value
    return cells


def _values_equal(left: float, right: float, *, tol: float) -> bool:
    if tol <= 0:
        return left == right
    return abs(left - right) <= tol


def compare_engine_results(
    analysis_type: str,
    python_results: dict[str, Any],
    r_results: dict[str, Any],
    *,
    float_tol: float = 0.0,
) -> dict[str, Any]:
    """Return an inspectable, language-neutral comparison; never executes either engine."""
    python = _canonical(python_results)
    r = _canonical(r_results)
    if analysis_type == "frequencies":
        left = {str(row.get("term")): row.get("count") for row in _rows(python.get("frequencies"))}
        right = {str(row.get("term")): row.get("count") for row in _rows(r.get("frequencies"))}
        left_terms = set(left)
        right_terms = set(right)
        shared = left_terms & right_terms
        term_sets_equal = left_terms == right_terms
        count_equal = term_sets_equal and all(left[term] == right[term] for term in left_terms)
        return {
            "analysis_type": analysis_type,
            "term_overlap": len(shared),
            "term_union": len(left_terms | right_terms),
            "term_sets_equal": term_sets_equal,
            "count_equal": count_equal,
            "count_differences": {
                term: {"python": left.get(term), "r": right.get(term)}
                for term in sorted(left_terms | right_terms)
                if left.get(term) != right.get(term)
            },
        }
    if analysis_type == "dfm":
        left = _dfm_payload(python)
        right = _dfm_payload(r)
        left_features = set(str(name) for name in (left.get("feature_names") or []))
        right_features = set(str(name) for name in (right.get("feature_names") or []))
        left_cells = _cell_map(left)
        right_cells = _cell_map(right)
        shared_keys = set(left_cells) & set(right_cells)
        cell_differences = {
            f"{unit}|{feature}": {
                "python": left_cells.get((unit, feature)),
                "r": right_cells.get((unit, feature)),
            }
            for unit, feature in sorted(set(left_cells) | set(right_cells))
            if (unit, feature) not in shared_keys
            or not _values_equal(
                left_cells[(unit, feature)],
                right_cells[(unit, feature)],
                tol=float_tol,
            )
        }
        left_nnz = left.get("nnz")
        right_nnz = right.get("nnz") or (right.get("summary") or {}).get("nnz")
        return {
            "analysis_type": analysis_type,
            "python_dimensions": left.get("dimensions"),
            "r_dimensions": right.get("dimensions"),
            "vocabulary_equal": left_features == right_features,
            "vocabulary_overlap": len(left_features & right_features),
            "python_nnz": left_nnz,
            "r_nnz": right_nnz,
            "cells_equal": not cell_differences,
            "cell_difference_count": len(cell_differences),
            "cell_differences": dict(list(cell_differences.items())[:50]),
        }
    if analysis_type == "kwic":
        left = Counter(_kwic_occurrence(row) for row in _rows(python.get("matches")))
        right = Counter(_kwic_occurrence(row) for row in _rows(r.get("matches")))
        return {
            "analysis_type": analysis_type,
            "match_overlap": sum((left & right).values()),
            "matches_equal": left == right,
            "python_match_count": sum(left.values()),
            "r_match_count": sum(right.values()),
        }
    raise ValueError(f"Engine comparison is not supported for {analysis_type!r}")
