"""Deterministic summaries for two independently persisted engine runs."""

from __future__ import annotations

from typing import Any


def _canonical(results: dict[str, Any]) -> dict[str, Any]:
    nested = results.get("analysis_result")
    if isinstance(nested, dict) and isinstance(nested.get("results"), dict):
        return nested["results"]
    return results


def _rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def compare_engine_results(
    analysis_type: str, python_results: dict[str, Any], r_results: dict[str, Any]
) -> dict[str, Any]:
    """Return an inspectable, language-neutral comparison; never executes either engine."""
    python = _canonical(python_results)
    r = _canonical(r_results)
    if analysis_type == "frequencies":
        left = {str(row.get("term")): row.get("count") for row in _rows(python.get("frequencies"))}
        right = {str(row.get("term")): row.get("count") for row in _rows(r.get("frequencies"))}
        shared = set(left) & set(right)
        return {
            "analysis_type": analysis_type,
            "term_overlap": len(shared),
            "term_union": len(set(left) | set(right)),
            "count_equal": all(left[term] == right[term] for term in shared),
            "count_differences": {
                term: {"python": left.get(term), "r": right.get(term)}
                for term in sorted(set(left) | set(right)) if left.get(term) != right.get(term)
            },
        }
    if analysis_type == "dfm":
        left = python.get("dfm") if isinstance(python.get("dfm"), dict) else python
        right = r
        left_features = set(left.get("feature_names") or [])
        right_features = set(right.get("feature_names") or [])
        return {
            "analysis_type": analysis_type,
            "python_dimensions": left.get("dimensions"),
            "r_dimensions": right.get("dimensions"),
            "vocabulary_equal": left_features == right_features,
            "vocabulary_overlap": len(left_features & right_features),
            "python_nnz": left.get("nnz"),
            "r_nnz": right.get("nnz") or (right.get("summary") or {}).get("nnz"),
        }
    if analysis_type == "kwic":
        left = {
            (row.get("text_unit_id"), row.get("keyword"))
            for row in _rows(python.get("matches"))
        }
        right = {
            (row.get("text_unit_id"), row.get("keyword"))
            for row in _rows(r.get("matches"))
        }
        return {
            "analysis_type": analysis_type,
            "match_overlap": len(left & right),
            "matches_equal": left == right,
        }
    raise ValueError(f"Engine comparison is not supported for {analysis_type!r}")
