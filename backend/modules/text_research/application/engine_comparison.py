"""Deterministic summaries for two independently persisted engine runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

COMPARISON_PARENT_RUN_ID_KEY = "comparison_parent_run_id"
COMPARISON_ROLE_KEY = "comparison_role"
COMPARISON_AWAITING_STAGE = "awaiting_children"

TERMINAL_COMPARISON_CHILD_STATUSES = frozenset({"completed", "failed", "cancelled"})

COMPARABLE_ANALYSIS_TYPES = frozenset(
    {"frequencies", "dfm", "kwic", "dictionary", "keyness", "cooccurrence"}
)

# Documented numerical tolerances (aligned with parity suite).
KEYNESS_FLOAT_TOL = 1e-6
COOCCURRENCE_FLOAT_TOL = 1e-9
PREVALENCE_FLOAT_TOL = 1e-9

_MAX_DIFFS = 50


def _canonical(results: dict[str, Any]) -> dict[str, Any]:
    nested = results.get("analysis_result")
    if isinstance(nested, dict) and isinstance(nested.get("results"), dict):
        return nested["results"]
    return results


def _rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _floats_equal(left: Any, right: Any, *, tol: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return left == right
    if tol <= 0:
        return left_f == right_f
    return abs(left_f - right_f) <= tol


def _kwic_occurrence(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("text_unit_id"),
        row.get("token_start"),
        row.get("token_end"),
        row.get("keyword"),
    )


def _dictionary_span(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("text_unit_id"),
        row.get("matched_expression"),
        row.get("token_start"),
        row.get("token_end"),
        row.get("category"),
        row.get("subcategory"),
    )


def _dfm_payload(results: dict[str, Any]) -> dict[str, Any]:
    nested = results.get("dfm") if isinstance(results.get("dfm"), dict) else None
    return nested if nested is not None else results


def _p_adjusted(row: dict[str, Any]) -> Any:
    return row.get("p_adjusted", row.get("p_value_adjusted"))


def _bounded_diffs(items: dict[str, Any], *, limit: int = _MAX_DIFFS) -> dict[str, Any]:
    return dict(list(items.items())[:limit])


def _compare_frequencies(python: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    left = {str(row.get("term")): row.get("count") for row in _rows(python.get("frequencies"))}
    right = {str(row.get("term")): row.get("count") for row in _rows(r.get("frequencies"))}
    left_terms = set(left)
    right_terms = set(right)
    shared = left_terms & right_terms
    term_sets_equal = left_terms == right_terms
    count_equal = term_sets_equal and all(left[term] == right[term] for term in left_terms)
    return {
        "analysis_type": "frequencies",
        "comparison_status": "equal" if count_equal else "unequal",
        "term_overlap": len(shared),
        "term_union": len(left_terms | right_terms),
        "term_sets_equal": term_sets_equal,
        "count_equal": count_equal,
        "count_differences": _bounded_diffs(
            {
                term: {"python": left.get(term), "r": right.get(term)}
                for term in sorted(left_terms | right_terms)
                if left.get(term) != right.get(term)
            }
        ),
    }


def _compare_kwic(python: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    left = Counter(_kwic_occurrence(row) for row in _rows(python.get("matches")))
    right = Counter(_kwic_occurrence(row) for row in _rows(r.get("matches")))
    equal = left == right
    return {
        "analysis_type": "kwic",
        "comparison_status": "equal" if equal else "unequal",
        "match_overlap": sum((left & right).values()),
        "matches_equal": equal,
        "python_match_count": sum(left.values()),
        "r_match_count": sum(right.values()),
    }


def _compare_dictionary(python: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    left_spans = Counter(_dictionary_span(row) for row in _rows(python.get("matches")))
    right_spans = Counter(_dictionary_span(row) for row in _rows(r.get("matches")))
    spans_equal = left_spans == right_spans
    total_hits_equal = python.get("total_hits") == r.get("total_hits")
    unit_prev_equal = _floats_equal(
        python.get("unit_prevalence"), r.get("unit_prevalence"), tol=PREVALENCE_FLOAT_TOL
    )
    doc_prev_equal = _floats_equal(
        python.get("document_prevalence"),
        r.get("document_prevalence"),
        tol=PREVALENCE_FLOAT_TOL,
    )
    left_cats = {
        str(row.get("category") or row.get("name") or key): int(row.get("hits") or 0)
        for key, row in _category_map(python).items()
    }
    right_cats = {
        str(row.get("category") or row.get("name") or key): int(row.get("hits") or 0)
        for key, row in _category_map(r).items()
    }
    categories_equal = left_cats == right_cats
    equal = (
        spans_equal and total_hits_equal and unit_prev_equal and doc_prev_equal and categories_equal
    )
    return {
        "analysis_type": "dictionary",
        "comparison_status": "equal" if equal else "unequal",
        "spans_equal": spans_equal,
        "matches_equal": spans_equal,
        "total_hits_equal": total_hits_equal,
        "python_total_hits": python.get("total_hits"),
        "r_total_hits": r.get("total_hits"),
        "unit_prevalence_equal": unit_prev_equal,
        "document_prevalence_equal": doc_prev_equal,
        "categories_equal": categories_equal,
        "category_differences": _bounded_diffs(
            {
                name: {"python": left_cats.get(name), "r": right_cats.get(name)}
                for name in sorted(set(left_cats) | set(right_cats))
                if left_cats.get(name) != right_cats.get(name)
            }
        ),
        "python_match_count": sum(left_spans.values()),
        "r_match_count": sum(right_spans.values()),
    }


def _category_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("by_category")
    if isinstance(raw, dict):
        return {
            str(key): value if isinstance(value, dict) else {"category": key, "hits": value}
            for key, value in raw.items()
        }
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for row in _rows(raw):
            key = str(row.get("category") or row.get("name") or len(out))
            out[key] = row
        return out
    return {}


def _compare_keyness(
    python: dict[str, Any],
    r: dict[str, Any],
    *,
    float_tol: float,
) -> dict[str, Any]:
    left_rows = {
        str(row.get("feature")): row
        for row in _rows(python.get("features") or python.get("keyness"))
        if row.get("feature") is not None
    }
    right_rows = {
        str(row.get("feature")): row
        for row in _rows(r.get("features") or r.get("keyness"))
        if row.get("feature") is not None
    }
    feature_sets_equal = set(left_rows) == set(right_rows)
    differences: dict[str, Any] = {}
    for feature in sorted(set(left_rows) | set(right_rows)):
        left = left_rows.get(feature)
        right = right_rows.get(feature)
        if left is None or right is None:
            differences[feature] = {"python": left, "r": right}
            continue
        field_diffs: dict[str, Any] = {}
        if left.get("freq_a") != right.get("freq_a") or left.get("freq_b") != right.get("freq_b"):
            field_diffs["counts"] = {
                "python": {"freq_a": left.get("freq_a"), "freq_b": left.get("freq_b")},
                "r": {"freq_a": right.get("freq_a"), "freq_b": right.get("freq_b")},
            }
        for field in ("keyness_statistic", "p_value"):
            if not _floats_equal(left.get(field), right.get(field), tol=float_tol):
                field_diffs[field] = {"python": left.get(field), "r": right.get(field)}
        left_adj = _p_adjusted(left)
        right_adj = _p_adjusted(right)
        if not _floats_equal(left_adj, right_adj, tol=float_tol):
            field_diffs["p_adjusted"] = {"python": left_adj, "r": right_adj}
        left_dir = left.get("effect_direction") or left.get("direction")
        right_dir = right.get("effect_direction") or right.get("direction")
        if left_dir != right_dir:
            field_diffs["direction"] = {"python": left_dir, "r": right_dir}
        left_effect = left.get("log_ratio", left.get("effect_size"))
        right_effect = right.get("log_ratio", right.get("effect_size"))
        if not _floats_equal(left_effect, right_effect, tol=float_tol):
            field_diffs["log_ratio"] = {"python": left_effect, "r": right_effect}
        if field_diffs:
            differences[feature] = field_diffs
    method_equal = python.get("method") == r.get("method")
    equal = feature_sets_equal and not differences and method_equal
    return {
        "analysis_type": "keyness",
        "comparison_status": "equal" if equal else "unequal",
        "float_tol": float_tol,
        "method_equal": method_equal,
        "python_method": python.get("method"),
        "r_method": r.get("method"),
        "feature_sets_equal": feature_sets_equal,
        "feature_overlap": len(set(left_rows) & set(right_rows)),
        "features_equal": equal,
        "feature_difference_count": len(differences),
        "feature_differences": _bounded_diffs(differences),
    }


def _compare_cooccurrence(
    python: dict[str, Any],
    r: dict[str, Any],
    *,
    float_tol: float,
) -> dict[str, Any]:
    left_method = python.get("association_method")
    right_method = r.get("association_method")
    if left_method != right_method:
        return {
            "analysis_type": "cooccurrence",
            "comparison_status": "inconclusive",
            "association_method_equal": False,
            "python_association_method": left_method,
            "r_association_method": right_method,
            "note": "Refusing to compare association scores across different statistical methods",
            "pairs_equal": None,
        }
    left_dir = bool(python.get("directional"))
    right_dir = bool(r.get("directional"))
    left_pairs = {
        (str(row.get("term_a")), str(row.get("term_b"))): row
        for row in _rows(python.get("pairs") or python.get("cooccurrence"))
    }
    right_pairs = {
        (str(row.get("term_a")), str(row.get("term_b"))): row
        for row in _rows(r.get("pairs") or r.get("cooccurrence"))
    }
    pair_sets_equal = set(left_pairs) == set(right_pairs)
    differences: dict[str, Any] = {}
    metric_fields = ("association_score", "pmi", "npmi", "dice", "log_dice", "t_score")
    for key in sorted(set(left_pairs) | set(right_pairs)):
        left = left_pairs.get(key)
        right = right_pairs.get(key)
        label = f"{key[0]}|{key[1]}"
        if left is None or right is None:
            differences[label] = {"python": left, "r": right}
            continue
        field_diffs: dict[str, Any] = {}
        for field in ("count", "freq_a", "freq_b"):
            if left.get(field) != right.get(field):
                field_diffs[field] = {"python": left.get(field), "r": right.get(field)}
        for field in metric_fields:
            if field not in left and field not in right:
                continue
            if not _floats_equal(left.get(field), right.get(field), tol=float_tol):
                field_diffs[field] = {"python": left.get(field), "r": right.get(field)}
        if field_diffs:
            differences[label] = field_diffs
    directionality_equal = left_dir == right_dir
    equal = pair_sets_equal and not differences and directionality_equal
    return {
        "analysis_type": "cooccurrence",
        "comparison_status": "equal" if equal else "unequal",
        "float_tol": float_tol,
        "association_method_equal": True,
        "association_method": left_method,
        "directionality_equal": directionality_equal,
        "python_directional": left_dir,
        "r_directional": right_dir,
        "pair_sets_equal": pair_sets_equal,
        "pair_overlap": len(set(left_pairs) & set(right_pairs)),
        "pairs_equal": equal,
        "pair_difference_count": len(differences),
        "pair_differences": _bounded_diffs(differences),
    }


def compare_engine_results(
    analysis_type: str,
    python_results: dict[str, Any],
    r_results: dict[str, Any],
    *,
    float_tol: float | None = None,
) -> dict[str, Any]:
    """Return an inspectable, language-neutral comparison; never executes either engine."""
    from backend.modules.text_research.infrastructure.dfm_matrix_identity import (
        compare_dfm_identities,
    )

    if analysis_type not in COMPARABLE_ANALYSIS_TYPES:
        raise ValueError(f"Engine comparison is not supported for {analysis_type!r}")

    python = _canonical(python_results)
    r = _canonical(r_results)
    if analysis_type == "frequencies":
        return _compare_frequencies(python, r)
    if analysis_type == "dfm":
        return compare_dfm_identities(
            _dfm_payload(python),
            _dfm_payload(r),
            float_tol=0.0 if float_tol is None else float_tol,
        )
    if analysis_type == "kwic":
        return _compare_kwic(python, r)
    if analysis_type == "dictionary":
        return _compare_dictionary(python, r)
    if analysis_type == "keyness":
        return _compare_keyness(
            python,
            r,
            float_tol=KEYNESS_FLOAT_TOL if float_tol is None else float_tol,
        )
    if analysis_type == "cooccurrence":
        return _compare_cooccurrence(
            python,
            r,
            float_tol=COOCCURRENCE_FLOAT_TOL if float_tol is None else float_tol,
        )
    raise ValueError(f"Engine comparison is not supported for {analysis_type!r}")


def extract_runtime(results: dict[str, Any]) -> Any:
    """Prefer top-level runtime, else nested analysis_result.runtime."""
    runtime = results.get("runtime")
    if runtime is not None:
        return runtime
    nested = results.get("analysis_result")
    if isinstance(nested, dict):
        return nested.get("runtime")
    return None


def child_failure_detail(*, python_run: Any, r_run: Any) -> str:
    """Human-readable diagnostic when a comparison child did not complete."""
    parts: list[str] = []
    for label, run in (("python", python_run), ("r", r_run)):
        if run is None:
            parts.append(f"{label}:missing")
            continue
        status = getattr(run, "status", None)
        if status != "completed":
            err = getattr(run, "error_message", None) or "failed"
            parts.append(f"{getattr(run, 'id', '?')}:{status}:{err}")
    return "; ".join(parts) if parts else "comparison child runs did not complete"
