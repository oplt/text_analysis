"""Optional cross-runtime contracts, enabled only in an R-capable CI worker."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from backend.modules.text_research.application.analysis_executor import run_prepared_analysis
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig

pytestmark = pytest.mark.skipif(
    not RAnalysisEngine.runtime_ready(),
    reason="requires an enabled R / quanteda runtime",
)

_FLOAT_TOL = 1e-9


def _spec(
    analysis_type: str,
    *,
    runtime: str,
    parameters: dict[str, object],
) -> AnalysisSpecification:
    return AnalysisSpecification.model_validate(
        {
            "corpus": {"corpus_id": "parity"},
            "analysis": {"type": analysis_type, "parameters": parameters},
            "engine": {"runtime": runtime, "preprocessing_mode": "standardized"},
        }
    )


def _run(
    analysis_type: str,
    texts: list[str],
    *,
    runtime: str,
    parameters: dict[str, object],
    unit_ids: list[str] | None = None,
    document_ids: list[str | None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_prepared_analysis(
        _spec(analysis_type, runtime=runtime, parameters=parameters),
        texts,
        unit_ids=unit_ids or [f"u{i + 1}" for i in range(len(texts))],
        document_ids=document_ids,
        config=PreprocessingConfig(),
        use_stage_cache=False,
        **kwargs,
    )


def _almost_equal(left: float, right: float, *, tol: float = _FLOAT_TOL) -> bool:
    return abs(float(left) - float(right)) <= tol


def test_frequency_counts_match_for_standardized_tokens() -> None:
    texts = ["Alpha beta", "beta gamma"]
    python = _run("frequencies", texts, runtime="python", parameters={"top_n": 10})
    r = _run("frequencies", texts, runtime="r", parameters={"top_n": 10})
    python_counts = {row["term"]: row["count"] for row in python["results"]["frequencies"]}
    r_counts = {row["term"]: row["count"] for row in r["results"]["frequencies"]}
    assert r_counts == python_counts


def test_dfm_dimensions_and_cells_match_for_standardized_tokens() -> None:
    texts = ["alpha beta", "beta gamma", "don't well-being"]
    python = _run("dfm", texts, runtime="python", parameters={"weighting": "count"})
    r = _run("dfm", texts, runtime="r", parameters={"weighting": "count"})
    python_dfm = python["results"]["dfm"]
    assert r["results"]["dimensions"]["documents"] == python_dfm["dimensions"]["units"]
    assert r["results"]["dimensions"]["features"] == python_dfm["dimensions"]["features"]
    assert set(r["results"]["feature_names"]) == set(python_dfm["feature_names"])


def test_kwic_occurrence_multiset_matches_for_standardized_tokens() -> None:
    texts = ["alpha beta beta", "beta gamma"]
    parameters = {"keyword": "beta", "window_size": 1, "query_mode": "word"}
    python = _run("kwic", texts, runtime="python", parameters=parameters)
    r = _run("kwic", texts, runtime="r", parameters=parameters)
    python_keys = Counter(
        (row["text_unit_id"], row["token_start"], row["token_end"], row["keyword"])
        for row in python["results"]["matches"]
    )
    r_keys = Counter(
        (row["text_unit_id"], row["token_start"], row["token_end"], row["keyword"])
        for row in r["results"]["matches"]
    )
    assert r_keys == python_keys


def test_dictionary_spans_categories_and_document_prevalence_match() -> None:
    texts = [
        "alpha beta gamma",
        "alpha only",
        "noise only",
        "alpha again",
    ]
    document_ids = ["d1", "d1", "d2", "d3"]
    parameters = {
        "hierarchy": {
            "theme": {
                "core": ["alpha", "beta gamma"],
            }
        },
        "rate_per": 1000,
    }
    python = _run(
        "dictionary",
        texts,
        runtime="python",
        parameters=parameters,
        unit_ids=["u1", "u2", "u3", "u4"],
        document_ids=document_ids,
    )
    r = _run(
        "dictionary",
        texts,
        runtime="r",
        parameters=parameters,
        unit_ids=["u1", "u2", "u3", "u4"],
        document_ids=document_ids,
    )
    python_results = python["results"]
    r_results = r["results"]
    assert r_results["total_hits"] == python_results["total_hits"]
    assert _almost_equal(r_results["unit_prevalence"], python_results["unit_prevalence"])
    assert _almost_equal(
        r_results["document_prevalence"], python_results["document_prevalence"]
    )
    # Two of three documents contain hits (d1 and d3); d2 does not.
    assert python_results["document_prevalence"] == pytest.approx(2 / 3)
    assert python_results["document_prevalence"] != python_results["unit_prevalence"]

    python_spans = Counter(
        (
            row["text_unit_id"],
            row["matched_expression"],
            row["token_start"],
            row["token_end"],
            row.get("category"),
        )
        for row in python_results["matches"]
    )
    r_spans = Counter(
        (
            row["text_unit_id"],
            row["matched_expression"],
            row["token_start"],
            row["token_end"],
            row.get("category"),
        )
        for row in r_results["matches"]
    )
    assert r_spans == python_spans


@pytest.mark.parametrize("method", ["log_likelihood", "chi_square", "fisher"])
def test_keyness_methods_and_adjusted_p_values_match(method: str) -> None:
    texts_a = ["freedom education policy", "freedom rights education"]
    texts_b = ["market growth policy", "market trade growth"]
    parameters = {
        "method": method,
        "correction": "bh",
        "min_frequency": 1,
        "top_n": 20,
        "group_a_label": "A",
        "group_b_label": "B",
    }
    python = _run(
        "keyness",
        texts_a,
        runtime="python",
        parameters=parameters,
        unit_ids=["a1", "a2"],
        texts_b=texts_b,
        unit_ids_b=["b1", "b2"],
    )
    r = _run(
        "keyness",
        texts_a,
        runtime="r",
        parameters=parameters,
        unit_ids=["a1", "a2"],
        texts_b=texts_b,
        unit_ids_b=["b1", "b2"],
    )
    assert r["results"]["method"] == python["results"]["method"] == method
    python_rows = {row["feature"]: row for row in python["results"]["features"]}
    r_rows = {row["feature"]: row for row in r["results"]["features"]}
    assert set(r_rows) == set(python_rows)
    for feature, left in python_rows.items():
        right = r_rows[feature]
        assert left["freq_a"] == right["freq_a"]
        assert left["freq_b"] == right["freq_b"]
        assert _almost_equal(left["keyness_statistic"], right["keyness_statistic"], tol=1e-6)
        assert _almost_equal(left["p_value"], right["p_value"], tol=1e-6)
        left_adj = left.get("p_adjusted", left.get("p_value_adjusted"))
        right_adj = right.get("p_value_adjusted", right.get("p_adjusted"))
        assert left_adj is not None and right_adj is not None
        assert _almost_equal(left_adj, right_adj, tol=1e-6)


@pytest.mark.parametrize(
    "association_method",
    ["count", "pmi", "npmi", "dice", "log_dice", "t_score"],
)
def test_cooccurrence_association_metrics_match(association_method: str) -> None:
    texts = [
        "universal education policy rights",
        "universal education market growth",
        "education policy market rights",
    ]
    parameters = {
        "window_size": 2,
        "top_n": 50,
        "association_method": association_method,
        "directional": False,
        "min_frequency": 1,
        "min_count": 1,
        "include_network": False,
    }
    python = _run("cooccurrence", texts, runtime="python", parameters=parameters)
    r = _run("cooccurrence", texts, runtime="r", parameters=parameters)
    assert r["results"]["association_method"] == association_method
    python_pairs = {
        (row["term_a"], row["term_b"]): row for row in python["results"]["pairs"]
    }
    r_pairs = {(row["term_a"], row["term_b"]): row for row in r["results"]["pairs"]}
    assert set(r_pairs) == set(python_pairs)
    for key, left in python_pairs.items():
        right = r_pairs[key]
        assert left["count"] == right["count"]
        assert left["freq_a"] == right["freq_a"]
        assert left["freq_b"] == right["freq_b"]
        for field in ("pmi", "npmi", "dice", "log_dice", "t_score", "association_score"):
            assert _almost_equal(left[field], right[field], tol=1e-9), field
