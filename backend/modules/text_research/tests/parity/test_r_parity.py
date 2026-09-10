"""Optional cross-runtime contracts, enabled only in an R-capable CI worker."""

from __future__ import annotations

import pytest
from backend.modules.text_research.application.analysis_executor import run_prepared_analysis
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig

pytestmark = pytest.mark.skipif(
    not RAnalysisEngine.available(),
    reason="requires an enabled R / quanteda runtime",
)


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


def test_frequency_counts_match_for_standardized_tokens() -> None:
    texts = ["Alpha beta", "beta gamma"]
    python = run_prepared_analysis(
        _spec("frequencies", runtime="python", parameters={"top_n": 10}),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    r = run_prepared_analysis(
        _spec("frequencies", runtime="r", parameters={"top_n": 10}),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    python_counts = {row["term"]: row["count"] for row in python["results"]["frequencies"]}
    r_counts = {row["term"]: row["count"] for row in r["results"]["frequencies"]}
    assert r_counts == python_counts


def test_dfm_dimensions_match_for_standardized_tokens() -> None:
    texts = ["alpha beta", "beta gamma"]
    python = run_prepared_analysis(
        _spec("dfm", runtime="python", parameters={"weighting": "count"}),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    r = run_prepared_analysis(
        _spec("dfm", runtime="r", parameters={"weighting": "count"}),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    assert r["results"]["dimensions"]["documents"] == python["results"]["dfm"][
        "dimensions"
    ]["units"]
    assert r["results"]["dimensions"]["features"] == python["results"]["dfm"][
        "dimensions"
    ]["features"]


def test_kwic_match_terms_and_units_match_for_standardized_tokens() -> None:
    texts = ["alpha beta", "beta gamma"]
    parameters = {"keyword": "beta", "window_size": 1, "query_mode": "word"}
    python = run_prepared_analysis(
        _spec("kwic", runtime="python", parameters=parameters),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    r = run_prepared_analysis(
        _spec("kwic", runtime="r", parameters=parameters),
        texts,
        unit_ids=["u1", "u2"],
        config=PreprocessingConfig(),
        use_stage_cache=False,
    )
    assert [(row["text_unit_id"], row["keyword"]) for row in r["results"]["matches"]] == [
        (row["text_unit_id"], row["keyword"])
        for row in python["results"]["matches"]
    ]
