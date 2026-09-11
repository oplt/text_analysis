from __future__ import annotations

from backend.modules.text_research.application.analysis_executor import run_prepared_analysis
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.engines.python_engine import PythonAnalysisEngine
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


def _spec(runtime: str = "python") -> AnalysisSpecification:
    return AnalysisSpecification.model_validate(
        {
            "corpus": {"corpus_id": "test"},
            "analysis": {"type": "frequencies", "parameters": {"top_n": 10}},
            "engine": {"runtime": runtime},
        }
    )


def test_python_engine_is_default_and_returns_canonical_result() -> None:
    result = run_prepared_analysis(
        _spec(), ["alpha beta", "beta"], unit_ids=["a", "b"], config=PreprocessingConfig()
    )
    assert result["analysis_result"].runtime.engine == "python"
    assert result["analysis_result"].runtime.implementation == PythonAnalysisEngine.implementation
    assert (
        result["analysis_result"].identity.engine_version
        == PythonAnalysisEngine.implementation_version
    )
    assert result["analysis_result"].results == result["results"]
    assert PythonAnalysisEngine().supports("frequencies")


def test_runtime_changes_spec_and_execution_identity() -> None:
    python = _spec("python")
    r = _spec("r")
    assert python.spec_hash() != r.spec_hash()
    assert compile_plan(python).engine_name == "python"
    assert compile_plan(r).engine_name == "r"
    assert compile_plan(r).engine_version == RAnalysisEngine.implementation_version
    assert compile_plan(python).engine_version == PythonAnalysisEngine.implementation_version
    assert RAnalysisEngine().supports("dfm")
    assert RAnalysisEngine().supports("dictionary")
    assert RAnalysisEngine().supports("keyness")
    assert RAnalysisEngine().supports("cooccurrence")


def test_compile_plan_ignores_client_implementation_as_version() -> None:
    """Clients may send the canonical family; version is always server-owned."""
    client_spoofed = AnalysisSpecification.model_validate(
        {
            "corpus": {"corpus_id": "test"},
            "analysis": {"type": "frequencies", "parameters": {"top_n": 10}},
            "engine": {"runtime": "r", "implementation": "quanteda"},
        }
    )
    plan = compile_plan(client_spoofed)
    assert plan.engine_version == RAnalysisEngine.implementation_version
    assert plan.engine_version != "quanteda"
    assert client_spoofed.engine.implementation == RAnalysisEngine.implementation


def test_absent_implementation_matches_canonical_identity() -> None:
    absent = _spec("r")
    with_canonical = AnalysisSpecification.model_validate(
        {
            "corpus": {"corpus_id": "test"},
            "analysis": {"type": "frequencies", "parameters": {"top_n": 10}},
            "engine": {"runtime": "r", "implementation": "quanteda"},
        }
    )
    assert absent.spec_hash() == with_canonical.spec_hash()


def test_unknown_implementation_rejected() -> None:
    from pydantic import ValidationError

    try:
        AnalysisSpecification.model_validate(
            {
                "corpus": {"corpus_id": "test"},
                "analysis": {"type": "frequencies", "parameters": {"top_n": 10}},
                "engine": {"runtime": "python", "implementation": "spoofed"},
            }
        )
    except ValidationError as exc:
        assert "incompatible" in str(exc).lower()
    else:
        raise AssertionError("expected ValidationError for arbitrary implementation")


def test_flat_request_accepts_an_engine_specification() -> None:
    specification = AnalysisSpecification.from_flat(
        corpus_id="test",
        analysis_type="kwic",
        engine={"runtime": "r"},
    )
    assert specification.engine.runtime == "r"
    assert specification.engine.implementation == "quanteda"
