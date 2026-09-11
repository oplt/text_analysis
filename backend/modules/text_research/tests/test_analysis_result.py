"""Regression tests for validated execution-engine result models."""

from backend.modules.text_research.domain.analysis_result import (
    AnalysisIdentity,
    AnalysisInputIdentity,
    AnalysisResult,
    RuntimeInfo,
    build_analysis_identity,
)


def test_analysis_result_models_instantiate_without_unresolved_annotations() -> None:
    input_identity = AnalysisInputIdentity(
        role="target",
        corpus_checksum="corpus-checksum",
        pipeline_checksum="pipeline-checksum",
    )
    identity = AnalysisIdentity(
        spec_hash="spec-hash",
        corpus_checksum="corpus-checksum",
        pipeline_checksum="pipeline-checksum",
        engine_name="python",
        engine_version="1.0",
        inputs=[input_identity],
    )
    result = AnalysisResult(
        analysis_type="frequencies",
        runtime=RuntimeInfo(engine="python", implementation="python"),
        identity=identity,
        results={},
    )

    assert result.identity.inputs == [input_identity]
    # Schema generation fails when Literal / nested models are unresolved.
    schema = AnalysisIdentity.model_json_schema()
    assert "inputs" in schema["properties"]
    dumped = result.model_dump(mode="json")
    assert dumped["identity"]["engine_name"] == "python"
    assert dumped["identity"]["inputs"][0]["role"] == "target"


def test_build_analysis_identity_round_trips_under_pydantic() -> None:
    identity = build_analysis_identity(
        spec_hash="spec",
        engine_name="r",
        engine_version="r-quanteda-2",
        inputs=[
            {
                "role": "target",
                "corpus_checksum": "c1",
                "pipeline_checksum": "p1",
            },
            {
                "role": "reference",
                "corpus_checksum": "c2",
                "pipeline_checksum": "p2",
            },
        ],
    )
    restored = AnalysisIdentity.model_validate(identity.model_dump(mode="json"))
    assert restored.engine_name == "r"
    assert [item.role for item in restored.inputs or []] == ["target", "reference"]
