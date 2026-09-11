"""Multi-input scientific identity for keyness must include corpus B everywhere."""

from __future__ import annotations

from backend.modules.text_research.application.analysis_executor import run_prepared_analysis
from backend.modules.text_research.domain.analysis_result import (
    AnalysisInputIdentity,
    ScientificInputIdentity,
    build_analysis_identity,
    build_scientific_inputs,
    scientific_inputs_digest,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.execution_defaults import computation_identity
from backend.modules.text_research.infrastructure.engines.python_engine import PythonAnalysisEngine
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.provenance import build_run_provenance
from backend.modules.text_research.infrastructure.r_runtime.serializer import serialize_r_job
from backend.modules.text_research.infrastructure.stage_cache import compute_identity_lookup


def _keyness_spec(runtime: str = "python") -> AnalysisSpecification:
    return AnalysisSpecification.model_validate(
        {
            "corpus": {"corpus_id": "corpus-a"},
            "analysis": {
                "type": "keyness",
                "parameters": {"method": "log_likelihood", "top_n": 10},
            },
            "engine": {"runtime": runtime},
            "output": {"include_manifest": True},
        }
    )


def _reference_role(inputs: list) -> AnalysisInputIdentity:
    for item in inputs:
        if isinstance(item, AnalysisInputIdentity):
            row = item
        else:
            row = AnalysisInputIdentity.model_validate(item)
        if row.role == "reference":
            return row
    raise AssertionError("reference input missing")


def test_scientific_input_alias_and_builders() -> None:
    assert ScientificInputIdentity is AnalysisInputIdentity
    inputs = build_scientific_inputs(
        target_corpus_checksum="a",
        target_pipeline_checksum="pa",
        reference_corpus_checksum="b",
        reference_pipeline_checksum="pb",
    )
    assert [item.role for item in inputs] == ["target", "reference"]
    assert scientific_inputs_digest(inputs)
    assert scientific_inputs_digest(inputs[:1]) == ""


def test_keyness_reference_change_invalidates_computation_and_cache_identity() -> None:
    texts_a = ["alpha beta gamma", "alpha delta"]
    texts_b1 = ["omega theta", "omega"]
    texts_b2 = ["omega changed", "omega"]
    cfg = PreprocessingConfig()
    prepared_a = prepare_texts(texts_a, cfg, unit_ids=["a1", "a2"], force_in_memory=True)
    prepared_b1 = prepare_texts(texts_b1, cfg, unit_ids=["b1", "b2"], force_in_memory=True)
    prepared_b2 = prepare_texts(texts_b2, cfg, unit_ids=["b1", "b2"], force_in_memory=True)
    assert prepared_a.corpus_checksum != prepared_b1.corpus_checksum
    assert prepared_b1.corpus_checksum != prepared_b2.corpus_checksum

    spec = _keyness_spec()
    plan = compile_plan(spec)
    inputs_1 = build_scientific_inputs(
        target_corpus_checksum=prepared_a.corpus_checksum,
        target_pipeline_checksum=prepared_a.pipeline_checksum,
        reference_corpus_checksum=prepared_b1.corpus_checksum,
        reference_pipeline_checksum=prepared_b1.pipeline_checksum,
    )
    inputs_2 = build_scientific_inputs(
        target_corpus_checksum=prepared_a.corpus_checksum,
        target_pipeline_checksum=prepared_a.pipeline_checksum,
        reference_corpus_checksum=prepared_b2.corpus_checksum,
        reference_pipeline_checksum=prepared_b2.pipeline_checksum,
    )

    computation_1 = computation_identity(
        plan.spec_hash,
        prepared_a.corpus_checksum,
        engine_version=plan.engine_version,
        engine_name=plan.engine_name,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=inputs_1,
    )
    computation_2 = computation_identity(
        plan.spec_hash,
        prepared_a.corpus_checksum,
        engine_version=plan.engine_version,
        engine_name=plan.engine_name,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=inputs_2,
    )
    assert computation_1 != computation_2

    cache_1 = compute_identity_lookup(
        plan.spec_hash,
        prepared_a.corpus_checksum,
        plan.engine_version,
        engine_name=plan.engine_name,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=inputs_1,
    )
    cache_2 = compute_identity_lookup(
        plan.spec_hash,
        prepared_a.corpus_checksum,
        plan.engine_version,
        engine_name=plan.engine_name,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=inputs_2,
    )
    assert cache_1 != cache_2

    provenance_1 = build_run_provenance(
        spec=spec,
        corpus_checksum=prepared_a.corpus_checksum,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=[item.model_dump(mode="json") for item in inputs_1],
    )
    provenance_2 = build_run_provenance(
        spec=spec,
        corpus_checksum=prepared_a.corpus_checksum,
        pipeline_checksum=prepared_a.pipeline_checksum,
        scientific_inputs=[item.model_dump(mode="json") for item in inputs_2],
    )
    assert provenance_1["scientific_inputs"] != provenance_2["scientific_inputs"]
    assert prepared_b1.corpus_checksum in provenance_1["parent_artifact_checksums"]
    assert prepared_b2.corpus_checksum in provenance_2["parent_artifact_checksums"]


def test_python_and_r_expose_reference_checksum_when_b_changes() -> None:
    texts_a = ["alpha beta", "alpha gamma"]
    texts_b1 = ["omega one", "omega two"]
    texts_b2 = ["omega changed", "omega two"]
    cfg = PreprocessingConfig()
    prepared_b1 = prepare_texts(texts_b1, cfg, unit_ids=["b1", "b2"], force_in_memory=True)
    prepared_b2 = prepare_texts(texts_b2, cfg, unit_ids=["b1", "b2"], force_in_memory=True)

    python_1 = run_prepared_analysis(
        _keyness_spec("python"),
        texts_a,
        unit_ids=["a1", "a2"],
        config=cfg,
        texts_b=texts_b1,
        unit_ids_b=["b1", "b2"],
        force_in_memory=True,
    )
    python_2 = run_prepared_analysis(
        _keyness_spec("python"),
        texts_a,
        unit_ids=["a1", "a2"],
        config=cfg,
        texts_b=texts_b2,
        unit_ids_b=["b1", "b2"],
        force_in_memory=True,
    )
    py_ref_1 = _reference_role(python_1["analysis_result"].identity.inputs or [])
    py_ref_2 = _reference_role(python_2["analysis_result"].identity.inputs or [])
    assert py_ref_1.corpus_checksum == prepared_b1.corpus_checksum
    assert py_ref_2.corpus_checksum == prepared_b2.corpus_checksum
    assert py_ref_1.corpus_checksum != py_ref_2.corpus_checksum
    assert python_1["computation_identity"] != python_2["computation_identity"]
    assert (
        python_1["manifest"]["provenance"]["scientific_inputs"]
        != python_2["manifest"]["provenance"]["scientific_inputs"]
    )
    assert python_1["analysis_result"].runtime.implementation == PythonAnalysisEngine.implementation

    prepared_a = python_1["prepared"]
    r_bundle_1 = serialize_r_job(
        specification=_keyness_spec("r"),
        prepared=prepared_a,
        comparison_prepared=prepared_b1,
        run_id="keyness-b1",
        engine_version="r-quanteda-2",
    )
    r_bundle_2 = serialize_r_job(
        specification=_keyness_spec("r"),
        prepared=prepared_a,
        comparison_prepared=prepared_b2,
        run_id="keyness-b2",
        engine_version="r-quanteda-2",
    )
    import json

    identity_1 = json.loads((r_bundle_1.manifest_path).read_text(encoding="utf-8"))["identity"]
    identity_2 = json.loads((r_bundle_2.manifest_path).read_text(encoding="utf-8"))["identity"]
    r_ref_1 = _reference_role(identity_1["inputs"])
    r_ref_2 = _reference_role(identity_2["inputs"])
    assert r_ref_1.corpus_checksum == prepared_b1.corpus_checksum
    assert r_ref_2.corpus_checksum == prepared_b2.corpus_checksum
    assert (
        build_analysis_identity(
            spec_hash=identity_1["spec_hash"],
            engine_name=identity_1["engine_name"],
            engine_version=identity_1["engine_version"],
            inputs=identity_1["inputs"],
        ).corpus_checksum
        == prepared_a.corpus_checksum
    )
