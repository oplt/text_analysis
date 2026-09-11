"""AnalysisRun ID must survive into the R manifest (never workdir inference)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.modules.text_research.domain.analysis_result import (
    AnalysisIdentity,
    AnalysisResult,
    RuntimeInfo,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.execution_defaults import R_ENGINE_VERSION
from backend.modules.text_research.infrastructure.engines.r_engine import (
    IN_MEMORY_ANALYSIS_RUN_ID,
    RAnalysisEngine,
    resolve_analysis_run_id,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.r_runtime.serializer import serialize_r_job
from backend.modules.text_research.infrastructure.stage_runner import StageRunner


class ResolveAnalysisRunIdTests(unittest.TestCase):
    def test_prefers_outer_run_id(self) -> None:
        self.assertEqual(
            resolve_analysis_run_id(run_id="outer", pipeline_context={"run_id": "nested"}),
            "outer",
        )

    def test_reads_nested_pipeline_context(self) -> None:
        self.assertEqual(
            resolve_analysis_run_id(pipeline_context={"run_id": "persisted-42"}),
            "persisted-42",
        )

    def test_headless_defaults_to_in_memory(self) -> None:
        self.assertEqual(resolve_analysis_run_id(), IN_MEMORY_ANALYSIS_RUN_ID)
        self.assertEqual(resolve_analysis_run_id(run_id=""), IN_MEMORY_ANALYSIS_RUN_ID)
        self.assertEqual(resolve_analysis_run_id(run_id=None), IN_MEMORY_ANALYSIS_RUN_ID)


class RManifestRunIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_serialize_manifest_run_id_matches_analysis_run(self) -> None:
        prepared = prepare_texts(
            ["alpha beta"],
            PreprocessingConfig(),
            unit_ids=["u1"],
            force_in_memory=True,
        )
        spec = AnalysisSpecification.model_validate(
            {
                "corpus": {"corpus_id": "c1"},
                "analysis": {"type": "frequencies", "parameters": {"top_n": 5}},
                "engine": {"runtime": "r", "preprocessing_mode": "standardized"},
            }
        ).normalize()
        with patch(
            "backend.modules.text_research.infrastructure.r_runtime.serializer.settings"
        ) as mocked:
            mocked.RESEARCH_R_WORK_DIR = self._tmpdir.name
            bundle = serialize_r_job(
                specification=spec,
                prepared=prepared,
                run_id="analysis-run-xyz",
                engine_version=R_ENGINE_VERSION,
            )
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "analysis-run-xyz")
        self.assertEqual(bundle.analysis_run_id, "analysis-run-xyz")
        self.assertNotEqual(bundle.analysis_run_id, bundle.workdir.name)
        self.assertTrue(str(bundle.workdir.name).startswith("job-"))

    def test_engine_uses_pipeline_context_run_id_when_outer_missing(self) -> None:
        prepared = prepare_texts(
            ["alpha"],
            PreprocessingConfig(),
            unit_ids=["u1"],
            force_in_memory=True,
        )
        spec = AnalysisSpecification.model_validate(
            {
                "corpus": {"corpus_id": "c1"},
                "analysis": {"type": "dfm", "parameters": {"weighting": "count"}},
                "engine": {"runtime": "r", "preprocessing_mode": "standardized"},
            }
        ).normalize()
        captured: dict[str, str] = {}

        def _fake_serialize(**kwargs):  # type: ignore[no-untyped-def]
            captured["run_id"] = kwargs["run_id"]
            return MagicMock(
                workdir=Path(self._tmpdir.name) / "job-temp",
                manifest_path=Path(self._tmpdir.name) / "manifest.json",
                result_path=Path(self._tmpdir.name) / "result.json",
                analysis_run_id=kwargs["run_id"],
            )

        fake_result = AnalysisResult(
            analysis_type="dfm",
            runtime=RuntimeInfo(engine="r", implementation="quanteda"),
            identity=AnalysisIdentity(
                spec_hash="s",
                corpus_checksum="c",
                pipeline_checksum="p",
                engine_name="r",
                engine_version=R_ENGINE_VERSION,
            ),
            results={},
        )
        with (
            patch(
                "backend.modules.text_research.infrastructure.engines.r_engine.serialize_r_job",
                side_effect=_fake_serialize,
            ),
            patch(
                "backend.modules.text_research.infrastructure.engines.r_engine.run_r_job",
                return_value=fake_result,
            ),
        ):
            RAnalysisEngine().execute(
                spec,
                prepared,
                pipeline_context={"run_id": "child-r-run-99", "prepared": prepared},
            )
        self.assertEqual(captured["run_id"], "child-r-run-99")

    def test_stage_runner_propagates_context_run_id_to_engine(self) -> None:
        spec = AnalysisSpecification.model_validate(
            {
                "corpus": {"corpus_id": "c1"},
                "analysis": {"type": "frequencies", "parameters": {"top_n": 5}},
                "engine": {"runtime": "r", "preprocessing_mode": "standardized"},
            }
        ).normalize()
        plan = compile_plan(spec)
        # Keep only validate + engine stage for a light test.
        from backend.modules.text_research.infrastructure.pipeline_compiler import ExecutionPlan

        light = ExecutionPlan(
            stages=["validate_spec", "frequencies"],
            spec_hash=plan.spec_hash,
            engine_name=plan.engine_name,
            engine_version=plan.engine_version,
        )
        prepared = prepare_texts(
            ["hello world"],
            PreprocessingConfig(),
            unit_ids=["u1"],
            force_in_memory=True,
        )
        seen: dict[str, object] = {}

        class _FakeEngine:
            def supports(self, analysis_type: str) -> bool:
                return True

            def execute(self, specification, prepared_arg, **kwargs):  # type: ignore[no-untyped-def]
                seen["run_id"] = kwargs.get("run_id")
                seen["pipeline_run_id"] = (kwargs.get("pipeline_context") or {}).get("run_id")
                return AnalysisResult(
                    analysis_type="frequencies",
                    runtime=RuntimeInfo(engine="r", implementation="quanteda"),
                    identity=AnalysisIdentity(
                        spec_hash="s",
                        corpus_checksum="c",
                        pipeline_checksum="p",
                        engine_name="r",
                        engine_version=R_ENGINE_VERSION,
                    ),
                    results={"frequencies": []},
                )

        context = {
            "spec": spec,
            "prepared": prepared,
            "run_id": "persisted-run-777",
            "texts": ["hello world"],
            "unit_ids": ["u1"],
            "config": PreprocessingConfig(),
        }
        with patch(
            "backend.modules.text_research.infrastructure.plugin_registry.get_plugin",
            return_value=_FakeEngine,
        ):
            StageRunner(light, context).run()
        self.assertEqual(seen["run_id"], "persisted-run-777")
        self.assertEqual(seen["pipeline_run_id"], "persisted-run-777")


if __name__ == "__main__":
    unittest.main()
