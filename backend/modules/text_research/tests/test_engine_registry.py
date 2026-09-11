"""Authoritative execution-engine registry resolution (§19)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.execution_defaults import (
    ENGINE_VERSION,
    R_ENGINE_VERSION,
)
from backend.modules.text_research.infrastructure import provenance
from backend.modules.text_research.infrastructure.engines.python_engine import (
    PythonAnalysisEngine,
)
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.plugin_registry import (
    list_execution_engines,
    resolve_execution_engine,
)


class ExecutionEngineRegistryTests(unittest.TestCase):
    def test_resolve_python_and_r_match_engine_classes(self) -> None:
        python = resolve_execution_engine("python")
        r_engine = resolve_execution_engine("r")

        self.assertEqual(python.runtime, "python")
        self.assertEqual(python.implementation, PythonAnalysisEngine.implementation)
        self.assertEqual(python.implementation_version, ENGINE_VERSION)
        self.assertEqual(python.implementation_version, PythonAnalysisEngine.implementation_version)
        self.assertTrue(python.supports("classification"))
        self.assertIs(python.factory, PythonAnalysisEngine)

        self.assertEqual(r_engine.runtime, "r")
        self.assertEqual(r_engine.implementation, RAnalysisEngine.implementation)
        self.assertEqual(r_engine.implementation_version, R_ENGINE_VERSION)
        self.assertEqual(r_engine.implementation_version, RAnalysisEngine.implementation_version)
        self.assertEqual(r_engine.supported_analyses, RAnalysisEngine.supported_analyses)
        self.assertFalse(r_engine.supports("classification"))
        self.assertIs(r_engine.factory, RAnalysisEngine)

    def test_list_order_python_then_r(self) -> None:
        names = [d.runtime for d in list_execution_engines()]
        self.assertEqual(names[:2], ["python", "r"])

    def test_compiler_and_provenance_use_same_versions(self) -> None:
        python_plan = compile_plan(
            AnalysisSpecification.from_flat(corpus_id="c1", analysis_type="frequencies")
        )
        r_plan = compile_plan(
            AnalysisSpecification.from_flat(
                corpus_id="c1",
                analysis_type="frequencies",
                engine={"runtime": "r"},
            )
        )
        self.assertEqual(
            python_plan.engine_version,
            resolve_execution_engine("python").implementation_version,
        )
        self.assertEqual(
            r_plan.engine_version,
            resolve_execution_engine("r").implementation_version,
        )
        runtime = provenance.runtime_environment()
        self.assertEqual(
            runtime["engine_version"],
            resolve_execution_engine("python").implementation_version,
        )
        self.assertEqual(runtime["engine_runtime"], "python")
        self.assertEqual(
            runtime["engine_implementation"],
            resolve_execution_engine("python").implementation,
        )

    def test_unknown_runtime_raises(self) -> None:
        with self.assertRaises(KeyError):
            resolve_execution_engine("julia")


if __name__ == "__main__":
    unittest.main()
