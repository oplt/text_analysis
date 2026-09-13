"""LATEST-016: API invoke_operator and StageRunner share deterministic results."""

from __future__ import annotations

import os
import tempfile
import unittest

from backend.modules.text_research.application.analysis_executor import build_spec_from_request
from backend.modules.text_research.application.analysis_operators import (
    OPERATORS,
    invoke_operator,
    list_registered_operators,
)
from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.stage_runner import StageRunner


class CanonicalOperatorDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        os.environ["RESEARCH_ARTIFACT_DIR"] = self.temp_dir.name
        stage_cache.invalidate()

    def tearDown(self) -> None:
        os.environ.pop("RESEARCH_ARTIFACT_DIR", None)

    def test_operators_alias_matches_registry(self) -> None:
        self.assertEqual(set(OPERATORS), set(list_registered_operators()))

    def _prepared(self):
        texts = ["alpha beta gamma delta", "beta gamma epsilon zeta", "alpha epsilon"]
        return prepare_texts(
            texts,
            PreprocessingConfig(),
            unit_ids=["u1", "u2", "u3"],
        )

    def _stage_runner_result(self, analysis_type: str, params: dict, prepared) -> dict:
        spec = build_spec_from_request(
            analysis_type,
            "corpus-test",
            analysis_parameters=params,
            output={"include_manifest": True},
        )
        plan = compile_plan(spec.normalize())
        context = {
            "spec": spec,
            "prepared": prepared,
            "texts": list(prepared.original_units),
            "unit_ids": list(prepared.unit_ids),
            "config": PreprocessingConfig(),
        }
        # Skip prepare_corpus by ensuring stages that need prepared already have it;
        # run only through stages present in the plan.
        return StageRunner(plan, context).run()["results"]

    def test_frequencies_api_operator_matches_stage_runner(self) -> None:
        prepared = self._prepared()
        params = {"top_n": 10, "rate_per": 1000}
        via_api = invoke_operator("frequencies", prepared, params)
        via_runner = self._stage_runner_result("frequencies", params, prepared)
        self.assertEqual(via_api["frequencies"], via_runner["frequencies"])
        self.assertEqual(via_api["metadata"], via_runner["metadata"])

    def test_ngrams_api_operator_matches_stage_runner(self) -> None:
        prepared = self._prepared()
        params = {"n": 2, "top_n": 5, "rate_per": 1000, "skip": 0}
        via_api = invoke_operator("ngrams", prepared, params)
        via_runner = self._stage_runner_result("ngrams", params, prepared)
        self.assertEqual(via_api["ngrams"], via_runner["ngrams"])

    def test_cooccurrence_api_operator_matches_stage_runner(self) -> None:
        prepared = self._prepared()
        params = {
            "window_size": 3,
            "top_n": 5,
            "association_method": "pmi",
            "directional": False,
            "min_frequency": 1,
            "min_count": 1,
            "include_network": False,
        }
        via_api = invoke_operator("cooccurrence", prepared, params)
        via_runner = self._stage_runner_result("cooccurrence", params, prepared)
        self.assertEqual(via_api["pairs"], via_runner["pairs"])

    def test_readability_api_operator_matches_stage_runner(self) -> None:
        prepared = self._prepared()
        via_api = invoke_operator("readability", prepared, {})
        via_runner = self._stage_runner_result("readability", {}, prepared)
        self.assertEqual(via_api["n_units"], via_runner["n_units"])
        self.assertEqual(via_api["corpus"], via_runner["corpus"])


if __name__ == "__main__":
    unittest.main()
