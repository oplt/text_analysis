"""Tests for the headless StageRunner pipeline (CLI/tests adapter; not the production API path)."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest

from backend.modules.text_research.application.analysis_executor import (
    build_spec_from_request,
    run_prepared_analysis,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.stage_runner import StageRunner


class StageRunnerFrequenciesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        os.environ["RESEARCH_ARTIFACT_DIR"] = self.temp_dir.name
        stage_cache.invalidate()

    def tearDown(self) -> None:
        os.environ.pop("RESEARCH_ARTIFACT_DIR", None)

    def _spec(self) -> AnalysisSpecification:
        return build_spec_from_request(
            "frequencies",
            "corpus-test",
            analysis_parameters={"top_n": 10, "rate_per": 1000},
            output={"include_manifest": True},
        )

    def test_prepare_and_frequencies_yields_results_and_checksums(self) -> None:
        texts = ["alpha beta gamma delta", "beta gamma epsilon zeta"]
        spec = self._spec()
        plan = compile_plan(spec.normalize())

        context = {
            "spec": spec,
            "texts": texts,
            "unit_ids": ["u1", "u2"],
            "config": PreprocessingConfig(),
        }
        result = StageRunner(plan, context).run()

        self.assertIn("prepared", result)
        self.assertIn("results", result)
        self.assertIn("frequencies", result["results"])
        self.assertIn("checksums", result)
        self.assertTrue(result["checksums"]["corpus_checksum"])
        self.assertTrue(result["checksums"]["pipeline_checksum"])
        self.assertEqual(result["checksums"]["analysis_spec_hash"], spec.spec_hash())
        self.assertIn("prepare_corpus", result["stage_timings"])
        self.assertIn("frequencies", result["stage_timings"])
        self.assertIn("manifest", result)

    def test_same_spec_texts_yield_same_pipeline_checksum(self) -> None:
        texts = ["hello world", "foo bar baz"]
        unit_ids = ["a", "b"]
        config = PreprocessingConfig(lowercase=True)
        spec = self._spec()

        first = run_prepared_analysis(
            spec,
            texts,
            unit_ids=unit_ids,
            config=config,
            use_stage_cache=False,
        )
        second = run_prepared_analysis(
            spec,
            texts,
            unit_ids=unit_ids,
            config=config,
            use_stage_cache=False,
        )

        self.assertEqual(
            first["checksums"]["pipeline_checksum"],
            second["checksums"]["pipeline_checksum"],
        )
        self.assertEqual(
            first["checksums"]["corpus_checksum"],
            second["checksums"]["corpus_checksum"],
        )

    def test_does_not_mutate_input_texts(self) -> None:
        texts = ["The Policy, IS Universal!", "Education matters."]
        originals = copy.deepcopy(texts)
        spec = self._spec()

        run_prepared_analysis(
            spec,
            texts,
            unit_ids=["u1", "u2"],
            config=PreprocessingConfig(),
            use_stage_cache=False,
        )

        self.assertEqual(texts, originals)
        for original, current in zip(originals, texts, strict=True):
            self.assertIs(original, current)


if __name__ == "__main__":
    unittest.main()
