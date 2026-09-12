"""TASK-013: API operators and StageRunner/CLI share deterministic analysis math."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.analysis_operators import (
    list_registered_operators,
    run_frequencies_operator,
    run_readability_operator,
)
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class OperatorParityTests(unittest.TestCase):
    def test_registry_includes_core_operators(self) -> None:
        names = set(list_registered_operators())
        self.assertTrue({"frequencies", "ngrams", "dfm", "kwic", "readability"} <= names)

    def test_frequencies_operator_matches_direct_quantitative(self) -> None:
        from backend.modules.text_research.infrastructure import quantitative

        texts = ["alpha beta gamma", "beta gamma delta"]
        prepared = prepare_texts(texts, PreprocessingConfig(), unit_ids=["u1", "u2"])
        via_operator = run_frequencies_operator(prepared, top_n=5, rate_per=1000)
        via_direct = quantitative.term_frequency_report(
            [list(t) for t in prepared.token_sequences],
            top_n=5,
            rate_per=1000,
            unit_ids=list(prepared.unit_ids),
        )
        self.assertEqual(via_operator["frequencies"], via_direct["frequencies"])

    def test_readability_operator_stable(self) -> None:
        texts = ["This is a simple sentence. Another one follows."]
        prepared = prepare_texts(texts, PreprocessingConfig(), unit_ids=["u1"])
        result = run_readability_operator(prepared)
        self.assertIn("corpus", result)
        self.assertEqual(result["n_units"], 1)


if __name__ == "__main__":
    unittest.main()
