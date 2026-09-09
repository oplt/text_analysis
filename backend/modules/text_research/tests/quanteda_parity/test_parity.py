"""Quanteda-analogue parity regression tests (§52).

Uses stored Python reference fixtures. R is not required in CI.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.modules.text_research.infrastructure.preprocessing import tokenize
from backend.modules.text_research.infrastructure.quantitative import build_dfm

FIXTURES = Path(__file__).parent / "fixtures"


class QuantedaParityTests(unittest.TestCase):
    def test_tokenization_and_dfm_match_fixture(self) -> None:
        payload = json.loads((FIXTURES / "tiny_corpus.json").read_text())
        cfg = payload["config"]
        tokens = [tokenize(text, cfg) for text in payload["corpus"]]
        self.assertEqual(tokens, payload["tokens"])

        dfm = build_dfm(payload["corpus"], cfg, weighting="count")
        expected_n = payload["dfm"]["n_features"]
        actual_n = dfm["dimensions"].get("features") or len(dfm["feature_names"])
        self.assertEqual(actual_n, expected_n)
        self.assertEqual(
            sorted(dfm["feature_names"]),
            payload["dfm"]["vocabulary_sorted"],
        )

    def test_known_differences_documented(self) -> None:
        text = (FIXTURES / "known_differences.md").read_text()
        self.assertIn("Intentional differences", text)
        self.assertIn("tolerances", text)


if __name__ == "__main__":
    unittest.main()
