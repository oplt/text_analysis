"""TASK-018: analysis applicability policy warnings."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.analysis_policy import (
    validate_analysis_policy,
)


class AnalysisPolicyTests(unittest.TestCase):
    def test_readability_warns_for_non_english(self) -> None:
        warnings = validate_analysis_policy(analysis_type="readability", language="tr")
        self.assertTrue(warnings)
        self.assertTrue(any("English" in w for w in warnings))

    def test_readability_ok_for_english(self) -> None:
        warnings = validate_analysis_policy(analysis_type="readability", language="en")
        self.assertEqual(warnings, [])

    def test_kwic_case_sensitive_vs_lowercase_profile(self) -> None:
        warnings = validate_analysis_policy(
            analysis_type="kwic",
            case_sensitive=True,
            preprocessing={"lowercase": True},
        )
        self.assertTrue(any("case_sensitive" in w for w in warnings))

    def test_phrase_morphology_stem_and_lemma(self) -> None:
        warnings = validate_analysis_policy(
            analysis_type="ngrams",
            preprocessing={"stemming": True, "lemmatization": True},
        )
        self.assertTrue(any("stemming" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main()
