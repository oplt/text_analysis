"""Tests for per-unit language detection and prepare_texts wiring."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.language_detection import detect_units
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class DetectUnitsTests(unittest.TestCase):
    def test_detect_units_returns_one_result_per_text(self):
        texts = [
            "The government policy was discussed in parliament.",
            "Die Regierung und der Bundesrat haben eine neue Politik beschlossen.",
        ]
        results = detect_units(texts, unit_ids=["u1", "u2"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["unit_id"], "u1")
        self.assertEqual(results[1]["unit_id"], "u2")
        self.assertIn(results[0]["language"], {"en", "und"})
        self.assertIn(results[1]["language"], {"de", "und"})

    def test_manual_overrides_by_unit_id(self):
        texts = ["any text", "other text"]
        results = detect_units(texts, manual_overrides={"u2": "fr"}, unit_ids=["u1", "u2"])
        self.assertNotEqual(results[0]["language"], "fr")
        self.assertEqual(results[1]["language"], "fr")
        self.assertTrue(results[1]["overridden"])


class PrepareTextsPerUnitLanguageTests(unittest.TestCase):
    def test_per_unit_language_stored_in_provenance(self):
        texts = [
            "The government policy was discussed in parliament.",
            "Die Regierung und der Bundesrat haben eine neue Politik beschlossen.",
        ]
        artifact = prepare_texts(
            texts,
            PreprocessingConfig(remove_stopwords=True),
            unit_ids=["en-doc", "de-doc"],
            per_unit_language=True,
        )
        detection = artifact.provenance.get("language_detection", {})
        self.assertEqual(detection.get("mode"), "per_unit")
        self.assertEqual(len(detection.get("units", [])), 2)

    def test_per_unit_override_wins(self):
        texts = ["The government policy was discussed in parliament."]
        artifact = prepare_texts(
            texts,
            PreprocessingConfig(),
            unit_ids=["u1"],
            per_unit_language=True,
            language_overrides_by_unit={"u1": "de"},
        )
        unit = artifact.provenance["language_detection"]["units"][0]
        self.assertEqual(unit["language"], "de")
        self.assertTrue(unit["overridden"])


if __name__ == "__main__":
    unittest.main()
