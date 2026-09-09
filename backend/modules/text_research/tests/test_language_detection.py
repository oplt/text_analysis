"""Tests for heuristic language detection and routing."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.language_detection import (
    HeuristicLanguageDetector,
    LanguageRouter,
    detect_with_override,
)


class HeuristicLanguageDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = HeuristicLanguageDetector()

    def test_detects_english(self):
        result = self.detector.detect(
            "The government policy was discussed in parliament and the public sector."
        )
        self.assertEqual(result["language"], "en")
        self.assertGreater(result["confidence"], 0.0)
        self.assertEqual(result["detector_name"], "heuristic")

    def test_detects_german(self):
        result = self.detector.detect(
            "Die Regierung und der Bundesrat haben eine neue Politik beschlossen."
        )
        self.assertEqual(result["language"], "de")

    def test_detects_french(self):
        result = self.detector.detect(
            "Le gouvernement et la politique nationale sont très importants pour nous."
        )
        self.assertEqual(result["language"], "fr")

    def test_detects_turkish(self):
        result = self.detector.detect(
            "Bu bir örnek metindir ve hükümet politikası hakkında bilgi verir."
        )
        self.assertIn(result["language"], {"tr", "und"})

    def test_empty_text_returns_und(self):
        result = self.detector.detect("   ")
        self.assertEqual(result["language"], "und")
        self.assertEqual(result["confidence"], 0.0)


class LanguageRouterTests(unittest.TestCase):
    def test_processor_mapping(self):
        router = LanguageRouter()
        self.assertEqual(router.processor_for("en"), "en")
        self.assertEqual(router.processor_for("de-DE"), "de")
        self.assertEqual(router.processor_for("xx"), "generic")


class DetectWithOverrideTests(unittest.TestCase):
    def test_manual_override_wins(self):
        result = detect_with_override("any text", manual_override="fr")
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["confidence"], 1.0)
        self.assertTrue(result["overridden"])
        self.assertEqual(result["detector_name"], "manual_override")


if __name__ == "__main__":
    unittest.main()
