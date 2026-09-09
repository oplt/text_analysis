"""Tests for multilingual / language-aware processing."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import language_processing as lang
from backend.modules.text_research.infrastructure import preprocessing, segmentation

UNICODE_SAMPLE = "é ü ç ñ ø ğ ş ą č å æ"


class LanguageProcessingTests(unittest.TestCase):
    def test_unicode_tokenize_keeps_diacritics(self):
        tokens = lang.tokenize_unicode(UNICODE_SAMPLE)
        self.assertEqual(tokens, UNICODE_SAMPLE.split())

    def test_unknown_language_degrades_without_english_stopwords(self):
        profile = lang.resolve_language("zz")
        self.assertTrue(profile.degraded)
        self.assertFalse(profile.known)
        self.assertEqual(profile.stopwords, frozenset())
        self.assertFalse(profile.stemming_available)
        # Must not silently apply English stopwords
        self.assertNotIn("the", profile.stopwords)

    def test_english_has_lexicon_and_stem(self):
        profile = lang.resolve_language("en")
        self.assertFalse(profile.degraded)
        self.assertIn("the", profile.stopwords)
        self.assertTrue(profile.stemming_available)

    def test_turkish_stopwords_and_negation(self):
        profile = lang.resolve_language("tr")
        self.assertIn("ve", profile.stopwords)
        self.assertIn("değil", profile.negation_words)
        self.assertFalse(profile.stemming_available)  # no Snowball Turkish

    def test_german_stemming_available(self):
        profile = lang.resolve_language("de")
        self.assertTrue(profile.stemming_available)
        self.assertEqual(profile.snowball_algorithm, "german")

    def test_preprocess_unknown_language_does_not_drop_english_stopwords(self):
        text = "the policy and reform"
        tokens = preprocessing.tokenize(
            text,
            {
                **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
                "language": "zz",
                "remove_stopwords": True,
            },
        )
        # "the"/"and" must survive — unknown lang has empty stopword lexicon
        self.assertIn("the", tokens)
        self.assertIn("and", tokens)

    def test_preprocess_german_removes_german_stopwords(self):
        tokens = preprocessing.tokenize(
            "Der Reform und die Politik",
            {
                **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
                "language": "de",
                "remove_stopwords": True,
                "lowercase": True,
            },
        )
        self.assertNotIn("der", tokens)
        self.assertNotIn("und", tokens)
        self.assertNotIn("die", tokens)
        self.assertIn("reform", tokens)
        self.assertIn("politik", tokens)

    def test_german_stemming(self):
        tokens = preprocessing.tokenize(
            "Verbindungen",
            {
                **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
                "language": "de",
                "stemming": True,
                "lemmatization": False,
            },
        )
        self.assertEqual(len(tokens), 1)
        self.assertNotEqual(tokens[0], "verbindungen")  # stemmed

    def test_describe_marks_degraded(self):
        impl = preprocessing.describe_implementation(
            {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "language": "zz"}
        )
        self.assertTrue(impl["degraded"])
        self.assertTrue(impl["language_profile"]["degraded"])

    def test_sentence_split_unicode_sample(self):
        text = f"Uno {UNICODE_SAMPLE}. Dos más."
        spans = lang.split_sentences(text, "es")
        self.assertGreaterEqual(len(spans), 2)
        self.assertTrue(any("é" in s[2] or "ñ" in s[2] for s in spans))

    def test_segmentation_accepts_language(self):
        text = "First sentence. Second sentence."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(len(units), 2)
        unknown = segmentation.segment_document(text, "sentence", language="zz")
        self.assertEqual(len(unknown), 2)  # still works, degraded heuristics

    def test_list_supported_languages_nonempty(self):
        catalog = lang.list_supported_languages()
        codes = {row["code"] for row in catalog}
        self.assertIn("en", codes)
        self.assertIn("de", codes)
        self.assertIn("tr", codes)


if __name__ == "__main__":
    unittest.main()
