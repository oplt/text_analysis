"""Tests for optional NLP preprocessing helpers and config wiring (§19)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.nlp_preprocessing import (
    merge_phrase_tokens,
    spacy_available,
)
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
    tokenize,
)


class NlpPreprocessingHelpersTests(unittest.TestCase):
    def test_merge_phrase_tokens_joins_known_phrases(self):
        tokens = ["universal", "education", "policy", "matters"]
        out = merge_phrase_tokens(tokens, ["universal_education_policy"])
        self.assertEqual(out, ["universal_education_policy", "matters"])

    def test_spacy_unavailable_is_honest(self):
        # Environment may or may not have spaCy; the helper must return a bool.
        self.assertIsInstance(spacy_available("en_core_web_sm"), bool)


class PreprocessingConfigNlpTests(unittest.TestCase):
    def test_classical_defaults_unchanged(self):
        cfg = PreprocessingConfig()
        self.assertFalse(cfg.pos_lemmatization)
        self.assertFalse(cfg.entity_masking)
        self.assertFalse(cfg.phrase_detection)
        self.assertFalse(cfg.enable_ner)
        self.assertEqual(cfg.language_mode, "manual")

    def test_auto_detect_promotes_language_mode(self):
        cfg = PreprocessingConfig.from_dict({"auto_detect_language": True})
        self.assertEqual(cfg.language_mode, "auto")

    def test_multilingual_sets_per_unit_mode(self):
        cfg = PreprocessingConfig.from_dict({"multilingual": True})
        self.assertEqual(cfg.language_mode, "per_unit")

    def test_pos_lemmatization_conflicts_with_stemming(self):
        with self.assertRaises(ValueError):
            PreprocessingConfig.from_dict({"pos_lemmatization": True, "stemming": True})

    def test_describe_implementation_records_nlp_flags(self):
        cfg = PreprocessingConfig(lemmatization=True, language="en")
        meta = describe_implementation(cfg)
        self.assertEqual(meta["preprocessing_implementation_version"], "3")
        self.assertTrue(meta["lemmatization_requested"])
        self.assertIn("spacy", meta)
        self.assertIn("spacy_available", meta)

    def test_classical_tokenize_still_works(self):
        tokens = tokenize(
            "The governments are discussing policies.",
            PreprocessingConfig(remove_stopwords=True, stemming=True).to_dict(),
        )
        self.assertTrue(tokens)
        self.assertNotIn("the", tokens)

    def test_entity_masking_without_spacy_raises(self):
        if spacy_available():
            self.skipTest("spaCy present; cannot assert unavailable path")
        with self.assertRaises(ValueError):
            tokenize(
                "Angela Merkel visited Paris.",
                PreprocessingConfig(entity_masking=True).to_dict(),
            )

    def test_prepare_texts_honors_auto_detect_from_config(self):
        artifact = prepare_texts(
            ["The education policy reform is important for students."],
            PreprocessingConfig(auto_detect_language=True),
            force_in_memory=True,
        )
        detection = artifact.provenance.get("language_detection")
        self.assertIsNotNone(detection)
        self.assertIn("language", detection)


if __name__ == "__main__":
    unittest.main()
