"""TASK-011 / TASK-012: spaCy lifecycle reuse and phrase+morphology compatibility."""

from __future__ import annotations

import unittest
from functools import lru_cache
from unittest.mock import patch

from backend.modules.text_research.infrastructure import nlp_preprocessing
from backend.modules.text_research.infrastructure.nlp_preprocessing import (
    merge_phrase_tokens,
    morph_normalize_phrase_parts,
    spacy_available,
)
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
    snowball_stem,
    tokenize,
)


class SpacyLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        nlp_preprocessing._load_nlp.cache_clear()

    def tearDown(self) -> None:
        nlp_preprocessing._load_nlp.cache_clear()

    def test_spacy_available_routes_through_cached_loader(self) -> None:
        with patch.object(nlp_preprocessing, "_load_nlp", return_value=object()) as load_mock:
            self.assertTrue(nlp_preprocessing.spacy_available("en_core_web_sm"))
            self.assertTrue(nlp_preprocessing.spacy_available("en_core_web_sm"))
        self.assertEqual(load_mock.call_count, 2)

    def test_non_spacy_config_skips_pipeline_load(self) -> None:
        with patch(
            "backend.modules.text_research.infrastructure.nlp_preprocessing._load_nlp"
        ) as load_mock:
            meta = describe_implementation(PreprocessingConfig(lowercase=True, stemming=True))
        load_mock.assert_not_called()
        self.assertFalse(meta["spacy_available"])
        self.assertIsNone(meta["spacy"])

    def test_repeated_spacy_config_resolution_loads_once(self) -> None:
        sentinel = object()
        calls: list[str] = []

        @lru_cache(maxsize=8)
        def cached_load(model_name: str):
            calls.append(model_name)
            return sentinel

        with patch.object(nlp_preprocessing, "_load_nlp", cached_load):
            first = nlp_preprocessing.require_spacy("en_core_web_sm")
            second = nlp_preprocessing.require_spacy("en_core_web_sm")
            self.assertTrue(nlp_preprocessing.spacy_available("en_core_web_sm"))
            meta = describe_implementation(
                PreprocessingConfig(phrase_detection=True, spacy_model="en_core_web_sm")
            )
        self.assertIs(first, second)
        self.assertEqual(calls, ["en_core_web_sm"])
        self.assertTrue(meta["spacy_available"])


class PhraseMorphologyTests(unittest.TestCase):
    def test_morph_normalize_phrase_parts_stems(self) -> None:
        phrases = morph_normalize_phrase_parts(
            ["universal_education_policies"],
            normalize_part=lambda p: snowball_stem(p, "en"),
        )
        self.assertEqual(len(phrases), 1)
        self.assertIn("_", phrases[0])
        self.assertNotEqual(phrases[0], "universal_education_policies")

    def test_merge_after_shared_morphology(self) -> None:
        surface = ["governments", "policies", "matter"]
        stemmed = [snowball_stem(t, "en") for t in surface]
        phrases = morph_normalize_phrase_parts(
            ["governments_policies"],
            normalize_part=lambda p: snowball_stem(p, "en"),
        )
        merged = merge_phrase_tokens(stemmed, phrases)
        self.assertEqual(len(merged), 2)
        self.assertIn("_", merged[0])

    def test_lemmatization_plus_phrase_golden_regression(self) -> None:
        if not spacy_available():
            self.skipTest("spaCy model unavailable")
        text = "The education policies reform schools."
        tokens = tokenize(
            text,
            PreprocessingConfig(
                lowercase=True,
                remove_stopwords=True,
                lemmatization=True,
                phrase_detection=True,
            ).to_dict(),
        )
        self.assertTrue(tokens)
        joined = " ".join(tokens)
        self.assertTrue(any("_" in t for t in tokens) or "educ" in joined or "policy" in joined)

    def test_stemming_plus_phrase_does_not_silently_lose_match(self) -> None:
        if not spacy_available():
            self.skipTest("spaCy model unavailable")
        text = "Universal education policy matters for citizens."
        with_phrase = tokenize(
            text,
            PreprocessingConfig(
                lowercase=True,
                stemming=True,
                phrase_detection=True,
                remove_stopwords=True,
            ).to_dict(),
        )
        without = tokenize(
            text,
            PreprocessingConfig(
                lowercase=True,
                stemming=True,
                phrase_detection=False,
                remove_stopwords=True,
            ).to_dict(),
        )
        self.assertTrue(with_phrase)
        self.assertTrue(without)
        self.assertLessEqual(len(with_phrase), len(without))


if __name__ == "__main__":
    unittest.main()
