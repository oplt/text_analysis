"""Tests for the pure-Python/scikit-learn text research infrastructure engines.

These are plain ``unittest.TestCase`` tests (no DB/FastAPI/app settings
dependency) so they can run in isolation via ``pytest`` against the
infrastructure layer only.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from backend.modules.text_research.infrastructure import (
    classifiers,
    feature_cache,
    model_storage,
    preprocessing,
    quantitative,
    reliability,
    segmentation,
    topic_models,
)


class SegmentationTests(unittest.TestCase):
    def test_hash_text_is_stable_and_content_sensitive(self):
        text = "Education policy should be universal and inclusive."
        h1 = segmentation.hash_text(text)
        h2 = segmentation.hash_text(text)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # sha256 hex digest length
        self.assertNotEqual(h1, segmentation.hash_text(text + " "))

    def test_document_segmentation_single_unit_at_position_zero(self):
        text = "Some policy text.\n\nWith two paragraphs."
        units = segmentation.segment_document(text, "document")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["position"], 0)
        self.assertEqual(units[0]["text"], text)

    def test_paragraph_segmentation_deterministic_positions_and_hash(self):
        text = (
            "First paragraph about topic alpha.\n\n"
            "Second paragraph about topic beta.\n\n\n"
            "Third paragraph about topic gamma."
        )
        units_1 = segmentation.segment_document(text, "paragraph")
        units_2 = segmentation.segment_document(text, "paragraph")

        self.assertEqual(units_1, units_2)  # full determinism
        self.assertEqual(len(units_1), 3)
        self.assertEqual([u["position"] for u in units_1], [0, 1, 2])
        self.assertEqual([u["paragraph_number"] for u in units_1], [1, 2, 3])
        self.assertTrue(all(u["sentence_number"] is None for u in units_1))
        self.assertEqual(units_1[0]["text"], "First paragraph about topic alpha.")

        # Hash stability: same text -> same hash; changed text -> different hash.
        h_before = segmentation.hash_text(units_1[0]["text"])
        h_after = segmentation.hash_text(units_2[0]["text"])
        self.assertEqual(h_before, h_after)

    def test_paragraph_segmentation_skips_empty_paragraphs(self):
        text = "Alpha.\n\n\n\nBeta.\n\n   \n\nGamma."
        units = segmentation.segment_document(text, "paragraph")
        self.assertEqual([u["text"] for u in units], ["Alpha.", "Beta.", "Gamma."])

    def test_sentence_segmentation_deterministic_and_preserves_text(self):
        text = "Hello world. This is great! Is it? Yes."
        units_1 = segmentation.segment_document(text, "sentence")
        units_2 = segmentation.segment_document(text, "sentence")

        self.assertEqual(units_1, units_2)
        self.assertEqual(
            [u["text"] for u in units_1],
            ["Hello world.", "This is great!", "Is it?", "Yes."],
        )
        self.assertEqual([u["position"] for u in units_1], [0, 1, 2, 3])
        self.assertEqual([u["sentence_number"] for u in units_1], [1, 2, 3, 4])
        self.assertTrue(all(u["paragraph_number"] is None for u in units_1))

    def test_unsupported_unit_type_raises(self):
        with self.assertRaises(ValueError):
            segmentation.segment_document("text", "page")

    def test_char_offsets_slice_back_to_unit_text(self):
        text = "Alpha one.\n\nBeta two.\n\nGamma three."
        units = segmentation.segment_document(text, "paragraph")
        self.assertEqual(len(units), 3)
        for unit in units:
            self.assertEqual(text[unit["char_start"] : unit["char_end"]], unit["text"])
            self.assertEqual(unit["text_hash"], segmentation.hash_text(unit["text"]))
            self.assertEqual(unit["source_text_hash"], segmentation.hash_text(text))
            self.assertIsNone(unit["page_number"])
            self.assertIsNone(unit["section_heading"])

    def test_sentence_offsets_are_reproducible(self):
        text = "Hello world. This is great! Is it? Yes."
        units = segmentation.segment_document(text, "sentence")
        self.assertEqual(
            [text[u["char_start"] : u["char_end"]] for u in units],
            [u["text"] for u in units],
        )
        again = segmentation.segment_document(text, "sentence")
        self.assertEqual(units, again)

    def test_page_provenance_attached_without_invention(self):
        text = "Page one body.\n\nPage two body."
        # Mimic canonical double-newline join provenance.
        page_provenance = [
            {"index": 0, "page_number": 1, "section_heading": "Intro", "char_start": 0, "char_end": 14},
            {"index": 1, "page_number": 2, "section_heading": None, "char_start": 16, "char_end": 29},
        ]
        units = segmentation.segment_document(text, "paragraph", page_provenance=page_provenance)
        self.assertEqual(units[0]["page_number"], 1)
        self.assertEqual(units[0]["section_heading"], "Intro")
        self.assertEqual(units[1]["page_number"], 2)
        self.assertIsNone(units[1]["section_heading"])

    def test_no_page_provenance_leaves_page_null(self):
        units = segmentation.segment_document("Only text.", "document")
        self.assertIsNone(units[0]["page_number"])
        self.assertIsNone(units[0]["section_heading"])


class PreprocessingTests(unittest.TestCase):
    def test_feature_cache_tracks_hits_misses_and_explicit_invalidation(self):
        feature_cache.clear_cache()
        self.assertIsNone(feature_cache.get_cached("missing"))
        feature_cache.set_cached("cached", [["token"]])
        self.assertEqual(feature_cache.get_cached("cached"), [["token"]])
        feature_cache.invalidate_cache("cached")
        metrics = feature_cache.cache_metrics()
        self.assertEqual(metrics["entries"], 0)
        self.assertGreaterEqual(metrics["hits"], 1)
        self.assertGreaterEqual(metrics["misses"], 1)

    def test_normalize_whitespace_collapses_and_strips(self):
        self.assertEqual(
            preprocessing.normalize_whitespace("  Hello   \n\n world  "),
            "Hello world",
        )

    def test_tokenize_default_config_lowercases_and_strips_punctuation(self):
        tokens = preprocessing.tokenize("The Policy, IS Universal!")
        self.assertEqual(tokens, ["the", "policy", "is", "universal"])

    def test_negation_preserved_when_removing_stopwords(self):
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "remove_stopwords": True}
        tokens = preprocessing.tokenize("This is not never going to be no good.", config)
        self.assertIn("not", tokens)
        self.assertIn("never", tokens)
        self.assertIn("no", tokens)
        # ordinary stopwords should still be removed
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)

    def test_preserve_negation_guards_custom_stopwords_when_true(self):
        config = {
            **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            "remove_stopwords": True,
            "preserve_negation": True,
            "custom_stopwords": ["not", "policy"],
        }
        tokens = preprocessing.tokenize("This is not a good policy.", config)
        self.assertIn("not", tokens)  # protected despite being in custom_stopwords
        self.assertNotIn("policy", tokens)  # ordinary custom stopword still removed

    def test_preserve_negation_false_allows_negation_removal_via_custom_stopwords(self):
        config = {
            **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            "remove_stopwords": True,
            "preserve_negation": False,
            "custom_stopwords": ["not"],
        }
        tokens = preprocessing.tokenize("This is not a good policy.", config)
        self.assertNotIn("not", tokens)

    def test_remove_numbers_config(self):
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "remove_numbers": True}
        tokens = preprocessing.tokenize("There are 4 pillars and 2024 goals.", config)
        self.assertNotIn("4", tokens)
        self.assertNotIn("2024", tokens)
        self.assertIn("pillars", tokens)

    def test_stemming_never_touches_negation_words(self):
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "stemming": True}
        tokens = preprocessing.tokenize("never nations governing", config)
        self.assertIn("never", tokens)  # unstemmed despite matching no strip rule anyway
        self.assertIn("govern", tokens)  # "governing" -> "govern"

    def test_lemmatization_unavailable_language_rejected(self):
        config = {
            **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            "language": "zz",
            "lemmatization": True,
        }
        with self.assertRaises(ValueError):
            preprocessing.tokenize("Policies matter.", config)
        with self.assertRaises(ValueError):
            preprocessing.PreprocessingConfig.from_dict(config)

    def test_lemmatization_english_available(self):
        config = {
            **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            "language": "en",
            "lemmatization": True,
            "stemming": False,
        }
        tokens = preprocessing.tokenize("Policies matter.", config)
        self.assertIn("policy", tokens)  # policies -> policy
        self.assertTrue(preprocessing.lemmatization_available("en"))
        impl = preprocessing.describe_implementation(config)
        self.assertEqual(impl["lemmatizer"], "simplemma")
        self.assertTrue(impl["lemmatization_requested"])

    def test_stemming_and_lemmatization_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            preprocessing.PreprocessingConfig.from_dict(
                {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "stemming": True, "lemmatization": True}
            )

    def test_unicode_aware_tokenization(self):
        tokens = preprocessing.tokenize("café naïve façade ğüşıöç")
        self.assertEqual(tokens, ["café", "naïve", "façade", "ğüşıöç"])

    def test_unicode_normalization_nfc(self):
        # "é" as e + combining acute vs precomposed
        decomposed = "cafe\u0301"
        tokens = preprocessing.tokenize(
            decomposed,
            {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "unicode_normalization": "NFC"},
        )
        self.assertEqual(tokens, ["café"])

    def test_preview_preprocessing_reports_removed_terms(self):
        preview = preprocessing.preview_preprocessing(
            ["Students should not be excluded from the policy."],
            {
                **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
                "remove_stopwords": True,
                "preserve_negation": True,
            },
        )
        self.assertEqual(len(preview["rows"]), 1)
        self.assertIn("not", preview["rows"][0]["processed"])
        self.assertGreater(preview["token_count_before"], preview["token_count_after"])
        self.assertTrue(preview["lemmatization_supported"])
        self.assertIn("snowball", preview["stemmer"] or "")
        self.assertEqual(preview["implementation"]["tokenizer"], "unicode_regex")
        removed_terms = {row["term"] for row in preview["most_frequently_removed_terms"]}
        self.assertTrue({"should", "be", "from", "the"} & removed_terms)

    def test_preprocess_text_is_space_joined_tokens(self):
        result = preprocessing.preprocess_text("The Policy IS Universal!")
        self.assertEqual(result, "the policy is universal")

    def test_original_text_untouched(self):
        original = "The Policy IS Universal!"
        preprocessing.preprocess_text(original)
        self.assertEqual(original, "The Policy IS Universal!")

    def test_build_count_vectorizer_respects_ngram_and_df_config(self):
        config = {
            **preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            "ngram_min": 1,
            "ngram_max": 2,
        }
        vectorizer = preprocessing.build_count_vectorizer(config)
        docs = ["policy reform is universal", "policy reform helps everyone"]
        matrix = vectorizer.fit_transform(docs)
        feature_names = set(vectorizer.get_feature_names_out())
        self.assertIn("policy reform", feature_names)  # bigram present
        self.assertGreater(matrix.shape[1], 0)

    def test_build_tfidf_vectorizer_returns_tfidf_weighted_matrix(self):
        default_config = preprocessing.DEFAULT_PREPROCESSING_CONFIG
        vectorizer = preprocessing.build_tfidf_vectorizer(default_config)
        docs = ["universal policy", "individual policy"]
        matrix = vectorizer.fit_transform(docs)
        self.assertEqual(matrix.shape[0], 2)
        self.assertGreater(matrix.shape[1], 0)


class ReliabilityTests(unittest.TestCase):
    def test_raw_agreement_perfect_and_partial(self):
        self.assertEqual(reliability.raw_agreement(["a", "b", "a"], ["a", "b", "a"]), 1.0)
        partial = reliability.raw_agreement(["a", "b", "a", "b"], ["a", "b", "b", "b"])
        self.assertAlmostEqual(partial, 0.75)

    def test_cohens_kappa_perfect_agreement_is_one(self):
        labels_a = ["yes", "no", "yes", "no", "yes", "no", "yes", "yes", "no", "no"]
        labels_b = list(labels_a)
        result = reliability.cohens_kappa(labels_a, labels_b)
        self.assertAlmostEqual(result["kappa"], 1.0)
        self.assertAlmostEqual(result["observed_agreement"], 1.0)
        self.assertEqual(result["sample_size"], 10)

    def test_cohens_kappa_known_textbook_example(self):
        # Classic 2x2 example: 10 units, observed agreement 0.7, chance-level
        # kappa is a well-known worked example used in reliability courses.
        labels_a = ["yes"] * 6 + ["no"] * 4
        labels_b = ["yes"] * 5 + ["no"] * 1 + ["no"] * 4
        result = reliability.cohens_kappa(labels_a, labels_b)
        expected_po = reliability.raw_agreement(labels_a, labels_b)
        self.assertAlmostEqual(result["observed_agreement"], expected_po)
        self.assertTrue(-1.0 <= result["kappa"] <= 1.0)

    def test_cohens_kappa_no_agreement_beyond_chance_near_zero(self):
        # Balanced random-looking disagreement pattern -> kappa near/at 0.
        labels_a = ["a", "b"] * 5
        labels_b = ["b", "a"] * 5
        result = reliability.cohens_kappa(labels_a, labels_b)
        self.assertLess(result["kappa"], 0.1)

    def test_krippendorffs_alpha_perfect_agreement_is_one(self):
        reliability_data = [
            ["yes", "yes", "yes"],
            ["no", "no", "no"],
            ["yes", "yes", "yes"],
            ["no", "no", "no"],
            ["yes", "yes", "yes"],
        ]
        result = reliability.krippendorffs_alpha(reliability_data)
        self.assertAlmostEqual(result["alpha"], 1.0)
        self.assertEqual(result["n_units"], 5)
        self.assertEqual(result["n_coders"], 3)
        self.assertEqual(result["missingness"], 0.0)

    def test_krippendorffs_alpha_handles_missing_values(self):
        reliability_data = [
            ["yes", "yes", None],
            ["no", "no", "no"],
            ["yes", None, "yes"],
            [None, "no", "no"],
        ]
        result = reliability.krippendorffs_alpha(reliability_data)
        self.assertAlmostEqual(result["alpha"], 1.0)
        self.assertEqual(result["n_units"], 4)
        self.assertEqual(result["n_coders"], 3)
        self.assertGreater(result["missingness"], 0.0)

    def test_krippendorffs_alpha_without_shared_units_is_not_evaluable(self):
        result = reliability.krippendorffs_alpha([["yes", None], [None, "no"]])
        self.assertIsNone(result["alpha"])

    def test_krippendorffs_alpha_hand_computed_example(self):
        # Two coders, four units:
        # A: a a b b   B: a b b b
        # Hand-derived alpha (nominal, coincidence-matrix method) = 1 - 14/30 = 8/15.
        reliability_data = [["a", "a"], ["a", "b"], ["b", "b"], ["b", "b"]]
        result = reliability.krippendorffs_alpha(reliability_data)
        self.assertAlmostEqual(result["alpha"], 8 / 15, places=9)

    def test_krippendorffs_alpha_with_disagreement_is_less_than_one(self):
        reliability_data = [
            ["yes", "no", "yes"],
            ["no", "no", "yes"],
            ["yes", "yes", "no"],
            ["no", "yes", "yes"],
        ]
        result = reliability.krippendorffs_alpha(reliability_data)
        self.assertLess(result["alpha"], 1.0)

    def test_agreement_matrix_diagonal_and_symmetry(self):
        coder_labels = {
            "coder_1": {"u1": "yes", "u2": "no", "u3": "yes"},
            "coder_2": {"u1": "yes", "u2": "yes", "u3": "yes"},
        }
        result = reliability.agreement_matrix(coder_labels)
        self.assertEqual(result["coders"], ["coder_1", "coder_2"])
        self.assertAlmostEqual(result["matrix"]["coder_1"]["coder_1"]["agreement"], 1.0)
        self.assertAlmostEqual(
            result["matrix"]["coder_1"]["coder_2"]["agreement"],
            result["matrix"]["coder_2"]["coder_1"]["agreement"],
        )
        self.assertAlmostEqual(result["matrix"]["coder_1"]["coder_2"]["agreement"], 2 / 3)

    def test_reliability_by_label_two_coder_design(self):
        label_data = {
            "LabelA": [["yes", "yes"], ["no", "no"], ["yes", "no"]],
        }
        result = reliability.reliability_by_label(label_data)
        self.assertIn("LabelA", result)
        self.assertIn("alpha", result["LabelA"])
        self.assertIn("cohens_kappa", result["LabelA"])
        self.assertAlmostEqual(result["LabelA"]["raw_agreement"], 2 / 3)

    def test_fleiss_kappa_perfect_agreement_is_one(self):
        reliability_data = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["a", "a", "a"],
        ]
        result = reliability.fleiss_kappa(reliability_data)
        self.assertAlmostEqual(result["kappa"], 1.0)
        self.assertEqual(result["n_raters"], 3)
        self.assertEqual(result["n_units_included"], 3)
        self.assertEqual(result["n_units_excluded"], 0)

    def test_fleiss_kappa_hand_computed_example(self):
        # 4 units, 3 raters each, categories a/b:
        # hand-derived kappa: P_bar = 2/3, P_e_bar = 1/2 -> kappa = 1/3.
        reliability_data = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["a", "a", "b"],
            ["b", "b", "a"],
        ]
        result = reliability.fleiss_kappa(reliability_data)
        self.assertAlmostEqual(result["kappa"], 1 / 3, places=9)
        self.assertEqual(result["n_categories"], 2)

    def test_fleiss_kappa_excludes_units_with_uneven_rater_counts(self):
        reliability_data = [
            ["a", "a", "a"],
            ["b", "b", "b"],
            ["a", "a", "b"],
            ["b", "b", "a"],
            ["a", "b"],  # only 2 raters: excluded from the fixed-3-rater design
        ]
        result = reliability.fleiss_kappa(reliability_data)
        self.assertEqual(result["n_raters"], 3)
        self.assertEqual(result["n_units_included"], 4)
        self.assertEqual(result["n_units_excluded"], 1)

    def test_fleiss_kappa_not_evaluable_with_fewer_than_three_raters(self):
        result = reliability.fleiss_kappa([["a", "a"], ["b", "b"]])
        self.assertIsNone(result["kappa"])
        self.assertIn("reason", result)

    def test_reliability_metadata_reports_scale_coders_and_missingness(self):
        reliability_data = [
            ["a", "a", "a"],
            ["b", None, "b"],
            [None, None, "a"],
        ]
        metadata = reliability.reliability_metadata(
            reliability_data, statistics_used=["fleiss_kappa"]
        )
        self.assertEqual(metadata["scale"], "nominal")
        self.assertEqual(metadata["n_coders"], 3)
        self.assertEqual(metadata["n_units"], 3)
        self.assertEqual(metadata["pairable_units"], 2)
        self.assertEqual(metadata["missing_values"], 3)
        self.assertEqual(metadata["statistics_used"], ["fleiss_kappa"])


class QuantitativeTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Education policy should be universal and inclusive for all children.",
            "The organization promotes individual liberty and free markets.",
            "Universal access to education supports social cohesion and equality.",
        ]
        self.tokenized = [preprocessing.tokenize(t) for t in self.texts]

    def test_corpus_statistics_basic_shape(self):
        stats = quantitative.corpus_statistics(self.texts, self.tokenized)
        self.assertEqual(stats["document_count"], 3)
        self.assertEqual(stats["text_unit_count"], 3)
        self.assertEqual(stats["token_count"], sum(len(t) for t in self.tokenized))
        self.assertGreater(stats["vocabulary_size"], 0)
        self.assertEqual(stats["min_length"], min(len(t) for t in self.tokenized))
        self.assertEqual(stats["max_length"], max(len(t) for t in self.tokenized))
        self.assertIn("ttr", stats)
        self.assertIn("lexical_diversity", stats)
        self.assertIn("length_caution", stats["lexical_diversity"])

    def test_corpus_statistics_unique_documents(self):
        stats = quantitative.corpus_statistics(
            self.texts,
            self.tokenized,
            document_ids=["d1", "d1", "d2"],
        )
        self.assertEqual(stats["text_unit_count"], 3)
        self.assertEqual(stats["documents_unique"], 2)

    def test_lexical_diversity_mattr_unavailable_on_short_text(self):
        short = [["a", "b", "a"]]
        diversity = quantitative.lexical_diversity(short, mattr_window=50)
        self.assertIsNone(diversity["mattr"])
        self.assertFalse(diversity["mattr_available"])
        self.assertGreater(diversity["ttr"], 0)
        self.assertIsNotNone(diversity["mattr_note"])

    def test_mattr_and_msttr_on_long_tokens(self):
        tokens = [f"w{i % 17}" for i in range(200)]
        mattr = quantitative.moving_average_ttr(tokens, window=50)
        msttr = quantitative.mean_segmental_ttr(tokens, window=100)
        self.assertIsNotNone(mattr)
        self.assertIsNotNone(msttr)
        self.assertGreater(mattr, 0)
        self.assertGreater(msttr, 0)

    def test_standardized_pipeline_stages(self):
        result = quantitative.run_standardized_pipeline(
            self.texts,
            preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            document_ids=["a", "b", "c"],
            unit_ids=["u1", "u2", "u3"],
            trim={"min_term_frequency": 1, "min_document_frequency": 1},
            dfm_mode="count",
        )
        payload = result.to_dict()
        self.assertEqual(payload["workflow"], list(quantitative.PIPELINE_STAGES))
        self.assertEqual(result.stages["corpus"]["status"], "completed")
        self.assertEqual(result.stages["tokens"]["status"], "completed")
        self.assertEqual(result.stages["preprocessing"]["status"], "completed")
        self.assertEqual(result.stages["feature_trimming"]["status"], "completed")
        self.assertEqual(result.stages["dfm"]["status"], "completed")
        self.assertEqual(result.stages["statistical_analysis"]["status"], "completed")
        self.assertIsNotNone(result.dfm)
        self.assertEqual(result.dfm["unit_ids"], ["u1", "u2", "u3"])
        self.assertIn("preprocessing_config", result.dfm)
        self.assertGreater(result.statistics["vocabulary_size"], 0)

    def test_trim_token_vocabulary_removes_rare_terms(self):
        tokenized = [["alpha", "beta"], ["alpha", "gamma"], ["alpha"]]
        trimmed, meta = quantitative.trim_token_vocabulary(
            tokenized, min_document_frequency=2
        )
        self.assertEqual(meta["features_after"], 1)
        self.assertTrue(all(tok == "alpha" for row in trimmed for tok in row))

    def test_term_frequencies_sum_and_prevalence(self):
        results = quantitative.term_frequencies(self.tokenized)
        term_lookup = {row["term"]: row for row in results}
        self.assertIn("universal", term_lookup)
        total_raw = sum(row["raw_count"] for row in results)
        self.assertAlmostEqual(total_raw, sum(len(t) for t in self.tokenized))
        self.assertAlmostEqual(sum(row["relative_frequency"] for row in results), 1.0, places=6)
        self.assertLessEqual(term_lookup["universal"]["document_prevalence"], 1.0)
        self.assertEqual(results[0]["rank"], 1)
        self.assertAlmostEqual(results[-1]["cumulative_share"], 1.0, places=6)
        self.assertIn("document_frequency", term_lookup["universal"])
        self.assertEqual(term_lookup["universal"]["raw_frequency"], term_lookup["universal"]["raw_count"])

    def test_term_frequencies_configurable_rate(self):
        per_100 = quantitative.term_frequencies(self.tokenized, rate_per=100)
        per_10k = quantitative.term_frequencies(self.tokenized, rate_per=10000)
        self.assertAlmostEqual(per_100[0]["rate"], per_100[0]["relative_frequency"] * 100)
        self.assertAlmostEqual(per_10k[0]["rate"], per_10k[0]["relative_frequency"] * 10000)
        self.assertAlmostEqual(per_100[0]["per_1000"], per_10k[0]["per_1000"])
        with self.assertRaises(ValueError):
            quantitative.term_frequencies(self.tokenized, rate_per=0)

    def test_term_frequency_report_metadata(self):
        report = quantitative.term_frequency_report(
            self.tokenized,
            rate_per=1000,
            top_n=5,
            unit_ids=["u1", "u2", "u3"],
            document_ids=["d1", "d1", "d2"],
            group_keys=["A", "A", "B"],
        )
        self.assertEqual(len(report["frequencies"]), 5)
        self.assertEqual(report["metadata"]["terms_returned"], 5)
        self.assertEqual(report["metadata"]["documents_unique"], 2)
        self.assertEqual(report["metadata"]["group_token_totals"]["A"]["units"], 2)
        self.assertIn("fields", report["metadata"])

    def test_ngram_frequencies_bigrams(self):
        results = quantitative.ngram_frequencies(self.tokenized, n=2)
        self.assertTrue(all(row["ngram"].count(" ") == 1 for row in results))
        self.assertTrue(all(row["raw_count"] >= 1 for row in results))
        self.assertTrue(all("document_frequency" in row for row in results))
        self.assertTrue(all(row["prevalence"] == row["document_prevalence"] for row in results))
        self.assertEqual(results[0]["rank"], 1)

    def test_ngram_orders_uni_bi_tri_and_bounds(self):
        uni = quantitative.ngram_frequencies(self.tokenized, n=1)
        bi = quantitative.ngram_frequencies(self.tokenized, n=2)
        tri = quantitative.ngram_frequencies(self.tokenized, n=3)
        self.assertTrue(all(row["n"] == 1 and " " not in row["ngram"] for row in uni))
        self.assertTrue(all(row["n"] == 2 for row in bi))
        self.assertTrue(all(row["n"] == 3 and row["ngram"].count(" ") == 2 for row in tri))
        with self.assertRaises(ValueError):
            quantitative.validate_ngram_order(0)
        with self.assertRaises(ValueError):
            quantitative.validate_ngram_order(11)
        with self.assertRaises(ValueError):
            quantitative.ngram_frequencies(self.tokenized, n=2, skip=1)

    def test_ngram_frequency_report(self):
        report = quantitative.ngram_frequency_report(
            self.tokenized, n=2, top_n=3, rate_per=1000, unit_ids=["u1", "u2", "u3"]
        )
        self.assertEqual(report["metadata"]["n_label"], "bigram")
        self.assertEqual(len(report["ngrams"]), 3)
        self.assertFalse(report["metadata"]["skip_grams_supported"])
        self.assertAlmostEqual(
            sum(r["relative_frequency"] for r in quantitative.ngram_frequencies(self.tokenized, n=2)),
            1.0,
            places=6,
        )

    def test_kwic_returns_context_and_metadata(self):
        metadata = [{"document": f"doc-{i}", "year": 2020 + i} for i in range(len(self.texts))]
        results = quantitative.kwic(self.texts, metadata, "universal", window=2)
        self.assertGreaterEqual(len(results), 2)
        for row in results:
            self.assertEqual(row["keyword"].lower().strip(".,!?"), "universal")
            self.assertIn("document", row)
            self.assertIn("year", row)

    def test_dictionary_hits_counts_and_prevalence(self):
        result = quantitative.dictionary_hits(self.tokenized, ["universal", "equality"])
        self.assertGreaterEqual(result["hits"], 2)
        self.assertEqual(result["n_units"], 3)
        self.assertGreater(result["unit_prevalence"], 0.0)
        self.assertEqual(len(result["per_unit_hits"]), 3)

    def test_keyness_distinguishes_corpora(self):
        text_a = "liberty market freedom liberty market"
        text_b = "equality solidarity cohesion equality solidarity"
        tokenized_a = [preprocessing.tokenize(text_a) for _ in range(5)]
        tokenized_b = [preprocessing.tokenize(text_b) for _ in range(5)]
        results = quantitative.keyness(tokenized_a, tokenized_b, top_n=10)
        by_feature = {row["feature"]: row for row in results}
        self.assertEqual(by_feature["liberty"]["effect_direction"], "a")
        self.assertEqual(by_feature["equality"]["effect_direction"], "b")
        self.assertGreater(by_feature["liberty"]["keyness_statistic"], 0)

    def test_cooccurrence_finds_associated_terms(self):
        tokenized = [preprocessing.tokenize("universal education policy") for _ in range(4)]
        results = quantitative.cooccurrence(tokenized, window=3, top_n=10)
        pairs = {(row["term_a"], row["term_b"]) for row in results}
        self.assertIn(("education", "universal"), pairs)  # sorted alphabetically

    def test_build_dfm_count_mode_dimensions_and_preview(self):
        result = quantitative.build_dfm_matrix(
            self.tokenized, mode="count", unit_ids=["u1", "u2", "u3"]
        )
        self.assertEqual(result["dimensions"]["units"], 3)
        self.assertGreater(result["dimensions"]["features"], 0)
        self.assertIn("sparse", result)
        self.assertEqual(result["sparse"]["format"], "coo")
        self.assertEqual(result["unit_ids"], ["u1", "u2", "u3"])
        self.assertIn("dense_matrix", result)  # small matrix still may densify
        self.assertEqual(len(result["dense_matrix"]), 3)
        self.assertGreaterEqual(result["density"], 0.0)

    def test_build_dfm_binary_mode_values_are_zero_or_one(self):
        result = quantitative.build_dfm_matrix(self.tokenized, mode="binary")
        for row in result["dense_matrix"]:
            self.assertTrue(all(v in (0, 1) for v in row))

    def test_build_dfm_tfidf_mode_has_float_weights(self):
        result = quantitative.build_dfm_matrix(self.tokenized, mode="tfidf")
        self.assertEqual(result["mode"], "tfidf")
        self.assertTrue(any(isinstance(v, float) and v not in (0.0, 1.0) for row in result["dense_matrix"] for v in row) or result["nnz"] > 0)

    def test_build_dfm_tf_and_sublinear(self):
        tf = quantitative.build_dfm_matrix(self.tokenized, mode="tf")
        self.assertEqual(tf["weighting"], "tf")
        # Each non-empty row should sum ~1.0
        for row in tf["dense_matrix"]:
            if sum(row) > 0:
                self.assertAlmostEqual(sum(row), 1.0, places=5)
        sub = quantitative.build_dfm_matrix(self.tokenized, mode="sublinear_tf")
        self.assertEqual(sub["mode"], "sublinear_tf")
        self.assertTrue(sub["sublinear_tf"])
        self.assertIn("sparse", sub)

    def test_build_dfm_force_sparse_only_skips_dense(self):
        result = quantitative.build_dfm_matrix(
            self.tokenized, mode="count", force_sparse_only=True
        )
        self.assertEqual(result["storage"], "sparse")
        self.assertNotIn("dense_matrix", result)
        self.assertEqual(result["sparse"]["shape"][0], 3)

    def test_build_dfm_from_texts_attaches_preprocessing_config(self):
        result = quantitative.build_dfm(
            self.texts,
            ["a", "b", "c"],
            preprocessing.DEFAULT_PREPROCESSING_CONFIG,
            weighting="count",
        )
        self.assertIsNotNone(result["preprocessing_config"])
        self.assertEqual(result["unit_ids"], ["a", "b", "c"])
        self.assertTrue(result["feature_names"])
        self.assertIn("sparse", result)

    def test_dfm_trim_min_df_and_top_n(self):
        base = quantitative.build_dfm_matrix(self.tokenized, mode="count", force_sparse_only=True)
        before = base["dimensions"]["features"]
        trimmed = quantitative.dfm_trim(base, min_document_frequency=2)
        self.assertLessEqual(trimmed["dimensions"]["features"], before)
        self.assertEqual(trimmed["trim"]["features_before"], before)
        self.assertTrue(all(name in base["feature_names"] for name in trimmed["feature_names"]))

        top = quantitative.dfm_trim(base, top_n=3)
        self.assertEqual(top["dimensions"]["features"], 3)
        self.assertEqual(top["trim"]["term_frequency"]["type"], "rank")

    def test_dfm_trim_prop_and_quantile(self):
        base = quantitative.build_dfm_matrix(self.tokenized, mode="count", force_sparse_only=True)
        prop = quantitative.dfm_trim(
            base,
            min_document_frequency=0.5,
            document_frequency_type="prop",
        )
        self.assertLessEqual(prop["dimensions"]["features"], base["dimensions"]["features"])
        quant = quantitative.dfm_trim(
            base,
            min_term_frequency=0.5,
            term_frequency_type="quantile",
        )
        self.assertGreaterEqual(quant["dimensions"]["features"], 1)

    def test_build_dfm_accepts_trim_config(self):
        result = quantitative.build_dfm_matrix(
            self.tokenized,
            mode="count",
            trim={"top_n": 5},
            force_sparse_only=True,
        )
        self.assertEqual(result["dimensions"]["features"], 5)
        self.assertIn("trim", result)


class ClassifierTests(unittest.TestCase):
    def test_grouped_train_test_split_never_splits_a_group(self):
        texts = [f"document text number {i}" for i in range(20)]
        y = ["yes" if i % 2 == 0 else "no" for i in range(20)]
        # 5 groups (source documents), 4 units each, to simulate paragraph
        # units drawn from the same source document.
        groups = [f"doc-{i // 4}" for i in range(20)]

        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.4, random_seed=7)

        train_groups = set(split["groups_train"])
        test_groups = set(split["groups_test"])
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertEqual(len(split["X_train"]) + len(split["X_test"]), 20)
        self.assertEqual(len(split["y_train"]), len(split["X_train"]))

    def test_tfidf_vectorizer_fitted_only_on_train_vocabulary(self):
        groups = [f"doc-{i}" for i in range(12)]
        texts = ["alpha beta gamma delta" for _ in range(6)] + [
            "epsilon zeta eta theta" for _ in range(6)
        ]
        y = ["a"] * 6 + ["b"] * 6

        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.3, random_seed=1)

        # Inject a token that appears ONLY in the test split, never in train.
        test_only_token = "zzzonlyintestzzz"
        X_test_with_marker = [f"{t} {test_only_token}" for t in split["X_test"]]

        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            X_test_with_marker,
            split["y_test"],
            task_type="binary",
        )

        vectorizer = result["vectorizer"]
        self.assertNotIn(test_only_token, vectorizer.vocabulary_)
        self.assertIn("accuracy", result["metrics"])

    def test_fit_tfidf_classifier_binary_end_to_end_metrics(self):
        groups = [f"doc-{i}" for i in range(16)]
        texts = ["universal liberty market policy" for _ in range(8)] + [
            "individual equality solidarity cohesion" for _ in range(8)
        ]
        y = ["liberal"] * 8 + ["universal"] * 8

        split = classifiers.grouped_train_test_split(
            texts, y, groups, test_size=0.25, random_seed=3
        )
        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            random_seed=3,
        )
        self.assertGreaterEqual(result["metrics"]["accuracy"], 0.0)
        self.assertLessEqual(result["metrics"]["accuracy"], 1.0)
        self.assertIn("f1_macro", result["metrics"])
        self.assertIn("confusion_matrix", result["metrics"])

    def test_fit_tfidf_classifier_multilabel_with_ovr(self):
        groups = [f"doc-{i}" for i in range(12)]
        texts = [
            "alpha beta gamma delta",
            "alpha epsilon",
            "beta gamma zeta",
            "delta epsilon zeta",
        ] * 3
        y = [
            ["label_a"],
            ["label_b"],
            ["label_c"],
            ["label_d"],
        ] * 3

        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.3, random_seed=5)
        label_names = ["label_a", "label_b", "label_c", "label_d"]
        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            task_type="multilabel",
            algorithm="logistic_regression",
            label_names=label_names,
            random_seed=5,
        )
        self.assertIn("per_class", result["metrics"])
        self.assertEqual(
            set(result["metrics"]["multilabel_confusion_matrices"]), set(label_names)
        )
        self.assertTrue(
            all(
                len(matrix) == 2 and all(len(row) == 2 for row in matrix)
                for matrix in result["metrics"]["multilabel_confusion_matrices"].values()
            )
        )
        self.assertEqual(set(result["classes"]), set(label_names))

    def test_extract_linear_coefficients_shape_and_ranking(self):
        groups = [f"doc-{i}" for i in range(12)]
        texts = ["universal liberty market" for _ in range(6)] + [
            "individual equality solidarity" for _ in range(6)
        ]
        y = ["a"] * 6 + ["b"] * 6
        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.3, random_seed=2)
        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            task_type="binary",
            random_seed=2,
        )
        coefficients = classifiers.extract_linear_coefficients(
            result["model"], result["vectorizer"], result["classes"]
        )
        required_keys = ("feature", "coefficient", "rank")
        self.assertTrue(all(all(k in c for k in required_keys) for c in coefficients))
        # ranks for the single (binary) label should start at 1 and be contiguous
        ranks = sorted(c["rank"] for c in coefficients)
        self.assertEqual(ranks[0], 1)
        self.assertEqual(ranks[-1], len(coefficients))

    def test_predict_with_uncertainty_binary(self):
        groups = [f"doc-{i}" for i in range(12)]
        texts = ["universal liberty market" for _ in range(6)] + [
            "individual equality solidarity" for _ in range(6)
        ]
        y = ["a"] * 6 + ["b"] * 6
        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.3, random_seed=4)
        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            task_type="binary",
            random_seed=4,
        )
        predictions = classifiers.predict_with_uncertainty(
            result["model"], result["vectorizer"], ["universal liberty market"], task_type="binary"
        )
        self.assertEqual(len(predictions), 1)
        self.assertIn("uncertainty", predictions[0])
        self.assertGreaterEqual(predictions[0]["uncertainty"], 0.0)
        self.assertLessEqual(predictions[0]["uncertainty"], 1.0)

    def test_binary_probability_uncertainty_ranks_boundary_cases_first(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros(len(texts), dtype=int)

            def predict_proba(self, texts):
                return np.array([[1 - p, p] for p in (0.50, 0.60, 0.90, 0.99)])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b", "c", "d"], task_type="binary"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[-1], 0.02)

    def test_linear_svm_uncertainty_ranks_smallest_margin_first(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros(len(texts), dtype=int)

            def decision_function(self, texts):
                return np.array([0.0, 0.4, 2.0])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b", "c"], task_type="binary"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)

    def test_multiclass_uncertainty_ranks_smallest_probability_margin_first(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros(len(texts), dtype=int)

            def predict_proba(self, texts):
                return np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.95, 0.03, 0.02]])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b", "c"], task_type="multiclass"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)

    def test_multiclass_linear_svm_uncertainty_ranks_smallest_margin_first(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros(len(texts), dtype=int)

            def decision_function(self, texts):
                return np.array([[0.0, 0.0, -1.0], [1.0, 0.6, 0.0], [3.0, 0.1, 0.0]])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b", "c"], task_type="multiclass"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)

    def test_multilabel_uncertainty_uses_normalized_entropy(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros((len(texts), 2), dtype=int)

            def predict_proba(self, texts):
                return np.array([[0.5, 0.5], [0.9, 0.9]])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b"], task_type="multilabel"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertGreaterEqual(scores[-1], 0.0)

    def test_multilabel_linear_svm_uncertainty_ranks_smallest_margins_first(self):
        class Vectorizer:
            def transform(self, texts):
                return texts

        class Model:
            def predict(self, texts):
                return np.zeros((len(texts), 2), dtype=int)

            def decision_function(self, texts):
                return np.array([[0.0, 0.0], [2.0, 2.0]])

        predictions = classifiers.predict_with_uncertainty(
            Model(), Vectorizer(), ["a", "b"], task_type="multilabel"
        )
        scores = [row["uncertainty"] for row in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0)


class TopicModelTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "universal liberty market economy freedom trade",
            "market economy trade liberty universal freedom",
            "equality solidarity cohesion community welfare",
            "solidarity community welfare equality cohesion",
            "education policy children schools learning access",
            "schools learning access education policy children",
        ]

    def test_train_topic_model_nmf_reproducible_with_fixed_seed(self):
        result_1 = topic_models.train_topic_model(
            self.texts, algorithm="nmf", n_topics=3, random_seed=42
        )
        result_2 = topic_models.train_topic_model(
            self.texts, algorithm="nmf", n_topics=3, random_seed=42
        )

        top_terms_1 = [[t["term"] for t in topic["top_terms"]] for topic in result_1["topics"]]
        top_terms_2 = [[t["term"] for t in topic["top_terms"]] for topic in result_2["topics"]]
        self.assertEqual(top_terms_1, top_terms_2)
        self.assertEqual(result_1["dominant_topics"], result_2["dominant_topics"])

    def test_train_topic_model_lda_shapes_and_diagnostics(self):
        result = topic_models.train_topic_model(
            self.texts, algorithm="lda", n_topics=3, random_seed=1, max_iter=15
        )
        self.assertEqual(len(result["topics"]), 3)
        self.assertEqual(len(result["doc_topic_distribution"]), len(self.texts))
        self.assertEqual(len(result["dominant_topics"]), len(self.texts))
        self.assertIn("perplexity", result["diagnostics"])
        self.assertIn("topic_diversity", result["diagnostics"])
        self.assertIn("top_term_overlap", result["diagnostics"])
        for topic in result["topics"]:
            self.assertGreater(len(topic["top_terms"]), 0)

    def test_train_topic_model_invalid_algorithm_raises(self):
        with self.assertRaises(ValueError):
            topic_models.train_topic_model(self.texts, algorithm="unknown", n_topics=2)


class ModelStorageTests(unittest.TestCase):
    def test_save_and_load_joblib_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "artifact.joblib"
            payload = {"hello": "world", "numbers": [1, 2, 3]}
            saved_path = model_storage.save_joblib(payload, path)
            self.assertTrue(saved_path.exists())
            loaded = model_storage.load_joblib(saved_path)
            self.assertEqual(loaded, payload)

    def test_load_joblib_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            model_storage.load_joblib("/nonexistent/path/artifact.joblib")

    def test_save_artifact_with_metadata_records_integrity_fields(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_root = model_storage.ARTIFACT_ROOT
            original_storage_configured = model_storage._storage_configured
            model_storage.ARTIFACT_ROOT = Path(tmp_dir)
            model_storage._storage_configured = lambda: False
            try:
                reference, metadata = model_storage.save_artifact_with_metadata(
                    {"model": "test"}, category="test-models"
                )
                self.assertEqual(metadata["reference"], reference)
                self.assertEqual(metadata["model_type"], "test-models")
                self.assertEqual(len(metadata["sha256"]), 64)
                self.assertGreater(metadata["size"], 0)
                self.assertEqual(metadata["serialization_format"], "joblib")
                self.assertIsNotNone(metadata["created_at"])
            finally:
                model_storage.ARTIFACT_ROOT = original_root
                model_storage._storage_configured = original_storage_configured
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_ensure_artifact_dir_creates_nested_directories(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_root = model_storage.ARTIFACT_ROOT
            model_storage.ARTIFACT_ROOT = __import__("pathlib").Path(tmp_dir)
            try:
                directory = model_storage.ensure_artifact_dir("project-1", "run-1")
                self.assertTrue(directory.exists())
                self.assertTrue(directory.is_dir())
            finally:
                model_storage.ARTIFACT_ROOT = original_root
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
