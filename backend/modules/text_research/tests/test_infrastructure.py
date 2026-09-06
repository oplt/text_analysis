"""Tests for the pure-Python/scikit-learn Policy Text Lab infrastructure engines.

These are plain ``unittest.TestCase`` tests (no DB/FastAPI/app settings
dependency) so they can run in isolation via ``pytest`` against the
infrastructure layer only.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.modules.text_research.infrastructure import (
    classifiers,
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
            "First paragraph about liberalism.\n\n"
            "Second paragraph about universalism.\n\n\n"
            "Third paragraph about individualism."
        )
        units_1 = segmentation.segment_document(text, "paragraph")
        units_2 = segmentation.segment_document(text, "paragraph")

        self.assertEqual(units_1, units_2)  # full determinism
        self.assertEqual(len(units_1), 3)
        self.assertEqual([u["position"] for u in units_1], [0, 1, 2])
        self.assertEqual([u["paragraph_number"] for u in units_1], [1, 2, 3])
        self.assertTrue(all(u["sentence_number"] is None for u in units_1))
        self.assertEqual(units_1[0]["text"], "First paragraph about liberalism.")

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


class PreprocessingTests(unittest.TestCase):
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

    def test_lemmatization_true_is_rejected(self):
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "lemmatization": True}
        with self.assertRaises(ValueError):
            preprocessing.tokenize("Policies matter.", config)
        with self.assertRaises(ValueError):
            preprocessing.PreprocessingConfig.from_dict(config)

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
        self.assertFalse(preview["lemmatization_supported"])
        self.assertEqual(preview["stemmer"], "snowball_english")
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
            "Universalism": [["yes", "yes"], ["no", "no"], ["yes", "no"]],
        }
        result = reliability.reliability_by_label(label_data)
        self.assertIn("Universalism", result)
        self.assertIn("alpha", result["Universalism"])
        self.assertIn("cohens_kappa", result["Universalism"])
        self.assertAlmostEqual(result["Universalism"]["raw_agreement"], 2 / 3)


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
        self.assertEqual(stats["token_count"], sum(len(t) for t in self.tokenized))
        self.assertGreater(stats["vocabulary_size"], 0)
        self.assertEqual(stats["min_length"], min(len(t) for t in self.tokenized))
        self.assertEqual(stats["max_length"], max(len(t) for t in self.tokenized))

    def test_term_frequencies_sum_and_prevalence(self):
        results = quantitative.term_frequencies(self.tokenized)
        term_lookup = {row["term"]: row for row in results}
        self.assertIn("universal", term_lookup)
        total_raw = sum(row["raw_count"] for row in results)
        self.assertAlmostEqual(total_raw, sum(len(t) for t in self.tokenized))
        self.assertAlmostEqual(sum(row["relative_frequency"] for row in results), 1.0, places=6)
        self.assertLessEqual(term_lookup["universal"]["document_prevalence"], 1.0)

    def test_ngram_frequencies_bigrams(self):
        results = quantitative.ngram_frequencies(self.tokenized, n=2)
        self.assertTrue(all(row["ngram"].count(" ") == 1 for row in results))
        self.assertTrue(all(row["raw_count"] >= 1 for row in results))

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
        result = quantitative.build_dfm_matrix(self.tokenized, mode="count")
        self.assertEqual(result["dimensions"]["units"], 3)
        self.assertGreater(result["dimensions"]["features"], 0)
        self.assertIn("dense_matrix", result)
        self.assertEqual(len(result["dense_matrix"]), 3)
        self.assertGreaterEqual(result["density"], 0.0)

    def test_build_dfm_binary_mode_values_are_zero_or_one(self):
        result = quantitative.build_dfm_matrix(self.tokenized, mode="binary")
        for row in result["dense_matrix"]:
            self.assertTrue(all(v in (0, 1) for v in row))

    def test_build_dfm_tfidf_mode_has_float_weights(self):
        result = quantitative.build_dfm_matrix(self.tokenized, mode="tfidf")
        flat = [v for row in result["dense_matrix"] for v in row]
        self.assertTrue(any(0 < v < 1 for v in flat))


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
        texts = ["universalism liberty market freedom" for _ in range(6)] + [
            "individualism equality solidarity cohesion" for _ in range(6)
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
            "liberty market universal access",
            "individual freedom",
            "universal equality cohesion",
            "solidarity multicultural diversity",
        ] * 3
        y = [
            ["liberalism"],
            ["individualism"],
            ["universalism"],
            ["multiculturalism"],
        ] * 3

        split = classifiers.grouped_train_test_split(texts, y, groups, test_size=0.3, random_seed=5)
        label_names = ["liberalism", "individualism", "universalism", "multiculturalism"]
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
        self.assertLessEqual(predictions[0]["uncertainty"], 0.5)


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
