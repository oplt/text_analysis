"""Tests for the §27-30 classifier upgrades:

- §27: user/config-driven task type inference (never silently forced to
  multilabel).
- §28: new classical baselines (multinomial_nb, complement_nb,
  sgd_classifier) alongside the existing logistic_regression/linear_svm.
- §29: configurable `FeatureConfig` (count/tfidf, word and/or char n-grams,
  combined via FeatureUnion).
- §30: grouped train/validation/test split with no group leakage, and a
  graceful validation-skip path when too few groups remain.

Plain ``unittest.TestCase`` tests (no DB/FastAPI dependency), consistent
with ``test_infrastructure.py``.
"""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import classifiers


class InferTaskTypeTests(unittest.TestCase):
    def test_single_label_two_classes_infers_binary(self):
        y = [["a"], ["b"], ["a"], ["b"]]
        self.assertEqual(classifiers.infer_task_type(y), "binary")

    def test_single_label_many_classes_infers_multiclass(self):
        y = [["a"], ["b"], ["c"], ["a"], ["b"], ["c"]]
        self.assertEqual(classifiers.infer_task_type(y), "multiclass")

    def test_any_example_with_two_active_labels_infers_multilabel(self):
        y = [["a"], ["a", "b"], ["b"]]
        self.assertEqual(classifiers.infer_task_type(y), "multilabel")

    def test_any_example_with_zero_active_labels_infers_multilabel(self):
        y = [["a"], [], ["b"]]
        self.assertEqual(classifiers.infer_task_type(y), "multilabel")

    def test_explicit_request_is_always_honored_verbatim(self):
        # Even though the label shape looks single-label, an explicit
        # request must never be silently overridden (§27).
        y = [["a"], ["b"], ["a"]]
        self.assertEqual(classifiers.infer_task_type(y, requested="multilabel"), "multilabel")

    def test_invalid_explicit_request_raises(self):
        with self.assertRaises(ValueError):
            classifiers.infer_task_type([["a"]], requested="not_a_real_task_type")

    def test_scalar_labels_are_treated_as_single_label(self):
        # Already-flattened scalar labels (not wrapped in a list) are valid
        # input too.
        y = ["a", "b", "a", "b"]
        self.assertEqual(classifiers.infer_task_type(y), "binary")


class FlattenSingleLabelTargetsTests(unittest.TestCase):
    def test_flattens_exactly_one_label_per_example(self):
        y = [["a"], ["b"], ["c"]]
        self.assertEqual(classifiers.flatten_single_label_targets(y), ["a", "b", "c"])

    def test_passthrough_for_already_scalar_labels(self):
        y = ["a", "b", "c"]
        self.assertEqual(classifiers.flatten_single_label_targets(y), ["a", "b", "c"])

    def test_raises_on_ambiguous_multi_label_example(self):
        with self.assertRaises(ValueError):
            classifiers.flatten_single_label_targets([["a", "b"]])

    def test_raises_on_ambiguous_zero_label_example(self):
        with self.assertRaises(ValueError):
            classifiers.flatten_single_label_targets([[]])


class FeatureConfigTests(unittest.TestCase):
    def test_defaults_are_word_tfidf_unigrams(self):
        fc = classifiers.FeatureConfig()
        self.assertEqual(fc.vectorizer, "tfidf")
        self.assertTrue(fc.use_word_ngrams)
        self.assertFalse(fc.use_char_ngrams)

    def test_from_dict_ignores_unknown_keys_and_fills_defaults(self):
        fc = classifiers.FeatureConfig.from_dict({"vectorizer": "count", "bogus_key": 123})
        self.assertEqual(fc.vectorizer, "count")
        self.assertEqual(fc.ngram_max, 1)

    def test_from_dict_passthrough_for_existing_instance(self):
        original = classifiers.FeatureConfig(vectorizer="count")
        self.assertIs(classifiers.FeatureConfig.from_dict(original), original)

    def test_invalid_vectorizer_type_raises(self):
        with self.assertRaises(ValueError):
            classifiers.FeatureConfig(vectorizer="word2vec")

    def test_disabling_both_feature_families_raises(self):
        with self.assertRaises(ValueError):
            classifiers.FeatureConfig(use_word_ngrams=False, use_char_ngrams=False)

    def test_inverted_word_ngram_range_raises(self):
        with self.assertRaises(ValueError):
            classifiers.FeatureConfig(ngram_min=3, ngram_max=1)

    def test_inverted_char_ngram_range_raises(self):
        with self.assertRaises(ValueError):
            classifiers.FeatureConfig(use_char_ngrams=True, char_ngram_min=5, char_ngram_max=2)

    def test_build_feature_extractor_word_only_returns_single_vectorizer(self):
        fc = classifiers.FeatureConfig(vectorizer="tfidf")
        extractor = classifiers.build_feature_extractor(fc)
        matrix = extractor.fit_transform(["alpha beta", "gamma delta"])
        self.assertEqual(matrix.shape[0], 2)

    def test_build_feature_extractor_char_only_returns_single_vectorizer(self):
        fc = classifiers.FeatureConfig(use_word_ngrams=False, use_char_ngrams=True)
        extractor = classifiers.build_feature_extractor(fc)
        matrix = extractor.fit_transform(["alpha beta", "gamma delta"])
        self.assertEqual(matrix.shape[0], 2)
        self.assertGreater(len(extractor.get_feature_names_out()), 0)

    def test_build_feature_extractor_combines_word_and_char_via_feature_union(self):
        fc = classifiers.FeatureConfig(use_word_ngrams=True, use_char_ngrams=True)
        extractor = classifiers.build_feature_extractor(fc)
        matrix = extractor.fit_transform(["alpha beta gamma", "delta epsilon zeta"])
        self.assertEqual(matrix.shape[0], 2)
        feature_names = list(extractor.get_feature_names_out())
        self.assertTrue(any(name.startswith("word__") for name in feature_names))
        self.assertTrue(any(name.startswith("char__") for name in feature_names))


class GroupedTrainValTestSplitTests(unittest.TestCase):
    def test_no_group_appears_in_more_than_one_partition(self):
        texts = [f"document text number {i}" for i in range(40)]
        y = ["yes" if i % 2 == 0 else "no" for i in range(40)]
        groups = [f"doc-{i // 4}" for i in range(40)]

        split = classifiers.grouped_train_val_test_split(
            texts, y, groups, test_size=0.2, val_size=0.2, random_seed=7
        )

        train_groups = set(split["groups_train"])
        val_groups = set(split["groups_val"])
        test_groups = set(split["groups_test"])
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        total = len(split["X_train"]) + len(split["X_val"]) + len(split["X_test"])
        self.assertEqual(total, 40)
        self.assertEqual(len(split["y_train"]), len(split["X_train"]))
        self.assertEqual(split["notes"], [])

    def test_val_size_zero_skips_validation_without_a_note(self):
        texts = [f"document text number {i}" for i in range(20)]
        y = ["yes" if i % 2 == 0 else "no" for i in range(20)]
        groups = [f"doc-{i // 4}" for i in range(20)]

        split = classifiers.grouped_train_val_test_split(
            texts, y, groups, test_size=0.2, val_size=0.0, random_seed=3
        )
        self.assertEqual(split["X_val"], [])
        self.assertEqual(split["notes"], [])

    def test_too_few_groups_after_test_split_skips_validation_with_note(self):
        # Only 2 distinct groups total: after holding out one for test,
        # a single remaining group cannot itself be split train/val.
        texts = ["alpha beta", "gamma delta"]
        y = ["a", "b"]
        groups = ["doc-1", "doc-2"]

        split = classifiers.grouped_train_val_test_split(
            texts, y, groups, test_size=0.5, val_size=0.5, random_seed=1
        )
        self.assertEqual(split["X_val"], [])
        self.assertTrue(len(split["notes"]) >= 1)
        # The split must not have raised/broken despite the tiny group count.
        self.assertEqual(len(split["X_train"]) + len(split["X_test"]), 2)

    def test_requires_at_least_two_distinct_groups(self):
        with self.assertRaises(ValueError):
            classifiers.grouped_train_val_test_split(
                ["a", "b"], ["x", "y"], ["only-one", "only-one"]
            )


class NewBaselineAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.groups = [f"doc-{i}" for i in range(16)]
        self.texts = ["universal liberty market policy" for _ in range(8)] + [
            "individual equality solidarity cohesion" for _ in range(8)
        ]
        self.y = ["liberal"] * 8 + ["universal"] * 8
        self.split = classifiers.grouped_train_test_split(
            self.texts, self.y, self.groups, test_size=0.25, random_seed=3
        )

    def test_multinomial_nb_end_to_end(self):
        result = classifiers.fit_tfidf_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="multinomial_nb",
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(result["algorithm"], "multinomial_nb")

    def test_complement_nb_end_to_end(self):
        result = classifiers.fit_tfidf_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="complement_nb",
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])

    def test_complement_nb_with_class_weight_uses_sample_weight_without_raising(self):
        result = classifiers.fit_tfidf_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="complement_nb",
            class_weight="balanced",
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])

    def test_sgd_classifier_log_loss_end_to_end(self):
        result = classifiers.fit_tfidf_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="sgd_classifier",
            sgd_loss="log_loss",
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        # log_loss supports predict_proba -> roc_auc should be computable.
        self.assertIn("roc_auc", result["metrics"])

    def test_sgd_classifier_hinge_end_to_end(self):
        result = classifiers.fit_tfidf_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="sgd_classifier",
            sgd_loss="hinge",
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        # hinge has no predict_proba -> no roc_auc, but must not raise.
        self.assertNotIn("roc_auc", result["metrics"])

    def test_multilabel_with_new_algorithm_still_works(self):
        y_ml = [["label_a"], ["label_b"]] * 8
        split = classifiers.grouped_train_test_split(
            self.texts, y_ml, self.groups, test_size=0.25, random_seed=3
        )
        result = classifiers.fit_tfidf_classifier(
            split["X_train"],
            split["y_train"],
            split["X_test"],
            split["y_test"],
            task_type="multilabel",
            algorithm="sgd_classifier",
            label_names=["label_a", "label_b"],
            random_seed=3,
        )
        self.assertIn("per_class", result["metrics"])

    def test_unsupported_algorithm_raises(self):
        with self.assertRaises(ValueError):
            classifiers.fit_tfidf_classifier(
                self.split["X_train"],
                self.split["y_train"],
                self.split["X_test"],
                self.split["y_test"],
                task_type="binary",
                algorithm="random_forest",
                random_seed=3,
            )


class FitTextClassifierTests(unittest.TestCase):
    def setUp(self):
        self.groups = [f"doc-{i}" for i in range(16)]
        self.texts = ["universal liberty market policy" for _ in range(8)] + [
            "individual equality solidarity cohesion" for _ in range(8)
        ]
        self.y = ["liberal"] * 8 + ["universal"] * 8
        self.split = classifiers.grouped_train_test_split(
            self.texts, self.y, self.groups, test_size=0.25, random_seed=3
        )

    def test_word_only_tfidf_matches_fit_tfidf_classifier_shape(self):
        result = classifiers.fit_text_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(vectorizer="tfidf"),
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        self.assertIn("feature_config", result)
        self.assertEqual(result["feature_config"]["vectorizer"], "tfidf")

    def test_char_ngrams_only(self):
        result = classifiers.fit_text_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(
                use_word_ngrams=False, use_char_ngrams=True, char_ngram_min=3, char_ngram_max=4
            ),
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        self.assertGreater(result["vocabulary_size"], 0)

    def test_word_and_char_ngrams_combined_via_feature_union(self):
        result = classifiers.fit_text_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(use_word_ngrams=True, use_char_ngrams=True),
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])
        feature_names = list(result["vectorizer"].get_feature_names_out())
        self.assertTrue(any(name.startswith("word__") for name in feature_names))
        self.assertTrue(any(name.startswith("char__") for name in feature_names))

    def test_count_vectorizer_variant(self):
        result = classifiers.fit_text_classifier(
            self.split["X_train"],
            self.split["y_train"],
            self.split["X_test"],
            self.split["y_test"],
            task_type="binary",
            algorithm="multinomial_nb",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            random_seed=3,
        )
        self.assertIn("accuracy", result["metrics"])

    def test_test_only_vocabulary_never_leaks_into_vectorizer(self):
        test_only_token = "zzzonlyintestzzz"
        x_test_marked = [f"{t} {test_only_token}" for t in self.split["X_test"]]
        result = classifiers.fit_text_classifier(
            self.split["X_train"],
            self.split["y_train"],
            x_test_marked,
            self.split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(use_word_ngrams=True, use_char_ngrams=True),
            random_seed=3,
        )
        feature_names = " ".join(result["vectorizer"].get_feature_names_out())
        self.assertNotIn(test_only_token, feature_names)


if __name__ == "__main__":
    unittest.main()
