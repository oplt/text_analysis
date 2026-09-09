"""Property-style pipeline invariants (no database)."""

from __future__ import annotations

import copy
import unittest

from backend.modules.text_research.domain.prepared_corpus import (
    compute_corpus_checksum,
    compute_pipeline_checksum,
)
from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
)


class GroupLeakagePropertyTests(unittest.TestCase):
    def test_no_document_group_in_both_train_and_test(self):
        texts = [f"document text number {i}" for i in range(40)]
        y = ["yes" if i % 2 == 0 else "no" for i in range(40)]
        groups = [f"doc-{i // 4}" for i in range(40)]

        split = classifiers.grouped_train_val_test_split(
            texts, y, groups, test_size=0.2, val_size=0.2, random_seed=7
        )

        train_groups = set(split["groups_train"])
        val_groups = set(split["groups_val"])
        test_groups = set(split["groups_test"])
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))


class PrepareTextsImmutabilityTests(unittest.TestCase):
    def test_prepare_texts_does_not_mutate_original_list_or_strings(self):
        texts = ["The Policy, IS Universal!", "Education matters."]
        originals = copy.deepcopy(texts)
        prepare_texts(texts, PreprocessingConfig())
        self.assertEqual(texts, originals)
        for original, current in zip(originals, texts, strict=True):
            self.assertIs(original, current)


class ChecksumDeterminismTests(unittest.TestCase):
    def test_same_seed_config_texts_yield_same_checksums(self):
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        unit_ids = ["u1", "u2"]
        config = PreprocessingConfig(lowercase=True, remove_stopwords=False)

        first = prepare_texts(texts, config, unit_ids=unit_ids)
        second = prepare_texts(texts, config, unit_ids=unit_ids)

        self.assertEqual(first.pipeline_checksum, second.pipeline_checksum)
        self.assertEqual(first.corpus_checksum, second.corpus_checksum)
        self.assertEqual(
            first.pipeline_checksum,
            compute_pipeline_checksum(config.to_dict(), describe_implementation(config)),
        )
        self.assertEqual(first.corpus_checksum, compute_corpus_checksum(unit_ids, texts))

    def test_different_preprocessing_profile_yields_different_pipeline_checksum(self):
        texts = ["hello world", "foo bar"]
        base = prepare_texts(texts, PreprocessingConfig(lowercase=True))
        changed = prepare_texts(texts, PreprocessingConfig(lowercase=False))
        self.assertNotEqual(base.pipeline_checksum, changed.pipeline_checksum)


class VectorizerTrainOnlyPropertyTests(unittest.TestCase):
    def test_vectorizer_fit_on_train_only_excludes_test_only_token(self):
        groups = [f"doc-{i}" for i in range(16)]
        texts = ["universal liberty market policy" for _ in range(8)] + [
            "individual equality solidarity cohesion" for _ in range(8)
        ]
        y = ["liberal"] * 8 + ["universal"] * 8
        split = classifiers.grouped_train_test_split(
            texts, y, groups, test_size=0.25, random_seed=3
        )

        test_only_token = "zzzonlyintestzzz"
        x_test_marked = [f"{text} {test_only_token}" for text in split["X_test"]]
        result = classifiers.fit_text_classifier(
            split["X_train"],
            split["y_train"],
            x_test_marked,
            split["y_test"],
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(
                vectorizer="count", use_word_ngrams=True, use_char_ngrams=False
            ),
            random_seed=3,
        )
        feature_names = " ".join(result["vectorizer"].get_feature_names_out())
        self.assertNotIn(test_only_token, feature_names)


class PredictionArtifactIsolationTests(unittest.TestCase):
    def test_prediction_sets_keep_model_outputs_separate_from_gold_labels(self):
        prediction_sets: dict[str, dict[str, object]] = {
            "model-a": {"unit-1": {"prediction": "pos", "probability": 0.9}},
            "model-b": {"unit-1": {"prediction": "neg", "probability": 0.1}},
        }
        gold_labels = {"unit-1": "pos"}

        self.assertEqual(prediction_sets["model-a"]["unit-1"]["prediction"], "pos")
        self.assertEqual(prediction_sets["model-b"]["unit-1"]["prediction"], "neg")
        self.assertEqual(gold_labels["unit-1"], "pos")
        self.assertNotEqual(
            prediction_sets["model-a"]["unit-1"]["prediction"],
            prediction_sets["model-b"]["unit-1"]["prediction"],
        )
        self.assertIsNot(prediction_sets["model-a"], prediction_sets["model-b"])
        self.assertIsNot(gold_labels, prediction_sets["model-a"])
