"""Golden multilingual corpus regression tests (deterministic pipeline invariants)."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from backend.modules.text_research.infrastructure import classifiers
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig, tokenize
from backend.modules.text_research.infrastructure.quantitative import build_dfm

GOLDEN = Path(__file__).parent


def _load_corpus() -> dict:
    return json.loads((GOLDEN / "corpus.json").read_text())


class GoldenPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_corpus()
        cls.config = cls.payload["preprocessing_config"]
        cls.documents = cls.payload["documents"]
        cls.en_docs = [doc for doc in cls.documents if doc["language"] == "en"]
        cls.expected_tokens = json.loads((GOLDEN / "expected_tokens_en.json").read_text())

    def test_tokenization_matches_expected_for_en_subset(self) -> None:
        for doc in self.en_docs:
            actual = tokenize(doc["text"], self.config)
            self.assertEqual(actual, self.expected_tokens[doc["id"]])

    def test_dfm_dimensions_are_stable(self) -> None:
        texts = [doc["text"] for doc in self.documents]
        dfm = build_dfm(texts, self.config, weighting="count")
        expected = self.payload["dfm_dimensions"]
        actual_features = dfm["dimensions"].get("features") or len(dfm["feature_names"])
        self.assertEqual(dfm["dimensions"]["units"], expected["units"])
        self.assertEqual(actual_features, expected["features"])

    def test_prepare_texts_checksums_are_deterministic(self) -> None:
        en_texts = [doc["text"] for doc in self.en_docs]
        en_ids = [doc["id"] for doc in self.en_docs]
        config = PreprocessingConfig.from_dict(self.config)

        first = prepare_texts(en_texts, config, unit_ids=en_ids)
        second = prepare_texts(en_texts, config, unit_ids=en_ids)
        expected = self.payload["checksums"]

        self.assertEqual(first.pipeline_checksum, second.pipeline_checksum)
        self.assertEqual(first.corpus_checksum, second.corpus_checksum)
        self.assertEqual(first.pipeline_checksum, expected["pipeline_checksum"])
        self.assertEqual(first.corpus_checksum, expected["corpus_checksum"])

    def test_grouped_split_has_no_document_leakage(self) -> None:
        texts: list[str] = []
        labels: list[str] = []
        groups: list[str] = []
        for repeat in range(4):
            for index, doc in enumerate(self.documents):
                texts.append(f"{doc['text']} repeat {repeat}")
                labels.append("pos" if (index + repeat) % 2 == 0 else "neg")
                groups.append(doc["group"])

        split = classifiers.grouped_train_val_test_split(
            texts,
            labels,
            groups,
            test_size=0.2,
            val_size=0.2,
            random_seed=11,
        )

        train_groups = set(split["groups_train"])
        val_groups = set(split["groups_val"])
        test_groups = set(split["groups_test"])
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))

    def test_prediction_dict_cannot_mutate_gold_labels(self) -> None:
        gold_labels = {doc["id"]: "pos" for doc in self.en_docs}
        gold_copy = copy.deepcopy(gold_labels)
        prediction_sets = {
            "model-a": {
                doc_id: {"prediction": "neg", "probability": 0.91}
                for doc_id in gold_labels
            }
        }

        prediction_sets["model-a"]["en-1"]["prediction"] = "maybe"
        self.assertEqual(gold_labels, gold_copy)
        self.assertIsNot(gold_labels, prediction_sets["model-a"])
        self.assertNotEqual(
            prediction_sets["model-a"]["en-1"]["prediction"],
            gold_labels["en-1"],
        )


if __name__ == "__main__":
    unittest.main()
