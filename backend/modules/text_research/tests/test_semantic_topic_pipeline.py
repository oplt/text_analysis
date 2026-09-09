"""Tests for decomposed semantic topic stack."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.semantic_topic_pipeline import (
    fit_semantic_topics,
)
from backend.modules.text_research.infrastructure.topic_engines import get_topic_engine


class SemanticTopicPipelineTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "market economy trade policy reform",
            "government parliament legislation vote",
            "trade export import tariff agreement",
            "economy growth inflation employment data",
            "policy reform legislation parliament debate",
            "market stock finance investment banking",
        ]

    def test_fit_semantic_topics_returns_topics_and_labels(self):
        result = fit_semantic_topics(self.texts, n_topics=2, random_seed=7)
        self.assertEqual(result["engine"], "semantic_stack")
        self.assertEqual(len(result["labels"]), len(self.texts))
        self.assertIn("embedding", result["components"])
        self.assertIn("reducer", result["components"])
        self.assertIn("clusterer", result["components"])
        self.assertTrue(result["topics"])

    def test_semantic_stack_registered_as_topic_engine(self):
        engine = get_topic_engine("semantic_stack")
        self.assertEqual(engine.name, "semantic_stack")
        fitted = engine.fit(self.texts, n_topics=2)
        self.assertIn("topics", fitted)
        self.assertIn("labels", fitted)
        inferred = engine.transform(["tariff trade agreement", "parliamentary vote"])
        self.assertEqual(inferred["engine"], "semantic_stack")
        self.assertEqual(len(inferred["labels"]), 2)

    def test_semantic_stack_rejects_inference_before_fit(self):
        engine = get_topic_engine("semantic_stack")
        with self.assertRaisesRegex(ValueError, "fitted"):
            engine.transform(["tariff trade agreement"])


if __name__ == "__main__":
    unittest.main()
