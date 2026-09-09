"""Tests for readability, clustering, embeddings stub, and NER honesty."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import (
    clustering,
    embeddings,
    linguistic_features,
    ner,
    readability,
)


class ReadabilityTests(unittest.TestCase):
    def test_flesch_on_simple_prose(self):
        text = "The cat sat on the mat. The dog ran in the park."
        metrics = readability.readability_metrics(text)
        self.assertGreater(metrics["n_words"], 5)
        self.assertGreaterEqual(metrics["n_sentences"], 2)
        self.assertIn("flesch_reading_ease", metrics)
        self.assertIn("flesch_kincaid_grade", metrics)


class ClusteringTests(unittest.TestCase):
    def test_kmeans_labels_and_top_terms(self):
        texts = [
            "education policy school learning curriculum teacher",
            "education school classroom learning student teacher",
            "market trade finance bank investment capital",
            "finance bank market stocks investment capital",
            "sports football soccer game team match",
            "sports basketball game team score match",
        ]
        unit_ids = [f"u{i}" for i in range(len(texts))]
        result = clustering.run_clustering(
            texts, unit_ids, n_clusters=3, algorithm="kmeans", random_seed=0, top_n_terms=5
        )
        self.assertEqual(len(result["labels"]), len(texts))
        self.assertEqual(result["n_clusters"], 3)
        self.assertIn("top_terms", result)
        self.assertIsInstance(result["note"], str)


class EmbeddingStubTests(unittest.TestCase):
    def test_default_provider_is_hashing_baseline(self):
        provider = embeddings.get_default_embedding_provider()
        vectors = provider.embed_texts(["hello world"])
        self.assertEqual(provider.name, "hashing")
        self.assertEqual(len(vectors), 1)

    def test_unavailable_provider_raises(self):
        provider = embeddings.get_embedding_provider("unavailable")
        with self.assertRaises(ValueError):
            provider.embed_texts(["hello"])


class NerHonestyTests(unittest.TestCase):
    def test_describe_does_not_fake_availability(self):
        caps = ner.describe_ner_capabilities()
        self.assertIn("available", caps)
        self.assertFalse(caps["required"])
        ling = linguistic_features.describe_linguistic_capabilities()
        self.assertIn("pos_available", ling)


if __name__ == "__main__":
    unittest.main()
