"""Tests for hashing embeddings, embedding cache/classifier, and spaCy honesty."""

from __future__ import annotations

import unittest

import numpy as np

from backend.modules.text_research.infrastructure import embeddings
from backend.modules.text_research.infrastructure.embedding_classifier import fit_embedding_classifier
from backend.modules.text_research.infrastructure.spacy_engine import SpacyLinguisticEngine


class HashingEmbeddingTests(unittest.TestCase):
    def test_hashing_provider_is_default(self):
        provider = embeddings.get_default_embedding_provider()
        self.assertEqual(provider.name, "hashing")

    def test_hashing_embeddings_are_deterministic(self):
        provider = embeddings.HashingEmbeddingProvider(n_features=64)
        texts = ["education policy school", "market finance trade"]
        first = provider.embed_texts(texts)
        second = provider.embed_texts(texts)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(first[0]), 64)

    def test_embed_texts_cached_reuses_vectors(self):
        provider = embeddings.HashingEmbeddingProvider(n_features=32)
        cache: dict[str, list[float]] = {}
        texts = ["hello world", "hello world", "other text"]
        vectors = embeddings.embed_texts_cached(provider, texts, cache=cache)
        self.assertEqual(vectors[0], vectors[1])
        self.assertEqual(len(cache), 2)

    def test_unavailable_provider_raises(self):
        provider = embeddings.get_embedding_provider("unavailable")
        with self.assertRaises(ValueError):
            provider.embed_texts(["hello"])

    def test_sentence_transformers_fails_honestly_when_missing(self):
        if not embeddings.SentenceTransformerEmbeddingProvider.available():
            with self.assertRaises(ValueError) as ctx:
                embeddings.get_embedding_provider("sentence_transformers")
            self.assertIn("sentence_transformers", str(ctx.exception).lower())


class EmbeddingClassifierTests(unittest.TestCase):
    def test_fit_logistic_regression_on_hashing_embeddings(self):
        provider = embeddings.HashingEmbeddingProvider(n_features=32)
        texts = ["alpha beta gamma"] * 4 + ["delta epsilon zeta"] * 4
        y = ["a"] * 4 + ["b"] * 4
        X = np.asarray(provider.embed_texts(texts), dtype=float)
        result = fit_embedding_classifier(X, y, algorithm="logistic_regression", random_seed=0)
        self.assertEqual(result["algorithm"], "logistic_regression")
        self.assertEqual(result["n_samples"], 8)
        self.assertEqual(len(result["predictions"]), 8)


class SpacyEngineTests(unittest.TestCase):
    def test_unavailable_path_raises_honestly_when_spacy_missing(self):
        if SpacyLinguisticEngine.available():
            self.skipTest("spaCy is installed in this environment")
        engine = SpacyLinguisticEngine()
        with self.assertRaises(ValueError) as ctx:
            engine.analyze(["The cat sat on the mat."])
        message = str(ctx.exception).lower()
        self.assertTrue("spacy" in message or "unavailable" in message)
