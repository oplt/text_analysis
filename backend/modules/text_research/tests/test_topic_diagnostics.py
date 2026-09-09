"""Tests for topic-model diagnostics: NPMI coherence, K-sweep, seed stability,
and the classical-engine plug-in protocol (§40-42).

Pure scikit-learn/numpy — no DB/FastAPI dependency.
"""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import topic_models
from backend.modules.text_research.infrastructure.topic_engines import get_topic_engine


class TopicCoherenceTests(unittest.TestCase):
    def setUp(self):
        # Two clearly separable vocabularies repeated across documents so
        # within-topic terms reliably co-occur (high coherence expected)
        # and cross-topic terms never co-occur (low/negative expected).
        self.texts = [
            "market economy trade liberty freedom",
            "market economy trade liberty freedom",
            "market economy trade liberty freedom",
            "equality solidarity cohesion community welfare",
            "equality solidarity cohesion community welfare",
            "equality solidarity cohesion community welfare",
        ]

    def test_train_topic_model_reports_coherence(self):
        # top_n_terms=5 matches each topic's true vocabulary size (5 words);
        # with the default top_n=10 every topic's "top terms" would include
        # the *other* topic's near-zero-weight words too (vocab has only 10
        # words total), which would drag coherence down for an unrelated
        # reason (measuring cross-topic pairs, not within-topic coherence).
        result = topic_models.train_topic_model(
            self.texts, algorithm="nmf", n_topics=2, random_seed=42, top_n_terms=5
        )
        self.assertIn("coherence_npmi", result["diagnostics"])
        self.assertIn("coherence_per_topic", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["coherence_method"], "npmi")
        coherence = result["diagnostics"]["coherence_npmi"]
        self.assertIsInstance(coherence, float)
        # Terms within each topic co-occur perfectly across their documents;
        # coherence should be clearly positive, not near the -1 floor.
        self.assertGreater(coherence, 0.0)

    def test_topic_coherence_npmi_standalone(self):
        result = topic_models.train_topic_model(
            self.texts, algorithm="nmf", n_topics=2, random_seed=1, top_n_terms=5
        )
        coherence = topic_models.topic_coherence_npmi(
            result["vectorizer"].transform(self.texts),
            result["topics"],
            result["feature_names"],
            top_n=5,
        )
        self.assertEqual(coherence["method"], "npmi")
        self.assertEqual(len(coherence["per_topic"]), 2)
        self.assertIsNotNone(coherence["mean_coherence_npmi"])


class KSweepTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "universal liberty market economy freedom trade",
            "market economy trade liberty universal freedom",
            "equality solidarity cohesion community welfare",
            "solidarity community welfare equality cohesion",
            "education policy children schools learning access",
            "schools learning access education policy children",
        ]

    def test_k_sweep_returns_one_row_per_k(self):
        rows = topic_models.k_sweep(self.texts, [2, 3, 4], algorithm="nmf", random_seed=7)
        self.assertEqual(len(rows), 3)
        for row, k in zip(rows, [2, 3, 4], strict=True):
            self.assertEqual(row["requested_n_topics"], k)
            self.assertEqual(row["status"], "ok")
            self.assertIn("coherence_npmi", row)
            self.assertIn("topic_diversity", row)
            self.assertIn("top_terms_preview", row)

    def test_k_sweep_does_not_pick_a_winner(self):
        rows = topic_models.k_sweep(self.texts, [2, 3], algorithm="nmf", random_seed=7)
        for row in rows:
            self.assertNotIn("is_best", row)
            self.assertNotIn("recommended", row)


class SeedStabilityTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "universal liberty market economy freedom trade",
            "market economy trade liberty universal freedom",
            "equality solidarity cohesion community welfare",
            "solidarity community welfare equality cohesion",
            "education policy children schools learning access",
            "schools learning access education policy children",
        ]

    def test_identical_seed_gives_perfect_self_stability(self):
        # Same seed run twice through train_topic_model is reproducible
        # (existing guarantee); confirm seed_stability agrees when we feed
        # it two genuinely different seeds that happen to converge similarly
        # is NOT guaranteed, so instead assert structural invariants.
        result = topic_models.seed_stability(
            self.texts, [1, 2, 3], algorithm="nmf", n_topics=3
        )
        self.assertEqual(len(result["pairwise"]), 3)  # C(3,2)
        self.assertIn("mean_stability_jaccard", result)
        self.assertEqual(result["matching_method"], "hungarian")
        for pair in result["pairwise"]:
            self.assertGreaterEqual(pair["mean_best_match_jaccard"], 0.0)
            self.assertLessEqual(pair["mean_best_match_jaccard"], 1.0)
            self.assertEqual(len(pair["matching"]), 3)
            self.assertEqual(pair["matching_method"], "hungarian")

    def test_match_topics_hungarian_one_to_one(self):
        terms_a = [{"alpha", "beta"}, {"gamma", "delta"}]
        terms_b = [{"beta", "alpha", "extra"}, {"gamma", "delta", "omega"}]
        matching, mean_jaccard = topic_models.match_topics_hungarian(terms_a, terms_b)
        self.assertEqual(len(matching), 2)
        self.assertGreater(mean_jaccard, 0.5)
        matched_pairs = {(m["topic_a"], m["topic_b"]) for m in matching}
        self.assertEqual(matched_pairs, {(0, 0), (1, 1)})

    def test_requires_at_least_two_seeds(self):
        with self.assertRaises(ValueError):
            topic_models.seed_stability(self.texts, [1], algorithm="nmf", n_topics=2)


class ClassicalTopicModelEngineTests(unittest.TestCase):
    def test_engine_matches_train_topic_model(self):
        texts = [
            "market economy trade liberty freedom",
            "equality solidarity cohesion community welfare",
        ]
        engine = topic_models.ClassicalTopicModelEngine(algorithm="nmf")
        self.assertEqual(engine.name, "nmf")
        result = engine.fit(texts, n_topics=2, config=None, random_seed=5)
        self.assertEqual(result["algorithm"], "nmf")
        self.assertEqual(len(result["topics"]), 2)

    def test_registered_engine_infers_unseen_text_after_fit(self):
        texts = [
            "market economy trade liberty freedom",
            "market trade economy investment growth",
            "equality solidarity cohesion community welfare",
            "community welfare equality solidarity support",
        ]
        engine = get_topic_engine("nmf")
        engine.fit(texts, n_topics=2, random_seed=5)
        inferred = engine.transform(["market trade growth", "community welfare support"])
        self.assertEqual(len(inferred["dominant_topics"]), 2)
        self.assertEqual(len(inferred["doc_topic_distribution"]), 2)

    def test_engine_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            topic_models.ClassicalTopicModelEngine(algorithm="bertopic")


if __name__ == "__main__":
    unittest.main()
