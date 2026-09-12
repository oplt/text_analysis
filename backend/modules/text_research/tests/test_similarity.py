"""Tests for text similarity (cosine-on-TFIDF, Jaccard, embedding_cosine)."""

from __future__ import annotations

import unittest
import unittest.mock

import numpy as np

from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure import similarity as sim
from backend.modules.text_research.infrastructure.preprocessing import tokenize


class NormalizationTests(unittest.TestCase):
    def test_normalize_similarity_method_aliases(self):
        self.assertEqual(sim.normalize_similarity_method("cosine"), "tfidf_cosine")
        self.assertEqual(sim.normalize_similarity_method("tfidf"), "tfidf_cosine")
        self.assertEqual(sim.normalize_similarity_method(None), "tfidf_cosine")
        self.assertEqual(sim.normalize_similarity_method("Jaccard"), "jaccard")
        self.assertEqual(sim.normalize_similarity_method("embedding"), "embedding_cosine")
        self.assertEqual(sim.normalize_similarity_method("semantic"), "embedding_cosine")

    def test_normalize_similarity_method_rejects_unknown(self):
        with self.assertRaises(ValueError):
            sim.normalize_similarity_method("levenshtein")

    def test_normalize_similarity_mode_aliases(self):
        self.assertEqual(sim.normalize_similarity_mode("document_to_document"), "pairwise")
        self.assertEqual(sim.normalize_similarity_mode("unit_to_unit"), "pairwise")
        self.assertEqual(sim.normalize_similarity_mode("query_to_document"), "query")
        self.assertEqual(sim.normalize_similarity_mode("centroid"), "group_centroid")
        self.assertEqual(sim.normalize_similarity_mode(None), "pairwise")

    def test_normalize_similarity_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            sim.normalize_similarity_mode("nonsense")


class JaccardSimilarityTests(unittest.TestCase):
    def test_identical_sets(self):
        self.assertEqual(sim.jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(sim.jaccard_similarity({"a"}, {"b"}), 0.0)

    def test_partial_overlap(self):
        score = sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        self.assertAlmostEqual(score, 2 / 4)

    def test_both_empty(self):
        self.assertEqual(sim.jaccard_similarity(set(), set()), 1.0)


class PairwiseSimilarityTests(unittest.TestCase):
    def setUp(self):
        # Two near-duplicate docs, one clearly different.
        self.tokenized = [
            tokenize("liberty market freedom liberty market economy"),
            tokenize("liberty market freedom liberty market economy growth"),
            tokenize("solidarity welfare equality solidarity community care"),
        ]
        self.ids = ["u1", "u2", "u3"]

    def test_tfidf_cosine_pairwise_ranks_similar_pair_highest(self):
        report = sim.pairwise_similarity(self.ids, method="tfidf_cosine", tokenized=self.tokenized)
        self.assertEqual(report["mode"], "pairwise")
        self.assertEqual(report["method"], "tfidf_cosine")
        self.assertEqual(report["item_count"], 3)
        self.assertEqual(report["pairs_tested"], 3)
        top = report["pairs"][0]
        self.assertEqual({top["source_id"], top["target_id"]}, {"u1", "u2"})
        self.assertGreater(top["score"], 0.5)

    def test_jaccard_pairwise(self):
        report = sim.pairwise_similarity(self.ids, method="jaccard", tokenized=self.tokenized)
        self.assertEqual(report["method"], "jaccard")
        top = report["pairs"][0]
        self.assertEqual({top["source_id"], top["target_id"]}, {"u1", "u2"})
        for pair in report["pairs"]:
            self.assertGreaterEqual(pair["score"], 0.0)
            self.assertLessEqual(pair["score"], 1.0)

    def test_top_k_and_min_score(self):
        report = sim.pairwise_similarity(
            self.ids, method="jaccard", tokenized=self.tokenized, top_k=1
        )
        self.assertEqual(report["pairs_returned"], 1)
        self.assertEqual(report["pairs_tested"], 3)

        high_bar = sim.pairwise_similarity(
            self.ids, method="jaccard", tokenized=self.tokenized, min_score=0.99
        )
        self.assertEqual(high_bar["pairs_returned"], 0)

    def test_blocked_topk_matches_dense_for_small_overflow(self):
        """Force blocked path by temporarily lowering the dense ceiling."""
        n = 12
        tokenized = [tokenize(f"topic alpha beta {i % 3}") for i in range(n)]
        ids = [f"u{i}" for i in range(n)]
        dense = sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=5)
        with (
            unittest.mock.patch.object(sim, "DENSE_PAIRWISE_MAX_N", 4),
            unittest.mock.patch.object(sim, "EXACT_ALL_PAIRS_MAX_N", 4),
        ):
            blocked = sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=5)
        self.assertEqual(blocked["computation"], "bounded_topk")
        self.assertEqual(
            [(p["source_id"], p["target_id"], round(p["score"], 8)) for p in dense["pairs"]],
            [(p["source_id"], p["target_id"], round(p["score"], 8)) for p in blocked["pairs"]],
        )

    def test_large_topk_never_allocates_dense_nxn(self):
        n = sim.DENSE_PAIRWISE_MAX_N + 20
        tokenized = [["tok", f"v{i % 7}"] for i in range(n)]
        ids = [f"id-{i}" for i in range(n)]
        allocated: list[tuple[int, ...]] = []
        real_ones = np.ones

        def tracking_ones(shape, *args, **kwargs):
            allocated.append(tuple(shape) if isinstance(shape, tuple) else (shape,))
            return real_ones(shape, *args, **kwargs)

        with unittest.mock.patch.object(np, "ones", side_effect=tracking_ones):
            report = sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=10)
        self.assertEqual(report["computation"], "bounded_topk")
        self.assertEqual(report["pairs_returned"], 10)
        self.assertFalse(
            any(len(shape) == 2 and shape[0] == n and shape[1] == n for shape in allocated)
        )

    def test_exact_mode_ceiling_without_topk(self):
        n = sim.EXACT_ALL_PAIRS_MAX_N + 1
        tokenized = [["a"] for _ in range(n)]
        ids = [f"u{i}" for i in range(n)]
        with self.assertRaises(ValueError) as ctx:
            sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=None)
        self.assertIn("top_k", str(ctx.exception))

    def test_deterministic_tie_breaking(self):
        tokenized = [["shared"], ["shared"], ["shared"]]
        ids = ["c", "a", "b"]
        with unittest.mock.patch.object(sim, "DENSE_PAIRWISE_MAX_N", 1):
            report = sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=3)
        # Ties break by ascending index order (i, j), not lexicographic ids.
        pairs = [(p["source_id"], p["target_id"]) for p in report["pairs"]]
        self.assertEqual(pairs, [("c", "a"), ("c", "b"), ("a", "b")])
        again = sim.pairwise_similarity(ids, method="jaccard", tokenized=tokenized, top_k=3)
        self.assertEqual(pairs, [(p["source_id"], p["target_id"]) for p in again["pairs"]])

    def test_requires_at_least_two_items(self):
        with self.assertRaises(ValueError):
            sim.pairwise_similarity(["only-one"], method="jaccard", tokenized=[["a"]])

    def test_embedding_cosine_without_vectors_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            sim.pairwise_similarity(self.ids, method="embedding_cosine", tokenized=self.tokenized)
        message = str(ctx.exception).lower()
        self.assertIn("embedding", message)
        self.assertIn("tfidf_cosine", message.replace(" ", "_") + message)

    def test_embedding_cosine_with_vectors(self):
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        report = sim.pairwise_similarity(self.ids, method="embedding_cosine", embeddings=vectors)
        self.assertEqual(report["method"], "embedding_cosine")
        top = report["pairs"][0]
        self.assertEqual({top["source_id"], top["target_id"]}, {"u1", "u2"})


class QuerySimilarityTests(unittest.TestCase):
    def setUp(self):
        self.tokenized = [
            tokenize("liberty market freedom economy"),
            tokenize("solidarity welfare equality community"),
        ]
        self.ids = ["doc_a", "doc_b"]

    def test_tfidf_query_ranks_closest_document_first(self):
        query_tokens = tokenize("liberty market economy growth")
        report = sim.query_similarity(
            "q1",
            self.ids,
            method="tfidf_cosine",
            query_tokens=query_tokens,
            tokenized=self.tokenized,
        )
        self.assertEqual(report["mode"], "query")
        self.assertEqual(report["pairs"][0]["target_id"], "doc_a")

    def test_jaccard_query(self):
        query_tokens = tokenize("solidarity equality")
        report = sim.query_similarity(
            "q2", self.ids, method="jaccard", query_tokens=query_tokens, tokenized=self.tokenized
        )
        self.assertEqual(report["pairs"][0]["target_id"], "doc_b")

    def test_embedding_query_requires_vectors(self):
        with self.assertRaises(ValueError):
            sim.query_similarity("q3", self.ids, method="embedding_cosine")


class GroupCentroidSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["u1", "u2", "u3", "u4"]
        self.tokenized = [
            tokenize("liberty market freedom economy"),
            tokenize("liberty market growth economy"),
            tokenize("solidarity welfare equality community"),
            tokenize("solidarity welfare care community"),
        ]
        self.group_keys = ["A", "A", "B", "B"]

    def test_between_groups_tfidf(self):
        report = sim.group_centroid_similarity(
            self.ids,
            self.group_keys,
            method="tfidf_cosine",
            tokenized=self.tokenized,
            target="between_groups",
        )
        self.assertEqual(report["mode"], "group_centroid")
        self.assertEqual(report["target"], "between_groups")
        self.assertEqual(set(report["groups"]), {"A", "B"})
        self.assertEqual(len(report["pairs"]), 1)
        pair = report["pairs"][0]
        self.assertEqual({pair["source_id"], pair["target_id"]}, {"A", "B"})

    def test_item_to_own_group_jaccard(self):
        report = sim.group_centroid_similarity(
            self.ids,
            self.group_keys,
            method="jaccard",
            tokenized=self.tokenized,
            target="item_to_own_group",
        )
        self.assertEqual(report["target"], "item_to_own_group")
        self.assertEqual(len(report["items"]), 4)
        by_item = {row["item_id"]: row for row in report["items"]}
        self.assertEqual(by_item["u1"]["group"], "A")
        for row in report["items"]:
            self.assertGreaterEqual(row["score"], 0.0)

    def test_requires_matching_lengths(self):
        with self.assertRaises(ValueError):
            sim.group_centroid_similarity(
                self.ids, ["A", "B"], method="jaccard", tokenized=self.tokenized
            )

    def test_between_groups_requires_two_groups(self):
        with self.assertRaises(ValueError):
            sim.group_centroid_similarity(
                self.ids, ["A", "A", "A", "A"], method="jaccard", tokenized=self.tokenized
            )


class DescribeCapabilitiesTests(unittest.TestCase):
    def test_no_hardcoded_research_concepts(self):
        caps = sim.describe_similarity_capabilities()
        blob = str(caps).lower()
        self.assertNotIn("cultural_sphere", blob)
        self.assertIn("tfidf_cosine", caps["methods"])
        self.assertIn("jaccard", caps["methods"])
        self.assertIn("embedding_cosine", caps["methods"])


class QuantitativeWrapperTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "liberty market freedom liberty market economy",
            "liberty market freedom liberty market economy growth",
            "solidarity welfare equality solidarity community care",
        ]
        self.ids = ["u1", "u2", "u3"]

    def test_similarity_for_texts_pairwise(self):
        report = quantitative.similarity_for_texts(
            self.texts, self.ids, config=None, method="tfidf_cosine", mode="pairwise"
        )
        self.assertEqual(report["mode"], "pairwise")
        self.assertTrue(report["pairs"])

    def test_similarity_for_texts_query_mode(self):
        report = quantitative.similarity_for_texts(
            self.texts,
            self.ids,
            config=None,
            method="jaccard",
            mode="query",
            query_text="solidarity community",
            query_id="q1",
        )
        self.assertEqual(report["mode"], "query")
        self.assertEqual(report["pairs"][0]["target_id"], "u3")

    def test_similarity_for_texts_group_centroid_mode(self):
        report = quantitative.similarity_for_texts(
            self.texts,
            self.ids,
            config=None,
            method="jaccard",
            mode="group_centroid",
            group_keys=["A", "A", "B"],
        )
        self.assertEqual(report["mode"], "group_centroid")

    def test_similarity_for_texts_embedding_without_vectors_raises(self):
        with self.assertRaises(ValueError):
            quantitative.similarity_for_texts(
                self.texts, self.ids, config=None, method="embedding_cosine", mode="pairwise"
            )

    def test_similarity_for_texts_embedding_with_vectors(self):
        embeddings = {"u1": [1.0, 0.0], "u2": [0.95, 0.05], "u3": [0.0, 1.0]}
        report = quantitative.similarity_for_texts(
            self.texts,
            self.ids,
            config=None,
            method="embedding_cosine",
            mode="pairwise",
            embeddings=embeddings,
        )
        self.assertEqual(report["method"], "embedding_cosine")
        top = report["pairs"][0]
        self.assertEqual({top["source_id"], top["target_id"]}, {"u1", "u2"})


if __name__ == "__main__":
    unittest.main()
