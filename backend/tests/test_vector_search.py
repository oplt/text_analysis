import unittest

from backend.lib.vector_search import (
    json_fallback_max_candidates,
    rank_embedding_matches,
)
from backend.lib.vectors import cosine_similarity


class VectorSearchHelpersTest(unittest.TestCase):
    def test_json_fallback_max_candidates_scales_with_top_k(self):
        self.assertEqual(json_fallback_max_candidates(5), 250)
        self.assertEqual(json_fallback_max_candidates(200), 5000)

    def test_rank_embedding_matches_returns_top_scoring_rows(self):
        rows = [
            {"id": "a", "embedding": [1.0, 0.0]},
            {"id": "b", "embedding": [0.0, 1.0]},
            {"id": "c", "embedding": [0.9, 0.1]},
        ]

        ranked = rank_embedding_matches(
            [1.0, 0.0],
            rows,
            top_k=2,
            score_threshold=0.1,
            build_match=lambda row, score: {"id": row["id"], "score": score},
        )

        self.assertEqual([item["id"] for item in ranked], ["a", "c"])
        self.assertAlmostEqual(ranked[0]["score"], 1.0)
        self.assertGreater(ranked[1]["score"], 0.8)

    def test_rank_embedding_matches_respects_threshold(self):
        rows = [{"id": "low", "embedding": [0.0, 1.0]}]
        ranked = rank_embedding_matches(
            [1.0, 0.0],
            rows,
            top_k=3,
            score_threshold=0.5,
            build_match=lambda row, score: {"id": row["id"], "score": score},
        )
        self.assertEqual(ranked, [])


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors_score_high(self):
        score = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
