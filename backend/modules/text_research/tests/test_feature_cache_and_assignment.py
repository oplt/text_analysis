"""Feature cache + assignment planning stress-oriented unit tests."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.assignment_planning import (
    assignment_pairs,
    plan_annotation_assignment,
)
from backend.modules.text_research.infrastructure import feature_cache, preprocessing


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        feature_cache.clear_cache()

    def test_tokenize_cache_hits(self):
        texts = ["Students should not be excluded from policy."]
        unit_ids = ["u1"]
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG, "remove_stopwords": True}
        first = feature_cache.get_or_tokenize(
            corpus_id="c1",
            unit_type="paragraph",
            unit_ids=unit_ids,
            texts=texts,
            config=config,
        )
        second = feature_cache.get_or_tokenize(
            corpus_id="c1",
            unit_type="paragraph",
            unit_ids=unit_ids,
            texts=texts,
            config=config,
        )
        self.assertEqual(first, second)
        self.assertIn("not", first[0])

    def test_cache_key_changes_when_persisted_text_hash_changes(self):
        config = {**preprocessing.DEFAULT_PREPROCESSING_CONFIG}
        first = feature_cache.build_cache_key(
            corpus_id="c1",
            unit_type="paragraph",
            unit_ids=["u1"],
            unit_hashes=["text-hash-a"],
            config=config,
            mode="tokens",
        )
        changed = feature_cache.build_cache_key(
            corpus_id="c1",
            unit_type="paragraph",
            unit_ids=["u1"],
            unit_hashes=["text-hash-b"],
            config=config,
            mode="tokens",
        )
        self.assertNotEqual(first, changed)


class AssignmentPlanningStressTests(unittest.TestCase):
    def test_large_shared_assignment_pair_count(self):
        unit_ids = [f"unit-{i}" for i in range(500)]
        annotators = [f"ann-{i}" for i in range(10)]
        plan = plan_annotation_assignment(
            unit_ids,
            annotators,
            sample_size=500,
            strategy="shared",
        )
        pairs = assignment_pairs(plan)
        self.assertEqual(len(pairs), 5000)
        self.assertEqual(len({(u, a) for u, a in pairs}), 5000)

    def test_large_overlap_assignment_is_linear_in_units(self):
        unit_ids = [f"unit-{i}" for i in range(500)]
        annotators = [f"ann-{i}" for i in range(10)]
        plan = plan_annotation_assignment(
            unit_ids,
            annotators,
            sample_size=500,
            strategy="overlap",
            overlap_count=50,
        )
        pairs = assignment_pairs(plan)
        # 50 shared * 10 annotators + 450 remainder = 500 + 450 = 950
        self.assertEqual(len(pairs), 950)


if __name__ == "__main__":
    unittest.main()
