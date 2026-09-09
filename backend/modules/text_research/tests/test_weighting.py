"""Tests for feature weighting schemes (separate from tokenization)."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import sparse

from backend.modules.text_research.infrastructure import quantitative, weighting
from backend.modules.text_research.infrastructure.weighting import (
    WeightingScheme,
    apply_weighting,
    describe_weighting,
    list_weighting_schemes,
    resolve_weighting_scheme,
)


class ResolveWeightingTests(unittest.TestCase):
    def test_aliases_and_required_schemes(self):
        self.assertEqual(resolve_weighting_scheme("term_frequency").name, "tf")
        self.assertEqual(resolve_weighting_scheme("tf-idf").name, "tfidf")
        self.assertEqual(resolve_weighting_scheme("logcount").name, "log_count")
        for name in ("count", "binary", "tf", "tfidf", "sublinear_tf", "log_count", "bm25"):
            self.assertEqual(resolve_weighting_scheme(name).name, name)

    def test_bm25_params_override(self):
        scheme = resolve_weighting_scheme("bm25", k1=2.0, b=0.5)
        self.assertEqual(scheme.k1, 2.0)
        self.assertEqual(scheme.b, 0.5)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_weighting_scheme("not-a-scheme")

    def test_catalog_lists_optional_schemes(self):
        names = {row["name"] for row in list_weighting_schemes()}
        self.assertTrue({"count", "binary", "tf", "tfidf", "log_count", "bm25"} <= names)


class ApplyWeightingTests(unittest.TestCase):
    def setUp(self):
        # 3 docs × 2 features: [[2, 0], [1, 1], [0, 3]]
        self.counts = sparse.csr_matrix(np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]]))

    def test_count_identity(self):
        out = apply_weighting(self.counts, "count")
        np.testing.assert_allclose(out.toarray(), self.counts.toarray())

    def test_binary(self):
        out = apply_weighting(self.counts, "binary").toarray()
        np.testing.assert_array_equal(out, [[1, 0], [1, 1], [0, 1]])

    def test_log_count(self):
        out = apply_weighting(self.counts, "log_count").toarray()
        expected = np.log1p(self.counts.toarray())
        np.testing.assert_allclose(out, expected)

    def test_tf_rows_sum_to_one(self):
        out = apply_weighting(self.counts, "tf").toarray()
        for row in out:
            if row.sum() > 0:
                self.assertAlmostEqual(float(row.sum()), 1.0, places=6)

    def test_bm25_positive_and_param_sensitive(self):
        a = apply_weighting(self.counts, WeightingScheme(name="bm25", k1=1.2, b=0.75))
        b = apply_weighting(self.counts, WeightingScheme(name="bm25", k1=2.5, b=0.5))
        self.assertGreater(a.nnz, 0)
        self.assertFalse(np.allclose(a.toarray(), b.toarray()))
        self.assertTrue(np.all(a.data >= 0))

    def test_describe_omits_unused_params(self):
        count_meta = describe_weighting("count")
        self.assertEqual(count_meta["parameters"], {})
        self.assertTrue(count_meta["weighting_separate_from_tokenization"])
        bm25_meta = describe_weighting("bm25")
        self.assertIn("k1", bm25_meta["parameters"])
        self.assertIn("b", bm25_meta["parameters"])


class DfmWeightingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tokenized = [
            ["alpha", "alpha", "beta"],
            ["beta", "gamma"],
            ["alpha", "gamma", "gamma", "gamma"],
        ]

    def test_same_tokens_different_weightings(self):
        count = quantitative.build_dfm_matrix(self.tokenized, mode="count")
        log = quantitative.build_dfm_matrix(self.tokenized, mode="log_count")
        bm25 = quantitative.build_dfm_matrix(self.tokenized, mode="bm25", k1=1.5, b=0.75)
        self.assertEqual(count["feature_names"], log["feature_names"])
        self.assertEqual(count["feature_names"], bm25["feature_names"])
        self.assertEqual(log["weighting"], "log_count")
        self.assertEqual(bm25["weighting"], "bm25")
        self.assertEqual(bm25["weighting_scheme"]["parameters"]["k1"], 1.5)
        # log(1+2) for alpha in doc0 vs raw count 2
        alpha_idx = count["feature_names"].index("alpha")
        self.assertAlmostEqual(count["dense_matrix"][0][alpha_idx], 2.0)
        self.assertAlmostEqual(log["dense_matrix"][0][alpha_idx], math.log1p(2.0))

    def test_weighting_does_not_change_tokenizer_path(self):
        # Identical token lists → identical vocab regardless of scheme.
        a = quantitative.build_dfm_matrix(self.tokenized, mode="tfidf", force_sparse_only=True)
        b = quantitative.build_dfm_matrix(self.tokenized, mode="bm25", force_sparse_only=True)
        self.assertEqual(a["feature_names"], b["feature_names"])
        self.assertTrue(a["weighting_scheme"]["weighting_separate_from_tokenization"])


if __name__ == "__main__":
    unittest.main()
