"""Tests for user-defined hierarchical dictionary matching."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import dictionary_matcher as dm
from backend.modules.text_research.infrastructure import quantitative


class ParseDictionaryTests(unittest.TestCase):
    def test_flat_list(self):
        spec = dm.parse_dictionary_payload(["alpha", "beta gamma"])
        self.assertEqual(spec.format, dm.FORMAT_FLAT)
        self.assertEqual(spec.source, "user")
        types = {e.expression: e.match_type for e in spec.entries}
        self.assertEqual(types["alpha"], "token")
        self.assertEqual(types["beta gamma"], "phrase")

    def test_hierarchical_yaml_shape(self):
        payload = {
            "category_a": {
                "subcategory_1": ["phrase one", "token_a"],
                "subcategory_2": [{"expression": "wild*", "match": "wildcard"}],
            },
            "category_b": {"subcategory_3": ["another phrase"]},
        }
        spec = dm.parse_dictionary_payload(payload)
        self.assertEqual(spec.format, dm.FORMAT_HIERARCHICAL)
        self.assertGreaterEqual(len(spec.entries), 4)
        by_expr = {e.expression: e for e in spec.entries}
        self.assertEqual(by_expr["phrase one"].category, "category_a")
        self.assertEqual(by_expr["phrase one"].subcategory, "subcategory_1")
        self.assertEqual(by_expr["wild*"].match_type, "wildcard")

    def test_envelope_with_exclusions_and_language(self):
        spec = dm.parse_dictionary_payload(
            {
                "language": "en",
                "exclusions": ["noise term"],
                "hierarchy": {"cat": {"sub": ["keep"]}},
            }
        )
        self.assertEqual(spec.language, "en")
        self.assertEqual(len(spec.exclusions), 1)
        self.assertEqual(spec.entries[0].expression, "keep")


class MatchDictionaryTests(unittest.TestCase):
    def test_token_phrase_wildcard_regex_and_provenance(self):
        tokenized = [
            ["alpha", "beta", "gamma"],
            ["delta", "extra", "word"],
            ["prefix", "match123", "end"],
        ]
        spec = dm.parse_dictionary_payload(
            {
                "hierarchy": {
                    "demo": {
                        "tokens": ["alpha"],
                        "phrases": ["beta gamma"],
                        "wild": [{"expression": "delt*", "match": "wildcard"}],
                        "re": [{"expression": r"match\d+", "match": "regex"}],
                    }
                }
            }
        )
        meta = [
            {"page_number": 1, "document_title": "A"},
            {"page_number": 2},
            {"page_number": 3},
        ]
        result = dm.match_dictionary(
            tokenized,
            spec,
            unit_ids=["u1", "u2", "u3"],
            metadata=meta,
        )
        self.assertGreaterEqual(result["total_hits"], 4)
        self.assertIn("normalized_hits", result)
        self.assertGreater(result["document_prevalence"], 0)
        exprs = {m["matched_expression"] for m in result["matches"]}
        self.assertIn("alpha", exprs)
        self.assertIn("beta gamma", exprs)
        self.assertTrue(any(m["page_number"] == 1 for m in result["matches"]))
        self.assertTrue(any(m["category"] == "demo" for m in result["matches"]))

    def test_exclusions_suppress_overlapping_hits(self):
        tokenized = [["beta", "gamma", "noise"]]
        spec = dm.parse_dictionary_payload(
            {
                "hierarchy": {"c": {"s": ["beta gamma"]}},
                "exclusions": ["beta gamma"],
            }
        )
        result = dm.match_dictionary(tokenized, spec, unit_ids=["u1"])
        self.assertEqual(result["total_hits"], 0)

    def test_synthetic_demo_fixture_marked(self):
        self.assertEqual(dm.SYNTHETIC_DEMO_DICTIONARY["source"], "synthetic_demo")
        self.assertIn("SYNTHETIC", dm.SYNTHETIC_DEMO_DICTIONARY["description"])

    def test_quantitative_dictionary_analysis_hierarchy(self):
        texts = ["alpha and beta gamma together", "nothing here"]
        result = quantitative.dictionary_analysis(
            texts,
            ["a", "b"],
            {
                "hierarchy": {
                    "cat": {"sub": ["alpha", "beta gamma"]},
                }
            },
            {"lowercase": True, "remove_stopwords": False},
        )
        self.assertGreaterEqual(result["total_hits"], 2)
        self.assertIn("matches", result)
        self.assertIn("by_category", result)
        self.assertEqual(result["dictionary"]["source"], "user")


if __name__ == "__main__":
    unittest.main()
