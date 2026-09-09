"""Tests for KWIC / concordance search."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import kwic, quantitative


class KwicModeTests(unittest.TestCase):
    def test_auto_resolves_word_phrase_wildcard(self):
        self.assertEqual(kwic.normalize_query_mode("auto", "freedom"), "word")
        self.assertEqual(kwic.normalize_query_mode("auto", "human rights"), "phrase")
        self.assertEqual(kwic.normalize_query_mode("auto", "educat*"), "wildcard")


class ConcordanceTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Universal education policy matters.",
            "The policy supports universal education rights.",
            "Case Test: Universal vs universal.",
        ]
        self.meta = [
            {
                "text_unit_id": "u1",
                "page_number": 1,
                "section_heading": "Intro",
                "unit_char_start": 100,
                "document_title": "Doc A",
            },
            {
                "text_unit_id": "u2",
                "page_number": 2,
                "corpus_document_id": "d2",
                "unit_char_start": 200,
            },
            {"text_unit_id": "u3", "page_number": 3},
        ]

    def test_word_case_insensitive_and_provenance(self):
        hits = kwic.concordance(self.texts, self.meta, "universal", window=2)
        self.assertGreaterEqual(len(hits), 3)
        first = hits[0]
        self.assertEqual(first["query_mode"], "word")
        self.assertEqual(first["context_unit"], "token")
        self.assertEqual(first["text_unit_id"], "u1")
        self.assertEqual(first["page_number"], 1)
        self.assertEqual(first["section_heading"], "Intro")
        self.assertEqual(first["document_title"], "Doc A")
        self.assertIn("left_tokens", first)
        self.assertIn("match_tokens", first)
        self.assertIsInstance(first["match_char_start"], int)
        self.assertEqual(first["absolute_char_start"], 100 + first["match_char_start"])

    def test_case_sensitive(self):
        hits = kwic.concordance(
            self.texts, self.meta, "Universal", window=1, case_sensitive=True, query_mode="word"
        )
        keywords = [h["keyword"] for h in hits]
        self.assertTrue(all(k.startswith("Universal") or k == "Universal" for k in keywords))
        self.assertTrue(all("universal" != k for k in keywords))

    def test_phrase(self):
        hits = kwic.concordance(
            self.texts, self.meta, "universal education", window=1, query_mode="phrase"
        )
        self.assertGreaterEqual(len(hits), 2)
        for hit in hits:
            self.assertEqual(len(hit["match_tokens"]), 2)
            self.assertIn("education", hit["keyword"].lower())

    def test_exact_phrase_char_span(self):
        hits = kwic.concordance(
            ["alpha beta gamma"],
            [{"text_unit_id": "x"}],
            "beta gamma",
            query_mode="exact_phrase",
            window=1,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["match_char_start"], 6)
        self.assertEqual(hits[0]["match_char_end"], 16)

    def test_regex(self):
        hits = kwic.concordance(
            self.texts, self.meta, r"educat\w+", query_mode="regex", window=1
        )
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(any("educat" in h["keyword"].lower() for h in hits))

    def test_wildcard(self):
        hits = kwic.concordance(self.texts, self.meta, "educat*", query_mode="wildcard", window=1)
        self.assertGreaterEqual(len(hits), 2)

    def test_lemma_english(self):
        texts = ["Policies running policies."]
        meta = [{"text_unit_id": "l1"}]
        hits = kwic.concordance(
            texts, meta, "policy", query_mode="lemma", language="en", window=1
        )
        self.assertGreaterEqual(len(hits), 1)

    def test_invalid_regex_raises(self):
        with self.assertRaises(ValueError):
            kwic.concordance(["a"], [{}], "(", query_mode="regex")

    def test_quantitative_wrapper_preserves_metadata(self):
        results = quantitative.kwic(self.texts, self.meta, "universal", window=2)
        self.assertGreaterEqual(len(results), 2)
        self.assertIn("text_unit_id", results[0])
        self.assertEqual(results[0]["context_unit"], "token")

    def test_kwic_search_payload(self):
        payload = [
            {"text": "human rights matter", "text_unit_id": "a", "page_number": 9},
            {"text": "other text", "text_unit_id": "b"},
        ]
        hits = quantitative.kwic_search(payload, "human rights", window_size=1, query_mode="phrase")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["page_number"], 9)
        self.assertEqual(hits[0]["left_tokens"], [])
        self.assertEqual(hits[0]["right_tokens"], ["matter"])


if __name__ == "__main__":
    unittest.main()
