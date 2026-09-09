"""Tests for exact / normalized / lexical / MinHash duplicate detection."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import duplicate_detection as dd
from backend.modules.text_research.infrastructure.canonical_text import sha256_text


class ChecksumTests(unittest.TestCase):
    def test_exact_checksum_matches_sha256_text(self):
        text = "Hello, world."
        self.assertEqual(dd.exact_checksum(text), sha256_text(text))

    def test_exact_checksum_sensitive_to_case_and_whitespace(self):
        self.assertNotEqual(dd.exact_checksum("Hello  World"), dd.exact_checksum("hello world"))

    def test_normalized_checksum_collapses_whitespace_and_case(self):
        a = dd.normalized_checksum("Hello   World\n\n")
        b = dd.normalized_checksum("hello world")
        self.assertEqual(a, b)

    def test_normalized_checksum_differs_from_exact_when_whitespace_differs(self):
        text = "Hello   World"
        self.assertNotEqual(dd.exact_checksum(text), dd.normalized_checksum(text))


class LexicalSimilarityTests(unittest.TestCase):
    def test_identical_text_scores_one(self):
        text = "The committee met on Tuesday to review the draft report."
        self.assertAlmostEqual(dd.lexical_similarity(text, text), 1.0)

    def test_similar_text_scores_high(self):
        base = "The committee met on Tuesday to review the draft report carefully."
        near = base + " Extra clause."
        score = dd.lexical_similarity(base, near)
        self.assertGreater(score, 0.7)

    def test_dissimilar_text_scores_low(self):
        score = dd.lexical_similarity(
            "The committee met on Tuesday.", "Quarterly revenue increased significantly."
        )
        self.assertLess(score, 0.3)

    def test_char_ngrams_short_text_fallback(self):
        grams = dd.char_ngrams("hi", n=5)
        self.assertEqual(sum(grams.values()), 1)

    def test_jaccard_counters_both_empty(self):
        self.assertEqual(dd.jaccard_counters(dd.char_ngrams(""), dd.char_ngrams("")), 1.0)


class MinHashTests(unittest.TestCase):
    def test_identical_text_signatures_match_fully(self):
        text = "the quick brown fox jumps over the lazy dog " * 3
        sig_a = dd.minhash_signature(text, num_perm=32, shingle_size=3)
        sig_b = dd.minhash_signature(text, num_perm=32, shingle_size=3)
        self.assertEqual(dd.minhash_jaccard(sig_a, sig_b), 1.0)

    def test_deterministic_given_same_seed(self):
        text = "reproducibility matters for research pipelines"
        sig_a = dd.minhash_signature(text, num_perm=16, seed=7)
        sig_b = dd.minhash_signature(text, num_perm=16, seed=7)
        self.assertEqual(sig_a.values, sig_b.values)

    def test_different_seed_gives_different_signature(self):
        text = "reproducibility matters for research pipelines and beyond"
        sig_a = dd.minhash_signature(text, num_perm=16, seed=1)
        sig_b = dd.minhash_signature(text, num_perm=16, seed=2)
        self.assertNotEqual(sig_a.values, sig_b.values)

    def test_similar_documents_have_high_estimated_jaccard(self):
        base = (
            "research on comparative discourse analysis requires careful "
            "attention to preprocessing pipelines and reproducibility"
        )
        near = base + " and transparent reporting"
        far = "completely unrelated financial quarterly earnings statement text"

        sig_base = dd.minhash_signature(base, num_perm=64, shingle_size=3)
        sig_near = dd.minhash_signature(near, num_perm=64, shingle_size=3)
        sig_far = dd.minhash_signature(far, num_perm=64, shingle_size=3)

        near_score = dd.minhash_jaccard(sig_base, sig_near)
        far_score = dd.minhash_jaccard(sig_base, sig_far)
        self.assertGreater(near_score, far_score)
        self.assertGreater(near_score, 0.5)

    def test_mismatched_num_perm_raises(self):
        sig_a = dd.minhash_signature("abc", num_perm=8)
        sig_b = dd.minhash_signature("abc", num_perm=16)
        with self.assertRaises(ValueError):
            dd.minhash_jaccard(sig_a, sig_b)

    def test_empty_text_signature_is_stable(self):
        sig_a = dd.minhash_signature("", num_perm=8)
        sig_b = dd.minhash_signature("   ", num_perm=8)
        self.assertEqual(sig_a.values, sig_b.values)

    def test_lsh_candidate_pairs_finds_near_duplicates(self):
        shared = "policy analysis requires careful attention to methodology and data quality " * 2
        near = shared + " plus extra context"
        far = "totally different unrelated content about cooking recipes"
        signatures = {
            "a": dd.minhash_signature(shared, num_perm=32, shingle_size=3, seed=1),
            "b": dd.minhash_signature(near, num_perm=32, shingle_size=3, seed=1),
            "c": dd.minhash_signature(far, num_perm=32, shingle_size=3, seed=1),
        }
        candidates = dd.minhash_lsh_candidate_pairs(signatures, num_bands=8)
        self.assertIn(("a", "b"), candidates)

    def test_lsh_requires_at_least_two_signatures(self):
        self.assertEqual(dd.minhash_lsh_candidate_pairs({}), set())
        one = {"a": dd.minhash_signature("x", num_perm=8)}
        self.assertEqual(dd.minhash_lsh_candidate_pairs(one), set())


class GroupByKeyTests(unittest.TestCase):
    def test_groups_ids_sharing_key(self):
        groups = dd.group_by_key([("a", "k1"), ("b", "k1"), ("c", "k2")])
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["ids"]), {"a", "b"})
        self.assertEqual(groups[0]["key"], "k1")

    def test_singleton_keys_excluded(self):
        groups = dd.group_by_key([("a", "k1"), ("b", "k2")])
        self.assertEqual(groups, [])


class DuplicateReportTests(unittest.TestCase):
    def setUp(self):
        shared = "The committee met on Tuesday to review the draft report carefully. " * 3
        self.items = [
            {"id": "doc_a", "text": shared},
            {"id": "doc_b", "text": shared},
            {"id": "doc_c", "text": shared + " Extra clause about the budget."},
            {"id": "doc_d", "text": "Completely unrelated content about quarterly revenue."},
        ]

    def test_exact_duplicates_grouped(self):
        report = dd.duplicate_report(self.items, methods=["exact"])
        groups = report["exact_duplicate_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["ids"]), {"doc_a", "doc_b"})
        self.assertEqual(report["summary"]["exact_duplicate_groups"], 1)

    def test_normalized_duplicates_catch_whitespace_variants(self):
        items = [
            {"id": "x", "text": "Hello   World"},
            {"id": "y", "text": "hello world"},
            {"id": "z", "text": "Something else entirely different here."},
        ]
        report = dd.duplicate_report(items, methods=["exact", "normalized"])
        self.assertEqual(report["exact_duplicate_groups"], [])
        self.assertEqual(len(report["normalized_duplicate_groups"]), 1)
        self.assertEqual(set(report["normalized_duplicate_groups"][0]["ids"]), {"x", "y"})

    def test_lexical_near_duplicates_detected(self):
        report = dd.duplicate_report(self.items, methods=["lexical"], lexical_threshold=0.85)
        pairs = report["lexical_near_duplicates"]
        self.assertTrue(pairs)
        pair_ids = {frozenset((p["source_id"], p["target_id"])) for p in pairs}
        self.assertIn(frozenset({"doc_a", "doc_b"}), pair_ids)
        for pair in pairs:
            self.assertGreaterEqual(pair["score"], 0.85)

    def test_minhash_tier_detects_near_duplicates(self):
        report = dd.duplicate_report(
            self.items,
            methods=["minhash"],
            minhash_threshold=0.5,
            minhash_num_perm=32,
        )
        pairs = report["minhash_near_duplicates"]
        pair_ids = {frozenset((p["source_id"], p["target_id"])) for p in pairs}
        self.assertIn(frozenset({"doc_a", "doc_b"}), pair_ids)
        self.assertIn("minhash_used_lsh", report)

    def test_all_tiers_together(self):
        report = dd.duplicate_report(
            self.items, methods=["exact", "normalized", "lexical", "minhash"]
        )
        self.assertEqual(report["methods"], ["exact", "normalized", "lexical", "minhash"])
        self.assertEqual(report["item_count"], 4)
        summary = report["summary"]
        for key in (
            "exact_duplicate_groups",
            "normalized_duplicate_groups",
            "lexical_near_duplicate_pairs",
            "minhash_near_duplicate_pairs",
        ):
            self.assertIn(key, summary)

    def test_unsupported_method_raises_clear_error(self):
        with self.assertRaises(ValueError):
            dd.duplicate_report(self.items, methods=["fuzzy_llm_match"])

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            dd.duplicate_report(self.items, methods=["lexical"], lexical_threshold=1.5)

    def test_empty_items_yield_empty_report(self):
        report = dd.duplicate_report([], methods=["exact", "normalized", "lexical"])
        self.assertEqual(report["item_count"], 0)
        self.assertEqual(report["summary"]["exact_duplicate_groups"], 0)

    def test_items_without_id_are_skipped(self):
        report = dd.duplicate_report([{"text": "no id here"}], methods=["exact"])
        self.assertEqual(report["item_count"], 0)


class CapabilitiesTests(unittest.TestCase):
    def test_describe_capabilities_lists_all_methods(self):
        caps = dd.describe_duplicate_detection_capabilities()
        self.assertEqual(set(caps["methods"]), dd.DUPLICATE_METHODS)
        self.assertIn("ingestion QA", " ".join(caps["usable_from"]))


class IngestionQaIntegrationTests(unittest.TestCase):
    """Ingestion QA delegates to duplicate_detection; verify they agree."""

    def test_ingestion_qa_exact_and_near_duplicates_use_duplicate_detection(self):
        from backend.modules.text_research.infrastructure import ingestion_qa as qa

        shared = "The committee met on Tuesday to review the draft report carefully. " * 3
        docs = [
            {"document_id": "a", "checksum": sha256_text(shared), "text": shared},
            {"document_id": "b", "checksum": sha256_text(shared), "text": shared},
            {
                "document_id": "c",
                "checksum": sha256_text(shared + " Extra clause."),
                "text": shared + " Extra clause.",
            },
        ]
        exact = qa.detect_exact_duplicates(docs)
        self.assertEqual(len(exact), 1)
        self.assertEqual(set(exact[0].details["document_ids"]), {"a", "b"})

        near = qa.detect_near_duplicates(
            [{"document_id": d["document_id"], "text": d["text"]} for d in docs],
            threshold=0.85,
        )
        self.assertTrue(any(f.code == "likely_near_duplicate" for f in near))


if __name__ == "__main__":
    unittest.main()
