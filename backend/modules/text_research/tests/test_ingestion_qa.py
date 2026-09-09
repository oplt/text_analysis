"""Tests for ingestion quality diagnostics (no text mutation)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import ingestion_qa as qa
from backend.modules.text_research.infrastructure.canonical_text import sha256_text


class IngestionQaTests(unittest.TestCase):
    def test_empty_document(self):
        findings, metrics = qa.analyze_document_text("   \n")
        codes = {f.code for f in findings}
        self.assertIn("empty_document", codes)
        self.assertFalse(metrics["text_mutated"])

    def test_short_and_replacement_chars(self):
        text = "hi\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd"
        findings, metrics = qa.analyze_document_text(text)
        codes = {f.code for f in findings}
        self.assertIn("suspiciously_short", codes)
        self.assertIn("excessive_replacement_chars", codes)
        self.assertEqual(metrics["text_checksum"], sha256_text(text))

    def test_scanned_pdf_heuristic(self):
        findings, _ = qa.analyze_document_text(
            "....",
            filename="scan.pdf",
            content_type="application/pdf",
        )
        self.assertTrue(any(f.code == "likely_image_only_or_scanned_pdf" for f in findings))

    def test_hyphenation_and_malformed(self):
        text = "edu-\ncation\n" * 10 + "\x00hidden"
        findings, _ = qa.analyze_document_text(text)
        codes = {f.code for f in findings}
        self.assertIn("line_break_hyphenation", codes)
        self.assertIn("malformed_extraction", codes)

    def test_language_mismatch_english_metadata(self):
        text = "これは日本語の文章です。" * 5
        findings, _ = qa.analyze_document_text(text, language="en")
        self.assertTrue(any(f.code == "language_mismatch" for f in findings))

    def test_exact_and_near_duplicates(self):
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

    def test_token_outliers(self):
        docs = [{"document_id": str(i), "token_count": 100} for i in range(8)]
        docs.append({"document_id": "outlier", "token_count": 5000})
        findings = qa.detect_token_count_outliers(docs, z_threshold=2.5)
        self.assertTrue(any(f.details.get("document_id") == "outlier" for f in findings))

    def test_analyze_does_not_mutate_input(self):
        original = "Stable text.\n\nSecond paragraph."
        before = original
        qa.analyze_document_text(original)
        self.assertEqual(original, before)


if __name__ == "__main__":
    unittest.main()
