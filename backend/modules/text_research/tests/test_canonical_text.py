"""Canonical research source must not be built from overlapping RAG chunks."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import canonical_text as ct


class CanonicalTextTests(unittest.TestCase):
    def test_build_from_pages_does_not_duplicate_content(self):
        pages = [
            {"content": "Alpha paragraph one.", "page_number": 1},
            {"content": "Beta paragraph two.", "page_number": 2},
        ]
        result = ct.build_canonical_from_pages(pages)
        self.assertEqual(result.text, "Alpha paragraph one.\n\nBeta paragraph two.")
        self.assertEqual(result.text_checksum, ct.sha256_text(result.text))
        self.assertEqual(len(result.page_provenance), 2)
        self.assertEqual(result.page_provenance[0]["char_start"], 0)
        self.assertEqual(result.page_provenance[0]["char_end"], len("Alpha paragraph one."))

    def test_retrieval_overlap_detected_and_naive_join_duplicates(self):
        # Simulate RecursiveCharacterTextSplitter-style adjacent overlap.
        shared = "jumps over the lazy dog near the river bank"
        left = f"The quick brown fox {shared}"
        right = f"{shared} while birds sing at dawn."
        self.assertGreater(ct.adjacent_chunk_overlap_chars(left, right), 0)
        self.assertTrue(ct.chunks_have_retrieval_overlap([left, right]))

        naive = ct.naive_join_chunks([left, right], separator="\n")
        # Overlapped span appears twice in naive reconstruction.
        self.assertEqual(naive.count(shared), 2)

        # Canonical path uses full document once — no chunk overlap join.
        canonical = ct.build_canonical_from_full_text(
            f"The quick brown fox {shared} while birds sing at dawn."
        )
        self.assertEqual(canonical.text.count(shared), 1)
        self.assertNotEqual(ct.sha256_text(naive), canonical.text_checksum)

    def test_non_overlapping_chunks_not_flagged(self):
        chunks = ["First complete sentence.", "Second complete sentence."]
        self.assertFalse(ct.chunks_have_retrieval_overlap(chunks))

    def test_full_text_normalizes_newlines(self):
        result = ct.build_canonical_from_full_text("line1\r\nline2\rline3")
        self.assertEqual(result.text, "line1\nline2\nline3")


if __name__ == "__main__":
    unittest.main()
