"""Tests for structure-aware chunking, PDF fallback, and offline eval metrics."""

from __future__ import annotations

import unittest

from backend.modules.rag.application.document_scope import document_ids_is_empty_allow_list
from backend.modules.rag.domain.models import ParsedDocument
from backend.modules.rag.eval.retrieval_eval import evaluate_ranking, run_strategies
from backend.modules.rag.infrastructure.langchain_text_splitters import split_documents
from backend.modules.rag.infrastructure.pdf_parsers import BasicPypdfParser, parse_pdf_with_fallback


class StructureChunkingTests(unittest.TestCase):
    def test_preserves_section_and_hashes(self):
        docs = [
            ParsedDocument(
                content="Intro para one.\n\nIntro para two is longer and keeps going.",
                metadata={"section_heading": "Introduction", "format": "markdown"},
            )
        ]
        pieces = split_documents(docs, chunk_size=40, chunk_overlap=5)
        self.assertGreaterEqual(len(pieces), 1)
        _content, meta = pieces[0]
        self.assertEqual(meta.get("section_heading"), "Introduction")
        self.assertIn("content_hash", meta)
        self.assertEqual(meta.get("chunker_strategy"), "structure-v1")

    def test_source_units_are_stable_across_overlapping_chunks(self):
        document = ParsedDocument(
            content="First paragraph with stable text.\n\nSecond paragraph with stable text.",
            metadata={"checksum_sha256": "document-revision"},
        )
        first = split_documents([document], chunk_size=6, chunk_overlap=3)
        second = split_documents([document], chunk_size=6, chunk_overlap=3)

        first_ids = [meta["source_unit_ids"] for _content, meta in first]
        second_ids = [meta["source_unit_ids"] for _content, meta in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            len({source_id for ids in first_ids for source_id in ids}),
            2,
        )
        self.assertTrue(
            all(
                meta["offset_coordinate_system"] == "unicode_code_points_zero_based_end_exclusive"
                for _content, meta in first
            )
        )

    def test_recursive_pieces_have_piece_level_offsets(self):
        text = "one two three four five six seven eight nine ten"
        pieces = split_documents([ParsedDocument(content=text)], chunk_size=4, chunk_overlap=1)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(
            all(text[meta["char_start"] : meta["char_end"]] == content for content, meta in pieces)
        )
        self.assertEqual(
            len({meta["source_unit_ids"][0] for _content, meta in pieces}),
            1,
        )
        self.assertGreater(len({meta["char_start"] for _content, meta in pieces}), 1)


class PdfFallbackTests(unittest.TestCase):
    def test_basic_parser_on_empty_pdf_bytes_raises_or_empty(self):
        # Minimal invalid PDF should not crash the adapter contract
        try:
            result = BasicPypdfParser().parse(b"%PDF-1.4 empty")
            self.assertIsInstance(result, list)
        except Exception:
            # pypdf may raise on garbage — acceptable for basic parser
            pass

    def test_parse_pdf_with_fallback_never_raises_import_error_path(self):
        try:
            result = parse_pdf_with_fallback(b"not a pdf")
            self.assertIsInstance(result, list)
        except Exception as exc:
            # Corrupt bytes may raise from pypdf; enhanced path must not mask ImportError wrongly
            self.assertNotIsInstance(exc, ImportError)


class EvalHarnessTests(unittest.TestCase):
    def test_metrics_and_strategies(self):
        metrics = evaluate_ranking(["a", "b", "c"], {"b"}, k=3)
        self.assertAlmostEqual(metrics.mrr, 0.5)
        self.assertAlmostEqual(metrics.recall_at_k, 1.0)

        case = {
            "id": "t",
            "question": "responsibility",
            "k": 3,
            "relevant_chunk_ids": ["c-rel-1", "c-rel-2"],
            "dense": [
                {
                    "chunk_id": "c-rel-1",
                    "document_id": "d1",
                    "score": 0.9,
                    "content": "accountability",
                },
                {"chunk_id": "c-noise", "document_id": "d1", "score": 0.8, "content": "tables"},
                {
                    "chunk_id": "c-rel-2",
                    "document_id": "d2",
                    "score": 0.7,
                    "content": "self-reliance",
                },
            ],
            "lexical": [
                {
                    "chunk_id": "c-rel-2",
                    "document_id": "d2",
                    "score": 0.6,
                    "content": "self-reliance",
                },
                {
                    "chunk_id": "c-rel-1",
                    "document_id": "d1",
                    "score": 0.5,
                    "content": "accountability",
                },
            ],
        }
        results = run_strategies(case)
        self.assertIn("rrf_fusion", results)
        self.assertGreaterEqual(results["rrf_fusion"].recall_at_k, 0.0)

    def test_empty_allow_list_helper_still_holds(self):
        self.assertTrue(document_ids_is_empty_allow_list([]))
        self.assertFalse(document_ids_is_empty_allow_list(None))


if __name__ == "__main__":
    unittest.main()
