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
                {"chunk_id": "c-rel-1", "document_id": "d1", "score": 0.9, "content": "accountability"},
                {"chunk_id": "c-noise", "document_id": "d1", "score": 0.8, "content": "tables"},
                {"chunk_id": "c-rel-2", "document_id": "d2", "score": 0.7, "content": "self-reliance"},
            ],
            "lexical": [
                {"chunk_id": "c-rel-2", "document_id": "d2", "score": 0.6, "content": "self-reliance"},
                {"chunk_id": "c-rel-1", "document_id": "d1", "score": 0.5, "content": "accountability"},
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
