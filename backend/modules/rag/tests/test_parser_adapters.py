"""CSV row-aware parser and PDF block metadata smoke tests."""

from __future__ import annotations

import unittest
from importlib.util import find_spec
from unittest.mock import patch

from backend.modules.rag.infrastructure.langchain_document_loaders import _parse_csv
from backend.modules.rag.infrastructure.pdf_parsers import (
    BasicPypdfParser,
    EnhancedPdfParser,
    _stable_block_id,
    parse_pdf_with_fallback,
)


class CsvParserTests(unittest.TestCase):
    def test_csv_emits_one_document_per_row(self):
        raw = b"name,topic\nAlice,activation\nBob,solidarity\n"
        docs = _parse_csv(raw)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["row_number"], 1)
        self.assertIn("name: Alice", docs[0].content)
        self.assertEqual(docs[1].metadata["headers"], ["name", "topic"])
        self.assertNotEqual(len(docs), 1)


class PdfParserSmokeTests(unittest.TestCase):
    def test_basic_parser_name(self):
        self.assertEqual(BasicPypdfParser.name, "pypdf-v1")

    def test_block_ids_are_deterministic(self):
        first = _stable_block_id(page_number=1, block_index=0, text="A", bbox=(1, 2, 3, 4))
        second = _stable_block_id(page_number=1, block_index=0, text="A", bbox=(1, 2, 3, 4))
        self.assertEqual(first, second)

    def test_parser_mode_can_force_reliable_baseline(self):
        with patch(
            "backend.modules.rag.infrastructure.pdf_parsers.BasicPypdfParser.parse",
            return_value=[],
        ) as parse:
            result = parse_pdf_with_fallback(b"not a pdf", parser_mode="pypdf")

        self.assertEqual(result, [])
        parse.assert_called_once_with(b"not a pdf", configured_parser="pypdf")

    @unittest.skipUnless(
        find_spec("pymupdf") or find_spec("fitz"),
        "PyMuPDF optional dependency is not installed",
    )
    def test_enhanced_parser_smoke_reports_capabilities(self):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]  # compatibility with older PyMuPDF

        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Enhanced parser smoke test")
        content = pdf.tobytes()
        pdf.close()

        documents = EnhancedPdfParser().parse(content, configured_parser="pymupdf")

        self.assertEqual(len(documents), 1)
        metadata = documents[0].metadata
        self.assertEqual(metadata["parser"], EnhancedPdfParser.name)
        self.assertEqual(metadata["parser_quality"], "enhanced_layout")
        self.assertTrue(metadata["layout_extraction_available"])
        self.assertFalse(metadata["table_extraction_available"])
        self.assertFalse(metadata["table_extraction_ran"])
        self.assertFalse(metadata["ocr_ran"])
        self.assertEqual(len(metadata["blocks"]), 1)
        self.assertEqual(
            metadata["blocks"][0]["block_id"],
            _stable_block_id(
                page_number=1,
                block_index=0,
                text="Enhanced parser smoke test",
                bbox=metadata["blocks"][0]["bbox"],
            ),
        )
