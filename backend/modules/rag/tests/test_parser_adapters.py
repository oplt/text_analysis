"""CSV row-aware parser and PDF block metadata smoke tests."""

from __future__ import annotations

import shutil
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from backend.modules.rag.infrastructure.langchain_document_loaders import _parse_csv
from backend.modules.rag.infrastructure.pdf_parsers import (
    BasicPypdfParser,
    EnhancedPdfParser,
    _extract_ocr_text,
    _extract_tables,
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
    @unittest.skipUnless(
        shutil.which("tesseract") and find_spec("pytesseract") and find_spec("pymupdf"),
        "Native OCR extra unavailable",
    )
    def test_native_ocr_runs_for_image_only_pdf(self):
        import pymupdf

        with pymupdf.open() as source, pymupdf.open() as scan:
            page = source.new_page()
            page.insert_text((72, 100), "RESEARCH EVIDENCE ACCOUNTABILITY", fontsize=24)
            image = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
            scanned_page = scan.new_page()
            scanned_page.insert_image(scanned_page.rect, stream=image)
            with patch(
                "backend.modules.rag.infrastructure.pdf_parsers.settings.RAG_PDF_OCR_ENABLED", True
            ):
                document = EnhancedPdfParser().parse(scan.tobytes())[0]
        self.assertIn("ACCOUNTABILITY", document.content)
        self.assertTrue(document.metadata["ocr_ran"])
        self.assertEqual(document.metadata["ocr_status"], "ocr_success")
        self.assertEqual(document.metadata["blocks"], [])

    def test_native_capability_failures_are_distinct(self):
        for binary, languages, expected in (
            (None, [], "ocr_binary_missing"),
            ("/usr/bin/tesseract", [], "ocr_language_missing"),
        ):
            ocr = SimpleNamespace(
                pytesseract=SimpleNamespace(tesseract_cmd="tesseract"),
                get_languages=lambda languages=languages, **kwargs: languages,
            )
            with (
                patch(
                    "backend.modules.rag.infrastructure.pdf_parsers.settings.RAG_PDF_OCR_ENABLED",
                    True,
                ),
                patch(
                    "backend.modules.rag.infrastructure.pdf_parsers.shutil.which",
                    return_value=binary,
                ),
                patch.dict(
                    "sys.modules", {"pytesseract": ocr, "PIL": SimpleNamespace(Image=object())}
                ),
            ):
                _, metadata = _extract_ocr_text(object(), object())
            self.assertEqual(metadata["ocr_status"], expected)
            self.assertFalse(metadata["ocr_available"])

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

    def test_ocr_is_not_attempted_without_explicit_configuration(self):
        text, metadata = _extract_ocr_text(object(), object())

        self.assertEqual(text, "")
        self.assertFalse(metadata["ocr_enabled"])
        self.assertFalse(metadata["ocr_ran"])

    def test_table_extraction_preserves_structured_cells(self):
        class Table:
            bbox = (1, 2, 30, 40)

            def extract(self):
                return [["year", "count"], ["2025", "3"]]

        class Page:
            def find_tables(self):
                return type("Found", (), {"tables": [Table()]})()

        with patch(
            "backend.modules.rag.infrastructure.pdf_parsers.settings.RAG_PDF_TABLE_EXTRACTION_ENABLED",
            True,
        ):
            tables = _extract_tables(Page(), page_number=2)

        self.assertEqual(tables[0]["page_number"], 2)
        self.assertEqual(tables[0]["bbox"], [1, 2, 30, 40])
        self.assertEqual(tables[0]["rows"], [["year", "count"], ["2025", "3"]])

    def test_missing_ocr_python_dependencies_are_explicit(self):
        with (
            patch(
                "backend.modules.rag.infrastructure.pdf_parsers.settings.RAG_PDF_OCR_ENABLED", True
            ),
            patch.dict("sys.modules", {"pytesseract": None}),
        ):
            _, metadata = _extract_ocr_text(object(), object())
        self.assertEqual(metadata["ocr_status"], "ocr_python_dependency_missing")
        self.assertFalse(metadata["ocr_available"])

    @unittest.skipUnless(find_spec("pymupdf"), "PyMuPDF optional dependency is not installed")
    def test_ocr_replacement_discards_extraction_block_coordinates(self):
        import pymupdf

        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Bad")
            content = pdf.tobytes()
        with (
            patch(
                "backend.modules.rag.infrastructure.pdf_parsers.settings.RAG_PDF_OCR_ENABLED", True
            ),
            patch(
                "backend.modules.rag.infrastructure.pdf_parsers._extract_ocr_text",
                return_value=(
                    "Readable OCR evidence",
                    {"ocr_ran": True, "ocr_transformed_extract": True, "ocr_status": "ocr_success"},
                ),
            ),
        ):
            document = EnhancedPdfParser().parse(content)[0]
        self.assertEqual(document.content, "Readable OCR evidence")
        self.assertEqual(document.metadata["blocks"], [])
        self.assertFalse(document.metadata["original_layout_coordinates_available"])
        self.assertTrue(
            all(span.startswith("transformed:") for span in document.metadata["source_span_ids"])
        )
        self.assertIn("tesseract-ocr", document.metadata["transformation_chain"]["transforms"])

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
        self.assertTrue(metadata["table_extraction_available"])
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
