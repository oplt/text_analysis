"""CSV row-aware parser and PDF block metadata smoke tests."""

from __future__ import annotations

import unittest

from backend.modules.rag.infrastructure.langchain_document_loaders import _parse_csv
from backend.modules.rag.infrastructure.pdf_parsers import BasicPypdfParser


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
