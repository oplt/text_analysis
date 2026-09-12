from __future__ import annotations

import unittest

from backend.modules.rag.infrastructure.pdf_quality import (
    assess_pdf_text_quality,
    decide_ocr,
    suppress_repeated_headers_footers,
)


class PdfQualityAndOcrTests(unittest.TestCase):
    def test_quality_gate_flags_empty_and_replacement_heavy_text(self):
        empty = assess_pdf_text_quality("")
        self.assertTrue(empty.is_low_quality)
        dirty = assess_pdf_text_quality("\ufffd" * 40)
        self.assertTrue(dirty.is_low_quality)
        clean = assess_pdf_text_quality("Institutional accountability requires evidence. " * 4)
        self.assertFalse(clean.is_low_quality)

    def test_ocr_flag_is_conservative_when_disabled(self):
        decision = decide_ocr("", ocr_enabled=False)
        self.assertFalse(decision.should_ocr)
        self.assertEqual(decision.reason, "ocr_disabled")

    def test_ocr_flag_triggers_only_for_low_quality_when_enabled(self):
        poor = decide_ocr("\ufffd", ocr_enabled=True, min_text_chars=40)
        self.assertTrue(poor.should_ocr)
        rich = decide_ocr(
            "Institutional accountability requires decision makers to explain. " * 3,
            ocr_enabled=True,
        )
        self.assertFalse(rich.should_ocr)
        self.assertEqual(rich.reason, "native_text_sufficient")

    def test_header_footer_suppression_removes_repeated_lines(self):
        pages = [
            "Header Corp\nBody one\nFooter 1",
            "Header Corp\nBody two\nFooter 1",
            "Header Corp\nBody three\nFooter 1",
        ]
        cleaned = suppress_repeated_headers_footers(pages)
        self.assertTrue(all("Header Corp" not in page for page in cleaned))
        self.assertTrue(all("Footer 1" not in page for page in cleaned))
        self.assertIn("Body one", cleaned[0])
