"""Tests for configurable document cleaning (raw never mutated)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import document_cleaning as dc
from backend.modules.text_research.infrastructure.canonical_text import sha256_text


class DocumentCleaningTests(unittest.TestCase):
    def test_identity_config_preserves_text(self):
        raw = "Hello   world.\n\nPage 2\n"
        result = dc.apply_cleaning(raw, {})
        self.assertEqual(result.cleaned_text, raw)
        self.assertEqual(result.raw_checksum, sha256_text(raw))
        self.assertTrue(all(not s.enabled for s in result.steps))

    def test_does_not_mutate_input(self):
        raw = "edu-\ncation  and  spaces"
        before = raw
        dc.apply_cleaning(
            raw,
            {"normalize_whitespace": True, "dehyphenate_line_breaks": True},
        )
        self.assertEqual(raw, before)

    def test_unicode_and_whitespace_and_dehyphenation(self):
        raw = "caf\u00e9 edu-\ncation\n\n\nNext"
        result = dc.apply_cleaning(
            raw,
            {
                "unicode_normalization": "NFC",
                "normalize_whitespace": True,
                "dehyphenate_line_breaks": True,
            },
        )
        self.assertIn("education", result.cleaned_text)
        self.assertNotIn("edu-\n", result.cleaned_text)
        meta = result.to_transformation_metadata()
        self.assertEqual(meta["raw_checksum"], sha256_text(raw))
        enabled = {s["name"] for s in meta["steps"] if s["enabled"]}
        self.assertIn("unicode_normalization", enabled)
        self.assertIn("normalize_whitespace", enabled)
        self.assertIn("dehyphenate_line_breaks", enabled)

    def test_page_numbers_and_bibliography_recorded(self):
        raw = (
            "Intro paragraph one.\n\n"
            "Body paragraph two with substance.\n\n"
            "Closing remarks here.\n\n"
            "12\n\n"
            "References\n"
            "Smith 2020. Book.\n"
            "Jones 2021. Article.\n"
        )
        result = dc.apply_cleaning(
            raw,
            {"remove_page_numbers": True, "exclude_bibliography": True},
        )
        self.assertNotIn("Smith 2020", result.cleaned_text)
        self.assertNotIn("\n12\n", f"\n{result.cleaned_text}\n")
        bib = next(s for s in result.steps if s.name == "exclude_bibliography")
        self.assertTrue(bib.enabled)
        self.assertGreater(bib.discarded_chars, 0)
        self.assertIsNotNone(bib.details.get("discarded_preview") or bib.discarded_preview)

    def test_table_exclusion(self):
        raw = "Prose line.\n| A | B |\n|---|---|\n| 1 | 2 |\nMore prose."
        result = dc.apply_cleaning(raw, {"exclude_tables": True})
        self.assertIn("Prose line", result.cleaned_text)
        self.assertIn("More prose", result.cleaned_text)
        self.assertNotIn("| A | B |", result.cleaned_text)

    def test_custom_regex(self):
        raw = "aaa SECRET bbb SECRET"
        result = dc.apply_cleaning(
            raw,
            {"custom_regex": [{"pattern": "SECRET", "replacement": "[redacted]"}]},
        )
        self.assertEqual(result.cleaned_text, "aaa [redacted] bbb [redacted]")

    def test_invalid_unicode_form_rejected(self):
        with self.assertRaises(ValueError):
            dc.CleaningConfig.from_dict({"unicode_normalization": "BOGUS"})

    def test_preview_cleaning(self):
        preview = dc.preview_cleaning(
            ["foo-\nbar"],
            {"dehyphenate_line_breaks": True},
        )
        self.assertEqual(preview["rows"][0]["cleaned_preview"], "foobar")
        self.assertEqual(preview["engine"], dc.CLEANING_ENGINE)


if __name__ == "__main__":
    unittest.main()
