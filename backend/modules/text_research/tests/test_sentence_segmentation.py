"""Regression tests for improved sentence segmentation."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure import language_processing as lang
from backend.modules.text_research.infrastructure import segmentation


def _texts(spans_or_units) -> list[str]:
    if not spans_or_units:
        return []
    first = spans_or_units[0]
    if isinstance(first, dict):
        return [u["text"] for u in spans_or_units]
    return [s[2] for s in spans_or_units]


class SentenceSegmentationRegressionTests(unittest.TestCase):
    def test_basic_still_splits(self):
        text = "Hello world. This is great! Is it? Yes."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(
            _texts(units),
            ["Hello world.", "This is great!", "Is it?", "Yes."],
        )

    def test_abbreviations_eg_ie_dr_fig(self):
        text = (
            "See e.g. the sample. Also i.e. this case. "
            "Dr. Smith arrived. Fig. 2 shows results."
        )
        units = segmentation.segment_document(text, "sentence", language="en")
        texts = _texts(units)
        self.assertEqual(len(texts), 4)
        self.assertTrue(texts[0].startswith("See e.g."))
        self.assertTrue(texts[1].startswith("Also i.e."))
        self.assertIn("Dr. Smith", texts[2])
        self.assertIn("Fig. 2", texts[3])

    def test_decimal_not_split(self):
        text = "The value is 3.14 exactly. Next sentence."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(_texts(units), ["The value is 3.14 exactly.", "Next sentence."])

    def test_us_acronym_before_lowercase(self):
        text = "He lives in the U.S. economy zone. True."
        units = segmentation.segment_document(text, "sentence", language="en")
        texts = _texts(units)
        self.assertEqual(len(texts), 2)
        self.assertIn("U.S. economy", texts[0])

    def test_url_not_split(self):
        text = "Read https://example.com/a.b.c/path for details. Then continue."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(len(_texts(units)), 2)
        self.assertIn("https://example.com/a.b.c/path", _texts(units)[0])

    def test_email_not_split(self):
        text = "Contact a.b@example.org today. Thanks."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(_texts(units), ["Contact a.b@example.org today.", "Thanks."])

    def test_initials(self):
        text = "J. K. Rowling wrote books. Readers agree."
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(len(_texts(units)), 2)
        self.assertTrue(_texts(units)[0].startswith("J. K. Rowling"))

    def test_quotations_keep_closer(self):
        text = 'She said "Stop." Then she left.'
        units = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(_texts(units), ['She said "Stop."', "Then she left."])

    def test_numbered_list(self):
        text = "1. First item continues here. 2. Second item follows."
        units = segmentation.segment_document(text, "sentence", language="en")
        texts = _texts(units)
        self.assertEqual(len(texts), 2)
        self.assertTrue(texts[0].startswith("1. First"))
        self.assertTrue(texts[1].startswith("2. Second"))

    def test_offsets_slice_original(self):
        text = "Dr. Smith saw 3.14 as pi. Done."
        units = segmentation.segment_document(text, "sentence", language="en")
        for unit in units:
            self.assertEqual(text[unit["char_start"] : unit["char_end"]], unit["text"])

    def test_deterministic(self):
        text = "See e.g. Fig. 2 and U.S. maps. Final."
        a = segmentation.segment_document(text, "sentence", language="en")
        b = segmentation.segment_document(text, "sentence", language="en")
        self.assertEqual(a, b)

    def test_german_abbreviation(self):
        text = "Siehe z.B. die Liste. Ende."
        spans = lang.split_sentences(text, "de")
        self.assertEqual(len(spans), 2)
        self.assertIn("z.B.", spans[0][2])

    def test_false_boundary_helper_decimal(self):
        text = "x 3.14 y"
        # period between 3 and 14
        idx = text.index(".")
        self.assertTrue(lang.is_false_sentence_boundary(text, idx, idx + 1, "en"))


if __name__ == "__main__":
    unittest.main()
