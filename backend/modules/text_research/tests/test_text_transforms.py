"""Tests for composable string-level text transforms."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.text_transforms import (
    Compose,
    FixEncoding,
    Lowercase,
    NormalizeWhitespace,
    UnicodeNormalize,
    build_default_text_pipeline,
)


class TextTransformTests(unittest.TestCase):
    def test_compose_applies_steps_in_order(self):
        pipeline = Compose(
            steps=[
                FixEncoding(),
                UnicodeNormalize(),
                NormalizeWhitespace(),
                Lowercase(),
            ]
        )
        raw = "  HELLO   WORLD  "
        out = pipeline.transform(raw, context={"config": {"unicode_normalization": "NFC"}})
        self.assertEqual(out, "hello world")

    def test_build_default_text_pipeline_respects_config(self):
        cfg = {
            "fix_encoding": False,
            "unicode_normalization": "NFC",
            "lowercase": False,
        }
        pipeline = build_default_text_pipeline(cfg)
        names = [step["name"] for step in pipeline.describe()]
        self.assertIn("unicode_normalize", names)
        self.assertIn("normalize_whitespace", names)
        self.assertNotIn("fix_encoding", names)
        self.assertNotIn("lowercase", names)

        out = pipeline.transform("  Mixed CASE  ", context={"config": cfg})
        self.assertEqual(out, "Mixed CASE")

    def test_lowercase_transform_tokens(self):
        step = Lowercase()
        tokens = step.transform_tokens(["Hello", "WORLD"], context={})
        self.assertEqual(tokens, ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
