"""Tests for full string + token text transform pipeline."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig, tokenize
from backend.modules.text_research.infrastructure.text_transforms import (
    apply_pipeline,
    build_default_text_pipeline,
    build_full_pipeline,
)


class FullTextTransformPipelineTests(unittest.TestCase):
    def test_build_full_pipeline_includes_token_stages(self):
        cfg = PreprocessingConfig(
            remove_stopwords=True,
            stemming=True,
        ).to_dict()
        pipeline = build_full_pipeline(cfg)
        names = [step["name"] for step in pipeline.describe()]
        self.assertIn("tokenize", names)
        self.assertIn("remove_stopwords", names)
        self.assertIn("stem", names)

    def test_apply_pipeline_matches_tokenize(self):
        texts = [
            "The governments are discussing policies in parliament.",
            "Another sentence about economy and trade.",
        ]
        cfg = PreprocessingConfig(
            lowercase=True,
            remove_stopwords=True,
            stemming=True,
        ).to_dict()
        pipeline = build_full_pipeline(cfg)
        transformed = apply_pipeline(texts, pipeline, context={"config": cfg})
        expected = [tokenize(text, cfg) for text in texts]
        self.assertEqual(transformed, expected)

    def test_build_default_text_pipeline_unchanged(self):
        cfg = PreprocessingConfig(lowercase=False).to_dict()
        pipeline = build_default_text_pipeline(cfg)
        names = [step["name"] for step in pipeline.describe()]
        self.assertNotIn("tokenize", names)
        self.assertNotIn("lowercase", names)


if __name__ == "__main__":
    unittest.main()
