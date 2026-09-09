"""Prepared corpus artifacts are content-addressable and reusable."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts_cached,
)


class PreparedCorpusCacheTests(unittest.TestCase):
    def test_identical_scientific_inputs_reuse_prepared_artifact(self):
        with tempfile.TemporaryDirectory() as artifact_dir, patch.dict(
            "os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}
        ):
            first = prepare_texts_cached(
                ["The market and trade"],
                {"lowercase": True},
                corpus_id="corpus-1",
                unit_type="paragraph",
                unit_ids=["unit-1"],
                filters={"language": "en"},
                cleaning_profile_hash="cleaning-v1",
            )
            second = prepare_texts_cached(
                ["The market and trade"],
                {"lowercase": True},
                corpus_id="corpus-1",
                unit_type="paragraph",
                unit_ids=["unit-1"],
                filters={"language": "en"},
                cleaning_profile_hash="cleaning-v1",
            )

        self.assertEqual(first.corpus_checksum, second.corpus_checksum)
        self.assertEqual(first.pipeline_checksum, second.pipeline_checksum)
        self.assertEqual(first.token_sequences, second.token_sequences)

    def test_changed_preprocessing_produces_distinct_artifact(self):
        with tempfile.TemporaryDirectory() as artifact_dir, patch.dict(
            "os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}
        ):
            lower = prepare_texts_cached(
                ["Market Trade"],
                {"lowercase": True},
                corpus_id="corpus-1",
                unit_type="paragraph",
                unit_ids=["unit-1"],
            )
            original_case = prepare_texts_cached(
                ["Market Trade"],
                {"lowercase": False},
                corpus_id="corpus-1",
                unit_type="paragraph",
                unit_ids=["unit-1"],
            )

        self.assertNotEqual(lower.pipeline_checksum, original_case.pipeline_checksum)
        self.assertNotEqual(lower.token_sequences, original_case.token_sequences)

