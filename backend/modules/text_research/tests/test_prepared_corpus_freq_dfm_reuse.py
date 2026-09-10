"""Phase 4: frequencies then DFM with identical prep reuse prepared-corpus cache."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts,
    prepare_texts_cached,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class FrequenciesThenDfmCacheReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.env = patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": self.temp_dir.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        stage_cache.invalidate()
        stage_cache.reset_hit_miss_counts_for_tests()

    def tearDown(self) -> None:
        stage_cache.invalidate()
        stage_cache.reset_hit_miss_counts_for_tests()

    def test_frequencies_then_dfm_identical_prep_reuses_cache(self) -> None:
        """Same corpus/preprocessing inputs → second prepare is a stage-cache hit."""
        texts = [
            "The market and trade policy shape outcomes.",
            "Trade and market institutions matter for growth.",
        ]
        unit_ids = ["u1", "u2"]
        config = PreprocessingConfig(lowercase=True).to_dict()
        shared_kwargs = dict(
            corpus_id="corpus-freq-dfm",
            unit_type="paragraph",
            unit_ids=unit_ids,
            document_ids=["d1", "d2"],
            filters={"language": "en"},
            cleaning_profile_hash="cleaning-v1",
            operation_config={},
        )

        with patch(
            "backend.modules.text_research.infrastructure.prepared_corpus_builder.prepare_texts",
            wraps=prepare_texts,
        ) as mocked_prepare:
            # Emulate frequencies prep
            freq_prepared = prepare_texts_cached(texts, config, **shared_kwargs)
            # Emulate DFM prep with identical scientific inputs
            dfm_prepared = prepare_texts_cached(texts, config, **shared_kwargs)

        self.assertEqual(mocked_prepare.call_count, 1)
        self.assertEqual(freq_prepared.token_sequences, dfm_prepared.token_sequences)
        self.assertEqual(freq_prepared.corpus_checksum, dfm_prepared.corpus_checksum)
        self.assertEqual(freq_prepared.pipeline_checksum, dfm_prepared.pipeline_checksum)

        counts = stage_cache.stage_cache_hit_miss_counts()
        self.assertGreaterEqual(counts["misses"], 1)
        self.assertGreaterEqual(counts["hits"], 1)
