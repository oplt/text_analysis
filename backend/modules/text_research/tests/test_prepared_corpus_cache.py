"""Prepared corpus artifacts are content-addressable and reusable."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    PreparationIdentity,
    _prepared_corpus_cache_parts,
    prepare_texts,
    prepare_texts_cached,
)


class PreparedCorpusCacheTests(unittest.TestCase):
    def test_identical_scientific_inputs_reuse_prepared_artifact(self):
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
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

    def test_prepare_texts_cached_hits_on_second_call(self):
        """Second identical call must not re-run prepare_texts (stage cache hit)."""
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
            patch(
                "backend.modules.text_research.infrastructure.prepared_corpus_builder.prepare_texts",
                wraps=prepare_texts,
            ) as mocked_prepare,
        ):
            kwargs = dict(
                corpus_id="corpus-hit",
                unit_type="paragraph",
                unit_ids=["unit-1"],
                filters={"language": "en"},
                cleaning_profile_hash="cleaning-v1",
            )
            first = prepare_texts_cached(["Cache me please"], {"lowercase": True}, **kwargs)
            second = prepare_texts_cached(["Cache me please"], {"lowercase": True}, **kwargs)

        self.assertEqual(mocked_prepare.call_count, 1)
        self.assertEqual(first.token_sequences, second.token_sequences)
        self.assertEqual(first.corpus_checksum, second.corpus_checksum)

    def test_changed_preprocessing_produces_distinct_artifact(self):
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
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

    def _cache_key(self, **kwargs):
        texts = kwargs.pop("texts", ["The market and trade"])
        config = kwargs.pop("config", {"lowercase": True})
        base = dict(
            corpus_id="corpus-1",
            unit_type="paragraph",
            unit_ids=["unit-1"],
            filters={"language": "en"},
            cleaning_profile_hash="cleaning-v1",
        )
        base.update(kwargs)
        key, _factory = _prepared_corpus_cache_parts(texts, config, **base)
        return key

    def test_language_override_changes_cache_key(self):
        base = self._cache_key()
        overridden = self._cache_key(language_override="de")
        self.assertNotEqual(base, overridden)

    def test_per_unit_language_map_changes_cache_key(self):
        without = self._cache_key(per_unit_language=True)
        with_map = self._cache_key(
            per_unit_language=True,
            language_overrides_by_unit={"unit-1": "fr"},
        )
        self.assertNotEqual(without, with_map)

    def test_metadata_by_unit_changes_cache_key(self):
        without = self._cache_key()
        with_meta = self._cache_key(metadata_by_unit={"unit-1": {"speaker": "A"}})
        self.assertNotEqual(without, with_meta)

    def test_cleaned_texts_and_provenance_change_cache_key(self):
        base = self._cache_key()
        cleaned = self._cache_key(cleaned_texts=["cleaned market trade"])
        provenanced = self._cache_key(provenance={"cleaner": "v2"})
        self.assertNotEqual(base, cleaned)
        self.assertNotEqual(base, provenanced)
        self.assertNotEqual(cleaned, provenanced)

    def test_same_semantic_inputs_same_cache_key(self):
        a = self._cache_key(
            language_mode="auto",
            language_override="en",
            metadata_by_unit={"unit-1": {"k": 1}},
            provenance={"src": "test"},
        )
        b = self._cache_key(
            language_mode="auto",
            language_override="en",
            metadata_by_unit={"unit-1": {"k": 1}},
            provenance={"src": "test"},
        )
        self.assertEqual(a, b)

    def test_preparation_identity_excludes_analysis_only_knobs(self):
        identity = PreparationIdentity.from_prepare_inputs(
            corpus_id="c",
            unit_type="paragraph",
            config={"lowercase": True},
            language_override="en",
        )
        spec = identity.to_cache_spec()
        self.assertIn("language_override", spec)
        self.assertNotIn("top_n", spec)
        self.assertNotIn("force_in_memory", spec)


if __name__ == "__main__":
    unittest.main()
