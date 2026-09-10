"""Tests for P2 stage cache, out-of-core processing, and parquet artifacts."""

from __future__ import annotations

import os
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse

from backend.modules.text_research.infrastructure import (
    artifact_registry,
    clustering,
    feature_cache,
    quantitative,
    stage_cache,
)
from backend.modules.text_research.infrastructure.out_of_core import (
    build_hashing_matrix,
    dense_preview,
    duplicate_methods_for_scale,
    iter_text_batches,
    prepare_texts_batched,
    recommend_clustering_algorithm,
    recommend_vectorizer_mode,
    should_force_sparse_only,
    should_use_out_of_core,
    sparse_payload_from_matrix,
    streaming_token_counts,
)
from backend.modules.text_research.infrastructure.parquet_artifacts import (
    load_sparse_matrix,
    load_unit_table,
    parquet_available,
    save_prepared_artifact_summary,
    save_sparse_matrix,
    save_unit_table,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig, tokenize


class StageCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        os.environ["RESEARCH_ARTIFACT_DIR"] = self.temp_dir.name
        stage_cache.invalidate()

    def tearDown(self) -> None:
        os.environ.pop("RESEARCH_ARTIFACT_DIR", None)

    def test_put_get_and_idempotent(self) -> None:
        key = stage_cache.stage_cache_key(
            engine_version=ENGINE_VERSION,
            stage_name="prepare_corpus",
            input_checksum="abc",
            spec_hash="def",
            params={"batch_size": 500},
        )
        meta = {"stage_name": "prepare_corpus", "status": "complete"}
        payload = {"unit_count": 3}

        first_path = stage_cache.put_stage(key, meta=meta, payload=payload, payload_format="json")
        second_path = stage_cache.put_stage(key, meta=meta, payload=payload, payload_format="json")
        self.assertEqual(first_path, second_path)
        self.assertTrue(stage_cache.has_stage(key))

        loaded = stage_cache.get_stage(key)
        assert loaded is not None
        self.assertEqual(loaded["stage_name"], "prepare_corpus")
        self.assertEqual(loaded["payload"], payload)

        record = artifact_registry.find_by_checksum("stage", key)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.artifact_id, key)

    def test_compute_identity_lookup(self) -> None:
        identity = stage_cache.compute_identity_lookup("spec", "corpus", ENGINE_VERSION)
        self.assertEqual(len(identity), 64)


class OutOfCoreTests(unittest.TestCase):
    def test_iter_text_batches_cover_all_units(self) -> None:
        texts = [f"text {index}" for index in range(23)]
        unit_ids = [f"u{index}" for index in range(23)]
        seen_ids: list[str] = []
        seen_texts: list[str] = []

        for batch_ids, batch_texts in iter_text_batches(texts, batch_size=5, unit_ids=unit_ids):
            seen_ids.extend(batch_ids)
            seen_texts.extend(batch_texts)

        self.assertEqual(seen_ids, unit_ids)
        self.assertEqual(seen_texts, texts)

    def test_should_use_out_of_core_threshold(self) -> None:
        self.assertFalse(should_use_out_of_core(10, threshold=50))
        self.assertTrue(should_use_out_of_core(50, threshold=50))

    def test_streaming_token_counts(self) -> None:
        texts = ["hello world", "hello again"]
        config = PreprocessingConfig().to_dict()
        counts = streaming_token_counts(texts, config, batch_size=1)
        expected = tokenize(texts[0], config) + tokenize(texts[1], config)
        for token in expected:
            self.assertEqual(counts[token], expected.count(token))

    def test_hashing_matrix_is_csr_and_batched(self) -> None:
        texts = [f"policy document number {index}" for index in range(12)]
        matrix = build_hashing_matrix(texts, n_features=256, batch_size=5)
        self.assertTrue(sparse.isspmatrix_csr(matrix))
        self.assertEqual(matrix.shape, (12, 256))

    def test_dense_preview_never_materializes_full_matrix(self) -> None:
        matrix = sparse.random(100, 200, density=0.05, format="csr")
        preview = dense_preview(matrix, max_rows=3, max_cols=4)
        self.assertEqual(len(preview), 3)
        self.assertEqual(len(preview[0]), 4)

    def test_should_force_sparse_for_large_cells(self) -> None:
        self.assertTrue(should_force_sparse_only(100, 100, dense_export_limit=5000))
        self.assertFalse(
            should_force_sparse_only(10, 10, dense_export_limit=5000, force_sparse_only=False)
        )

    def test_recommend_helpers(self) -> None:
        self.assertEqual(recommend_clustering_algorithm(10, "auto"), "kmeans")
        self.assertEqual(recommend_clustering_algorithm(100, "auto"), "minibatch_kmeans")
        self.assertEqual(recommend_vectorizer_mode(10, "auto"), "count")
        self.assertEqual(recommend_vectorizer_mode(100, "auto"), "hashing")

    def test_duplicate_methods_skip_lexical_at_scale(self) -> None:
        methods, notes = duplicate_methods_for_scale(["exact", "lexical"], 100)
        self.assertNotIn("lexical", methods)
        self.assertIn("minhash", methods)
        self.assertTrue(notes)

    def test_sparse_payload_truncation(self) -> None:
        matrix = sparse.random(20, 50, density=0.5, format="csr")
        payload = sparse_payload_from_matrix(matrix, nnz_cap=10)
        self.assertTrue(payload["truncated"])
        self.assertEqual(len(payload["data"]), 10)


class PrepareTextsLargePathTests(unittest.TestCase):
    def test_prepare_texts_delegates_to_batched_path(self) -> None:
        texts = [f"document number {index} about policy" for index in range(6)]
        config = PreprocessingConfig().to_dict()

        with mock.patch(
            "backend.modules.text_research.infrastructure.out_of_core.should_use_out_of_core",
            return_value=True,
        ):
            batched = prepare_texts(texts, config)

        in_memory = prepare_texts(texts, config, force_in_memory=True)
        self.assertEqual(
            [list(seq) for seq in batched.token_sequences],
            [list(seq) for seq in in_memory.token_sequences],
        )
        self.assertIn("out_of_core", batched.provenance)

    def test_prepare_texts_batched_matches_tokenize(self) -> None:
        texts = ["The Policy, IS Universal!", "Education matters."]
        config = PreprocessingConfig().to_dict()
        artifact = prepare_texts_batched(texts, config, batch_size=1)
        expected = [tokenize(text, config) for text in texts]
        self.assertEqual([list(seq) for seq in artifact.token_sequences], expected)


class FeatureCacheSpillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        os.environ["RESEARCH_ARTIFACT_DIR"] = self.temp_dir.name
        feature_cache.clear_cache()

    def tearDown(self) -> None:
        os.environ.pop("RESEARCH_ARTIFACT_DIR", None)
        feature_cache.clear_cache()

    def test_large_token_lists_spill_and_reload(self) -> None:
        tokens = [[f"t{i}", "word"] for i in range(60)]
        feature_cache.set_cached("spill-key", tokens)
        loaded = feature_cache.get_cached("spill-key")
        self.assertEqual(loaded, tokens)
        with feature_cache._LOCK:
            stored = feature_cache._CACHE["spill-key"]
        self.assertIn("__spill__", stored)


class HashingDfmAndClusteringTests(unittest.TestCase):
    def test_hashing_dfm_batches_without_copying_the_full_input_sequence(self) -> None:
        class SliceOnlyTexts(Sequence[str]):
            def __init__(self, values: list[str]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

            def __iter__(self):
                raise AssertionError("full input iteration would copy the corpus before batching")

        matrix = build_hashing_matrix(
            SliceOnlyTexts(["policy reform", "trade policy"]), batch_size=1
        )

        self.assertEqual(matrix.shape[0], 2)

    def test_hashing_dfm_stays_sparse(self) -> None:
        tokenized = [["policy", "reform"], ["trade", "market"], ["policy", "vote"]]
        result = quantitative.build_dfm_matrix(
            tokenized,
            mode="count",
            vectorizer_mode="hashing",
            hashing_n_features=128,
            force_sparse_only=True,
        )
        self.assertEqual(result["vectorizer_mode"], "hashing")
        self.assertEqual(result["storage"], "sparse")
        self.assertNotIn("dense_matrix", result)
        self.assertEqual(result["dimensions"]["features"], 128)

    def test_clustering_auto_selects_minibatch_for_large_n(self) -> None:
        texts = [f"document about topic {index % 3} content" for index in range(55)]
        unit_ids = [f"u{i}" for i in range(55)]
        result = clustering.run_clustering(
            texts, unit_ids, n_clusters=3, algorithm="auto", vectorizer_mode="tfidf"
        )
        self.assertEqual(result["algorithm"], "minibatch_kmeans")


class ParquetArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_unit_table_roundtrip(self) -> None:
        rows = [
            {"unit_id": "u1", "text": "alpha", "token_count": 1},
            {"unit_id": "u2", "text": "beta", "token_count": 2},
        ]
        meta = save_unit_table(self.root / "units", rows)
        loaded = load_unit_table(meta["path"])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["unit_id"], "u1")
        if parquet_available():
            self.assertEqual(meta["format"], "parquet")
        else:
            self.assertEqual(meta["format"], "jsonl")

    def test_sparse_matrix_roundtrip(self) -> None:
        matrix = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        features = ["a", "b", "c"]
        meta = save_sparse_matrix(self.root / "matrix", matrix, features)
        loaded_matrix, loaded_features = load_sparse_matrix(self.root / "matrix")
        self.assertEqual(loaded_features, features)
        np.testing.assert_array_equal(loaded_matrix.toarray(), matrix.toarray())
        if parquet_available():
            self.assertEqual(meta["format"], "parquet_coo")
        else:
            self.assertEqual(meta["format"], "npz")

    def test_prepared_artifact_summary(self) -> None:
        texts = ["hello world", "education policy"]
        prepared = prepare_texts(texts, PreprocessingConfig().to_dict(), force_in_memory=True)
        meta = save_prepared_artifact_summary(self.root / "summary", prepared)
        self.assertIn("path", meta)
        self.assertIn(meta["format"], {"parquet", "json"})
