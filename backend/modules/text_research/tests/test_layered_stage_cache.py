"""Phase 5: layered stage cache (L1/L2/L3) + stampede locks."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from backend.modules.text_research.infrastructure import stage_cache
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION


class FakeRedis:
    """Minimal sync Redis stand-in for lock/meta/status tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def setex(self, key: str, _ttl: int, value: str) -> bool:
        self.store[key] = value
        return True

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if self.store.pop(key, None) is not None:
                deleted += 1
        return deleted

    def eval(self, _script: str, _numkeys: int, key: str, token: str) -> int:
        if self.store.get(key) == token:
            self.store.pop(key, None)
            return 1
        return 0


class LayeredStageCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        os.environ["RESEARCH_ARTIFACT_DIR"] = self.temp_dir.name
        stage_cache.invalidate()
        stage_cache.reset_redis_client_for_tests()
        self.fake_redis = FakeRedis()

    def tearDown(self) -> None:
        stage_cache.invalidate()
        stage_cache.reset_redis_client_for_tests()
        os.environ.pop("RESEARCH_ARTIFACT_DIR", None)

    def _key(self, stage: str = "prepare_corpus", **params) -> str:
        return stage_cache.stage_cache_key(
            engine_version=ENGINE_VERSION,
            stage_name=stage,
            input_checksum="abc",
            spec_hash="def",
            params=params or {"batch_size": 1},
            preprocessing_config={"lowercase": True},
        )

    def test_l1_hit_after_put(self) -> None:
        key = self._key()
        stage_cache.put_stage(
            key,
            meta={"stage_name": "prepare_corpus"},
            payload={"n": 3},
            payload_format="json",
        )
        metrics = stage_cache.cache_layer_metrics()
        self.assertGreaterEqual(metrics["l1_entries"], 1)
        loaded = stage_cache.get_stage(key)
        assert loaded is not None
        self.assertEqual(loaded["payload"], {"n": 3})

    def test_redis_meta_published(self) -> None:
        key = self._key()
        with (
            patch.object(stage_cache, "_redis_enabled", return_value=True),
            patch.object(stage_cache, "_get_sync_redis", return_value=self.fake_redis),
        ):
            stage_cache.put_stage(
                key,
                meta={"stage_name": "prepare_corpus"},
                payload={"ok": True},
                payload_format="json",
            )
            raw = self.fake_redis.get(stage_cache._meta_redis_key(key))
            self.assertIsNotNone(raw)
            self.assertEqual(self.fake_redis.get(stage_cache._status_redis_key(key)), "ready")
            # Clear L1 + local disk to force Redis pointer path... keep disk for payload.
            stage_cache._l1_delete(key)
            loaded = stage_cache.get_stage(key)
            assert loaded is not None
            self.assertEqual(loaded["payload"], {"ok": True})

    def test_get_or_compute_runs_once_under_contention(self) -> None:
        key = self._key(stage="frequencies")
        calls: list[int] = []
        barrier = threading.Barrier(3)
        results: list[dict] = []

        def factory():
            calls.append(1)
            return {"stage_name": "frequencies"}, {"terms": ["a"]}

        def worker():
            barrier.wait()
            with (
                patch.object(stage_cache, "_redis_enabled", return_value=True),
                patch.object(stage_cache, "_get_sync_redis", return_value=self.fake_redis),
                patch.object(stage_cache, "_lock_wait", return_value=2.0),
            ):
                results.append(
                    stage_cache.get_or_compute(key, factory, payload_format="json")
                )

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["payload"], {"terms": ["a"]})

    def test_refuses_supervised_stage_key(self) -> None:
        with self.assertRaises(ValueError):
            stage_cache.stage_cache_key(
                engine_version="1",
                stage_name="supervised_vectorizer",
                input_checksum="x",
                spec_hash="y",
            )

    def test_refuses_supervised_put(self) -> None:
        key = self._key(stage="frequencies")
        with self.assertRaises(ValueError):
            stage_cache.put_stage(
                key,
                meta={"stage_name": "train_fitted_tfidf"},
                payload={"bad": True},
            )

    def test_preprocessing_changes_identity(self) -> None:
        a = stage_cache.stage_cache_key(
            engine_version="1",
            stage_name="prepared_corpus",
            input_checksum="c",
            spec_hash="s",
            preprocessing_config={"lowercase": True},
        )
        b = stage_cache.stage_cache_key(
            engine_version="1",
            stage_name="prepared_corpus",
            input_checksum="c",
            spec_hash="s",
            preprocessing_config={"lowercase": False},
        )
        self.assertNotEqual(a, b)

    def test_input_checksum_change_misses(self) -> None:
        a = stage_cache.stage_cache_key(
            engine_version="1",
            stage_name="prepared_corpus",
            input_checksum="snapshot-a",
            spec_hash="s",
            preprocessing_config={"lowercase": True},
        )
        b = stage_cache.stage_cache_key(
            engine_version="1",
            stage_name="prepared_corpus",
            input_checksum="snapshot-b",
            spec_hash="s",
            preprocessing_config={"lowercase": True},
        )
        self.assertNotEqual(a, b)
        stage_cache.put_stage(
            a,
            meta={"stage_name": "prepared_corpus"},
            payload={"docs": 1},
            payload_format="json",
        )
        self.assertIsNotNone(stage_cache.get_stage(a))
        self.assertIsNone(stage_cache.get_stage(b))

    def test_engine_version_change_misses(self) -> None:
        a = stage_cache.stage_cache_key(
            engine_version="engine-1",
            stage_name="prepared_corpus",
            input_checksum="c",
            spec_hash="s",
        )
        b = stage_cache.stage_cache_key(
            engine_version="engine-2",
            stage_name="prepared_corpus",
            input_checksum="c",
            spec_hash="s",
        )
        self.assertNotEqual(a, b)
        stage_cache.put_stage(
            a,
            meta={"stage_name": "prepared_corpus"},
            payload={"docs": 2},
            payload_format="json",
        )
        self.assertIsNotNone(stage_cache.get_stage(a))
        self.assertIsNone(stage_cache.get_stage(b))

    def test_distributed_lock_releases_token(self) -> None:
        key = "deadbeef"
        with (
            patch.object(stage_cache, "_redis_enabled", return_value=True),
            patch.object(stage_cache, "_get_sync_redis", return_value=self.fake_redis),
            patch.object(stage_cache, "_lock_wait", return_value=1.0),
        ):
            with stage_cache.distributed_lock(key) as held:
                self.assertTrue(held)
                self.assertIn(stage_cache._lock_redis_key(key), self.fake_redis.store)
            self.assertNotIn(stage_cache._lock_redis_key(key), self.fake_redis.store)

    def test_object_storage_backend_flag(self) -> None:
        key = self._key(stage="topic_artifacts", unique=1)
        stage_cache.invalidate(key)
        self.assertFalse(stage_cache.has_stage(key))
        with patch.object(
            stage_cache,
            "_persist_object_payload",
            return_value={
                "storage_backend": "object",
                "object_key": f"research-stage-cache/{key}/payload.json",
                "object_uri": f"s3://bucket/research-stage-cache/{key}/payload.json",
            },
        ) as persist:
            path = stage_cache.put_stage(
                key,
                meta={"stage_name": "topic_artifacts"},
                payload={"x": 1},
                payload_format="json",
            )
            self.assertTrue(path)
            persist.assert_called_once()
            loaded = stage_cache.get_stage(key)
            assert loaded is not None
            self.assertEqual(loaded.get("storage_backend"), "object")
            self.assertTrue(str(loaded.get("object_key", "")).startswith("research-stage-cache/"))


if __name__ == "__main__":
    unittest.main()
