"""Synthetic scale fixtures and performance-sensitive benchmarks (§24).

Scales:
  * 1_000  — always on in CI
  * 10_000 — on in CI (default)
  * 100_000 — opt-in via ``BENCHMARK_INCLUDE_100K=1`` (nightly / manual)

Each case records wall time, peak RSS (best-effort), and stage-cache reuse.
"""

from __future__ import annotations

import os
import resource
import time
import unittest
from pathlib import Path

from backend.modules.text_research.infrastructure import quantitative, stage_cache
from backend.modules.text_research.infrastructure.out_of_core import (
    build_hashing_matrix,
    should_use_out_of_core,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


def _peak_rss_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        if usage > 10_000_000:
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return None


def _scale_enabled(n_units: int) -> bool:
    if n_units <= 1_000:
        return True
    if n_units <= 10_000:
        return os.environ.get("BENCHMARK_INCLUDE_10K", "1") == "1"
    return os.environ.get("BENCHMARK_INCLUDE_100K", "0") == "1"


def synthetic_texts(n_units: int) -> list[str]:
    """Deterministic short documents for throughput benchmarks."""
    topics = ("education", "trade", "health", "climate", "governance")
    return [
        f"Document {index} discusses {topics[index % len(topics)]} policy reform "
        f"and institutional change in region {index % 17}."
        for index in range(n_units)
    ]


def _record(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    existing: list[dict] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(row)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


class ScaleBenchmarkTests(unittest.TestCase):
    """Performance-sensitive gates for current-HEAD CI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report_path = Path(
            os.environ.get(
                "BENCHMARK_REPORT_PATH",
                "var/benchmarks/text_research_scale.json",
            )
        )
        cls.config = PreprocessingConfig(lowercase=True, remove_stopwords=False).to_dict()

    def _run_scale(self, n_units: int) -> None:
        if not _scale_enabled(n_units):
            self.skipTest(f"scale {n_units} disabled by BENCHMARK_INCLUDE_* env")

        texts = synthetic_texts(n_units)
        stage_cache.invalidate()
        os.environ.setdefault("RESEARCH_ARTIFACT_DIR", "var/research_artifacts_bench")

        started = time.perf_counter()
        rss_before = _peak_rss_mb()
        prepared = prepare_texts(texts, self.config, force_in_memory=n_units < 50)
        prepare_seconds = time.perf_counter() - started

        # Second prepare should hit content-addressable stage cache when enabled.
        cache_started = time.perf_counter()
        prepared_again = prepare_texts(texts, self.config, force_in_memory=n_units < 50)
        cache_seconds = time.perf_counter() - cache_started
        cache_reuse = (
            prepared.corpus_checksum == prepared_again.corpus_checksum
            and prepared.pipeline_checksum == prepared_again.pipeline_checksum
            and cache_seconds <= prepare_seconds * 1.05
        )

        dfm_started = time.perf_counter()
        if should_use_out_of_core(n_units) or n_units >= 1_000:
            matrix = build_hashing_matrix(texts, n_features=2**14, batch_size=1_000)
            dfm = {
                "dimensions": {"units": matrix.shape[0], "features": matrix.shape[1]},
                "nnz": int(matrix.nnz),
                "vectorizer_mode": "hashing",
            }
        else:
            tokenized = [list(seq) for seq in prepared.token_sequences]
            dfm = quantitative.build_dfm_matrix(
                tokenized,
                mode="count",
                force_sparse_only=True,
                vectorizer_mode="count",
            )
        dfm_seconds = time.perf_counter() - dfm_started
        rss_after = _peak_rss_mb()

        row = {
            "n_units": n_units,
            "engine_version": ENGINE_VERSION,
            "prepare_seconds": round(prepare_seconds, 4),
            "cache_seconds": round(cache_seconds, 4),
            "dfm_seconds": round(dfm_seconds, 4),
            "total_seconds": round(prepare_seconds + dfm_seconds, 4),
            "peak_rss_mb_before": rss_before,
            "peak_rss_mb_after": rss_after,
            "corpus_checksum": prepared.corpus_checksum,
            "pipeline_checksum": prepared.pipeline_checksum,
            "cache_reuse_observed": cache_reuse,
            "out_of_core": should_use_out_of_core(n_units),
            "dfm_units": dfm["dimensions"]["units"],
            "dfm_features": dfm["dimensions"]["features"],
            "dfm_nnz": dfm.get("nnz"),
            "git_sha_env": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
        }
        _record(self.report_path, row)

        self.assertEqual(len(prepared.unit_ids), n_units)
        self.assertEqual(dfm["dimensions"]["units"], n_units)
        # Soft latency budgets — fail loudly on pathological regressions only.
        if n_units <= 1_000:
            self.assertLess(prepare_seconds + dfm_seconds, 30.0)
        elif n_units <= 10_000:
            self.assertLess(prepare_seconds + dfm_seconds, 120.0)

    def test_benchmark_1k_units(self) -> None:
        self._run_scale(1_000)

    def test_benchmark_10k_units(self) -> None:
        self._run_scale(10_000)

    def test_benchmark_100k_units(self) -> None:
        self._run_scale(100_000)


class FixtureContractTests(unittest.TestCase):
    def test_synthetic_texts_are_deterministic(self) -> None:
        self.assertEqual(synthetic_texts(5), synthetic_texts(5))
        self.assertEqual(len(synthetic_texts(100)), 100)
