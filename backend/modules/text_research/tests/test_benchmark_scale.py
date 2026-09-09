"""Phase 17: lightweight reproducible performance benchmarks.

Scales:
  * micro / 1_000 — always on in CI (soft latency budgets only)
  * 10_000 — on in CI by default (``BENCHMARK_INCLUDE_10K=1``)
  * 100_000 — opt-in via ``BENCHMARK_INCLUDE_100K=1``
  * 1_000_000 annotation cells — opt-in via ``BENCHMARK_INCLUDE_1M=1``

These are regression measurements, not hard SLO gates. Soft budgets are
generous so CI stays non-flaky; wall times are recorded under
``BENCHMARK_REPORT_PATH`` for trend comparison.
"""

from __future__ import annotations

import json
import os
import resource
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.modules.text_research.application.prediction_service import _prediction_row
from backend.modules.text_research.domain.models import AnalysisRun
from backend.modules.text_research.infrastructure import quantitative, run_events, stage_cache
from backend.modules.text_research.infrastructure.classifiers import fit_text_classifier
from backend.modules.text_research.infrastructure.out_of_core import (
    build_hashing_matrix,
    should_use_out_of_core,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.reliability import (
    bootstrap_unit_statistic,
    cohens_kappa,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    raw_agreement,
)


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


def _million_annotations_enabled() -> bool:
    return os.environ.get("BENCHMARK_INCLUDE_1M", "0") == "1"


def synthetic_texts(n_units: int) -> list[str]:
    """Deterministic short documents for throughput benchmarks."""
    topics = ("education", "trade", "health", "climate", "governance")
    return [
        f"Document {index} discusses {topics[index % len(topics)]} policy reform "
        f"and institutional change in region {index % 17}."
        for index in range(n_units)
    ]


def synthetic_coder_pairs(n_units: int, *, seed: int = 0) -> tuple[list[str], list[str]]:
    """Two-coder nominal labels with ~70% agreement (deterministic)."""
    labels_a: list[str] = []
    labels_b: list[str] = []
    categories = ("yes", "no", "maybe")
    for index in range(n_units):
        value = categories[(index + seed) % len(categories)]
        labels_a.append(value)
        labels_b.append(value if (index + seed) % 10 < 7 else categories[(index + 1) % 3])
    return labels_a, labels_b


def synthetic_fleiss_matrix(n_units: int, n_coders: int = 3) -> list[list[str]]:
    categories = ("yes", "no")
    matrix: list[list[str]] = []
    for unit in range(n_units):
        row = [categories[(unit + coder) % 2] for coder in range(n_coders)]
        matrix.append(row)
    return matrix


def _record(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []
    existing.append(row)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def _report_path() -> Path:
    return Path(
        os.environ.get(
            "BENCHMARK_REPORT_PATH",
            "var/benchmarks/text_research_scale.json",
        )
    )


def _base_row(benchmark: str, **fields: object) -> dict:
    return {
        "benchmark": benchmark,
        "engine_version": ENGINE_VERSION,
        "git_sha_env": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT"),
        "peak_rss_mb": _peak_rss_mb(),
        **fields,
    }


class ScaleBenchmarkTests(unittest.TestCase):
    """Corpus prepare + DFM throughput at 1k / 10k / optional 100k."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report_path = _report_path()
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
        token_count = sum(len(seq) for seq in prepared.token_sequences)
        tokens_per_sec = token_count / prepare_seconds if prepare_seconds > 0 else None

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

        row = _base_row(
            "scale_prepare_dfm",
            n_units=n_units,
            prepare_seconds=round(prepare_seconds, 4),
            tokenization_tokens=token_count,
            tokenization_tokens_per_sec=round(tokens_per_sec, 2) if tokens_per_sec else None,
            cache_seconds=round(cache_seconds, 4),
            dfm_seconds=round(dfm_seconds, 4),
            total_seconds=round(prepare_seconds + dfm_seconds, 4),
            peak_rss_mb_before=rss_before,
            peak_rss_mb_after=rss_after,
            corpus_checksum=prepared.corpus_checksum,
            pipeline_checksum=prepared.pipeline_checksum,
            cache_reuse_observed=cache_reuse,
            out_of_core=should_use_out_of_core(n_units),
            dfm_units=dfm["dimensions"]["units"],
            dfm_features=dfm["dimensions"]["features"],
            dfm_nnz=dfm.get("nnz"),
        )
        _record(self.report_path, row)

        self.assertEqual(len(prepared.unit_ids), n_units)
        self.assertEqual(dfm["dimensions"]["units"], n_units)
        # Soft latency budgets — fail only on pathological regressions.
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


class WorkflowBenchmarkTests(unittest.TestCase):
    """Representative workflow micro-benchmarks (always-on soft budgets)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report_path = _report_path()

    def test_reliability_two_coder_compute(self) -> None:
        n_units = 5_000
        labels_a, labels_b = synthetic_coder_pairs(n_units)
        pairs = list(zip(labels_a, labels_b, strict=True))

        started = time.perf_counter()
        kappa = cohens_kappa(labels_a, labels_b)
        alpha = krippendorff_alpha_nominal([[a, b] for a, b in pairs])
        agreement = raw_agreement(labels_a, labels_b)

        def _kappa_stat(sample: list[tuple[str, str]]) -> float | None:
            result = cohens_kappa([a for a, _ in sample], [b for _, b in sample])
            return result.get("kappa")

        ci = bootstrap_unit_statistic(
            pairs,
            _kappa_stat,
            bootstrap_samples=200,
            confidence_level=0.95,
            random_seed=42,
        )
        seconds = time.perf_counter() - started

        _record(
            self.report_path,
            _base_row(
                "reliability_two_coder",
                n_units=n_units,
                n_annotations=n_units * 2,
                seconds=round(seconds, 4),
                kappa=kappa.get("kappa"),
                alpha=alpha.get("alpha"),
                raw_agreement=agreement,
                bootstrap_samples=200,
                ci_lower=(ci or {}).get("lower"),
                ci_upper=(ci or {}).get("upper"),
            ),
        )
        self.assertIsNotNone(kappa.get("kappa"))
        self.assertIsNotNone(alpha.get("alpha"))
        self.assertLess(seconds, 20.0)

    def test_reliability_three_coder_fleiss(self) -> None:
        n_units = 2_000
        matrix = synthetic_fleiss_matrix(n_units, n_coders=3)
        started = time.perf_counter()
        fleiss = fleiss_kappa(matrix)
        alpha = krippendorff_alpha_nominal(matrix)
        seconds = time.perf_counter() - started
        _record(
            self.report_path,
            _base_row(
                "reliability_three_coder",
                n_units=n_units,
                n_annotations=n_units * 3,
                seconds=round(seconds, 4),
                fleiss_kappa=fleiss.get("kappa"),
                alpha=alpha.get("alpha"),
            ),
        )
        self.assertIsNotNone(fleiss.get("kappa"))
        self.assertLess(seconds, 10.0)

    def test_reliability_one_million_annotations_opt_in(self) -> None:
        if not _million_annotations_enabled():
            self.skipTest("set BENCHMARK_INCLUDE_1M=1 for 1M annotation reliability bench")
        # 500k shared units × 2 coders = 1M annotation cells
        n_units = 500_000
        labels_a, labels_b = synthetic_coder_pairs(n_units)
        started = time.perf_counter()
        kappa = cohens_kappa(labels_a, labels_b)
        seconds = time.perf_counter() - started
        _record(
            self.report_path,
            _base_row(
                "reliability_1m_annotations",
                n_units=n_units,
                n_annotations=n_units * 2,
                seconds=round(seconds, 4),
                kappa=kappa.get("kappa"),
            ),
        )
        self.assertIsNotNone(kappa.get("kappa"))

    def test_classification_preparation(self) -> None:
        train_x = synthetic_texts(400)
        test_x = synthetic_texts(100)
        train_y = ["yes" if i % 2 == 0 else "no" for i in range(len(train_x))]
        test_y = ["yes" if i % 3 == 0 else "no" for i in range(len(test_x))]
        started = time.perf_counter()
        result = fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            algorithm="logistic_regression",
            random_seed=7,
            tune_thresholds=False,
            n_bootstrap=10,
        )
        seconds = time.perf_counter() - started
        feature_space = result.get("feature_space") or {}
        _record(
            self.report_path,
            _base_row(
                "classification_preparation",
                n_train=len(train_x),
                n_test=len(test_x),
                seconds=round(seconds, 4),
                vocabulary_size=result.get("vocabulary_size"),
                feature_space=feature_space,
            ),
        )
        self.assertIn("metrics", result)
        self.assertLess(seconds, 45.0)

    def test_prediction_row_serialization(self) -> None:
        n_rows = 50_000
        started = time.perf_counter()
        rows = [
            _prediction_row(
                model_id="model-bench",
                unit_id=f"u-{index}",
                task_type="binary",
                label_names=["no", "yes"],
                prediction={
                    "prediction": index % 2,
                    "probability": 0.5 + (index % 50) / 100.0,
                    "uncertainty": 0.1,
                },
            )
            for index in range(n_rows)
        ]
        payload = json.dumps(rows)
        seconds = time.perf_counter() - started
        _record(
            self.report_path,
            _base_row(
                "prediction_insertion_serialize",
                n_rows=n_rows,
                seconds=round(seconds, 4),
                payload_bytes=len(payload.encode("utf-8")),
                rows_per_sec=round(n_rows / seconds, 2) if seconds > 0 else None,
            ),
        )
        self.assertEqual(len(rows), n_rows)
        self.assertLess(seconds, 15.0)

    def test_run_event_serialize_throughput(self) -> None:
        n_events = 5_000
        run = AnalysisRun(
            id="run-bench",
            project_id="proj-1",
            corpus_id="corp-1",
            run_type="topic_model",
            status="running",
            progress_stage="training",
            created_by="user-1",
        )
        started = time.perf_counter()
        previous = None
        names: list[str] = []
        with patch.object(run_events, "_get_sync_redis", return_value=None):
            for index in range(n_events):
                run.progress_stage = f"stage-{index % 20}"
                name = run_events.publish_run_event(
                    run,
                    previous=previous,
                )
                names.append(name)
                previous = {
                    "status": run.status,
                    "progress_stage": run.progress_stage,
                    "artifact_path": run.artifact_path,
                }
        seconds = time.perf_counter() - started
        _record(
            self.report_path,
            _base_row(
                "run_event_serialize",
                n_events=n_events,
                seconds=round(seconds, 4),
                events_per_sec=round(n_events / seconds, 2) if seconds > 0 else None,
                sample_event=names[0] if names else None,
            ),
        )
        self.assertEqual(len(names), n_events)
        self.assertLess(seconds, 10.0)

    def test_cache_hit_miss_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["RESEARCH_ARTIFACT_DIR"] = temp_dir
            stage_cache.invalidate()
            stage_cache.reset_redis_client_for_tests()
            key_a = stage_cache.stage_cache_key(
                engine_version=ENGINE_VERSION,
                stage_name="prepare_corpus",
                input_checksum="bench-a",
                spec_hash="spec-a",
                preprocessing_config={"lowercase": True},
            )
            key_b = stage_cache.stage_cache_key(
                engine_version=ENGINE_VERSION,
                stage_name="prepare_corpus",
                input_checksum="bench-b",
                spec_hash="spec-a",
                preprocessing_config={"lowercase": True},
            )

            miss_started = time.perf_counter()
            miss = stage_cache.get_stage(key_a)
            miss_seconds = time.perf_counter() - miss_started
            self.assertIsNone(miss)

            stage_cache.put_stage(
                key_a,
                meta={"stage_name": "prepare_corpus"},
                payload={"n": 42},
                payload_format="json",
            )
            hit_started = time.perf_counter()
            hit = stage_cache.get_stage(key_a)
            hit_seconds = time.perf_counter() - hit_started
            self.assertIsNotNone(hit)

            other_miss_started = time.perf_counter()
            other = stage_cache.get_stage(key_b)
            other_miss_seconds = time.perf_counter() - other_miss_started
            self.assertIsNone(other)

            _record(
                self.report_path,
                _base_row(
                    "cache_hit_miss_latency",
                    miss_seconds=round(miss_seconds, 6),
                    hit_seconds=round(hit_seconds, 6),
                    changed_input_miss_seconds=round(other_miss_seconds, 6),
                ),
            )
            # Soft: hit should be sub-second on local disk/L1.
            self.assertLess(hit_seconds, 1.0)


class FixtureContractTests(unittest.TestCase):
    def test_synthetic_texts_are_deterministic(self) -> None:
        self.assertEqual(synthetic_texts(5), synthetic_texts(5))
        self.assertEqual(len(synthetic_texts(100)), 100)

    def test_synthetic_coder_pairs_length(self) -> None:
        a, b = synthetic_coder_pairs(50)
        self.assertEqual(len(a), 50)
        self.assertEqual(len(b), 50)


if __name__ == "__main__":
    unittest.main()
