"""Streaming / bounded-memory Parquet writers for R serialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.modules.text_research.infrastructure.parquet_artifacts import (
    iter_token_column_batches,
    load_unit_table,
    parquet_available,
    save_token_table,
    save_unit_table,
    write_parquet_row_batches,
)


@unittest.skipUnless(parquet_available(), "pyarrow required")
class StreamingParquetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_token_batches_exceed_one_batch(self) -> None:
        unit_ids = [f"u{i}" for i in range(10)]
        sequences = [[f"t{j}" for j in range(30)] for _ in unit_ids]
        batches = list(iter_token_column_batches(unit_ids, sequences, batch_size=100))
        self.assertEqual(len(batches), 3)
        self.assertEqual(sum(len(batch["token"]) for batch in batches), 300)
        self.assertLessEqual(max(len(batch["token"]) for batch in batches), 100)

    def test_save_token_table_exercises_multiple_batches(self) -> None:
        unit_ids = [f"u{i}" for i in range(5)]
        sequences = [["alpha", "beta"] * 40 for _ in unit_ids]  # 400 tokens
        meta = save_token_table(
            self.root / "tokens.parquet",
            unit_ids,
            sequences,
            batch_size=100,
        )
        self.assertTrue(meta["streaming"])
        self.assertEqual(meta["row_count"], 400)
        self.assertGreaterEqual(meta["batch_count"], 4)
        loaded = load_unit_table(meta["path"])
        self.assertEqual(len(loaded), 400)
        self.assertEqual(loaded[0]["token"], "alpha")
        self.assertEqual(loaded[1]["token_position"], 1)

    def test_row_batch_writer_streams_without_full_materialization(self) -> None:
        def _rows():
            for i in range(250):
                yield {"unit_id": f"u{i}", "n": i}

        meta = write_parquet_row_batches(
            self.root / "rows.parquet",
            _rows(),
            batch_size=100,
        )
        self.assertEqual(meta["row_count"], 250)
        self.assertEqual(meta["batch_count"], 3)
        self.assertTrue(meta["streaming"])

    def test_column_slices_batch_large_unit_table(self) -> None:
        n = 250
        meta = save_unit_table(
            self.root / "units.parquet",
            columns={
                "unit_id": [f"u{i}" for i in range(n)],
                "document_id": [f"d{i}" for i in range(n)],
            },
            chunk_size=100,
        )
        self.assertEqual(meta["row_count"], n)
        self.assertEqual(meta["batch_count"], 3)
        self.assertEqual(len(load_unit_table(meta["path"])), n)

    def test_empty_token_table(self) -> None:
        meta = save_token_table(self.root / "empty.parquet", [], [])
        self.assertEqual(meta["row_count"], 0)
        self.assertEqual(load_unit_table(meta["path"]), [])


@unittest.skipUnless(parquet_available(), "pyarrow required")
class StreamingParquetScaleSmokeTests(unittest.TestCase):
    """Exercises >> one batch without a full 100k-unit memory bench harness."""

    def test_hundred_thousand_token_rows_multi_batch(self) -> None:
        # 2k units × 60 tokens = 120k token rows; batch 25k → ≥5 batches.
        unit_ids = [f"u{i}" for i in range(2_000)]
        sequences = [["w"] * 60 for _ in unit_ids]
        with tempfile.TemporaryDirectory() as tmp:
            meta = save_token_table(
                Path(tmp) / "tokens.parquet",
                unit_ids,
                sequences,
                batch_size=25_000,
            )
            self.assertEqual(meta["row_count"], 120_000)
            self.assertGreaterEqual(meta["batch_count"], 5)
            self.assertTrue(meta["streaming"])


if __name__ == "__main__":
    unittest.main()
