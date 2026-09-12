from backend.modules.rag.eval.load_benchmark import (
    LoadScale,
    memory_ceiling_smoke,
    synthetic_chunk_rows,
)


def test_synthetic_load_rows_stream_deterministically():
    rows = list(synthetic_chunk_rows(LoadScale(documents=2, chunks=3), words_per_chunk=4))
    assert [row["document_id"] for row in rows] == ["doc-0", "doc-1", "doc-0"]


def test_memory_ceiling_smoke_on_synthetic_sizes():
    for chunks in (10_000, 100_000, 1_000_000):
        report = memory_ceiling_smoke(
            LoadScale(documents=max(1, chunks // 100), chunks=chunks),
            max_materialized_rows=128,
            max_estimated_bytes=5_000_000,
        )
        assert report["within_ceiling"]
        assert report["sampled_rows"] <= 128
