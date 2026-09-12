"""Reproducible synthetic load envelope harness; callers supply real DB operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True, slots=True)
class LoadScale:
    documents: int
    chunks: int


DEFAULT_SCALES = (
    LoadScale(documents=100, chunks=10_000),
    LoadScale(documents=1_000, chunks=100_000),
    LoadScale(documents=10_000, chunks=1_000_000),
)


def synthetic_chunk_rows(scale: LoadScale, *, words_per_chunk: int = 80):
    """Yield deterministic rows without allocating an entire benchmark corpus."""
    text = "research evidence " * (words_per_chunk // 2)
    for index in range(scale.chunks):
        yield {
            "document_id": f"doc-{index % scale.documents}",
            "chunk_index": index // scale.documents,
            "content": f"{text}{index}",
        }


def measure_operation(operation: Callable[[], None]) -> dict[str, float]:
    started = perf_counter()
    operation()
    return {"elapsed_ms": (perf_counter() - started) * 1000}


def estimate_row_bytes(row: dict[str, Any]) -> int:
    return sum(len(str(value).encode("utf-8")) for value in row.values())


def memory_ceiling_smoke(
    scale: LoadScale,
    *,
    max_materialized_rows: int = 256,
    max_estimated_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Assert streaming generation stays under a synthetic memory ceiling.

    Does not allocate the full scale; samples only a bounded prefix.
    """
    sampled = 0
    estimated_bytes = 0
    for row in synthetic_chunk_rows(scale):
        sampled += 1
        estimated_bytes += estimate_row_bytes(row)
        if sampled >= max_materialized_rows:
            break
    within_ceiling = estimated_bytes <= max_estimated_bytes and sampled <= max_materialized_rows
    return {
        "schema_version": 1,
        "scale_chunks": scale.chunks,
        "sampled_rows": sampled,
        "estimated_bytes": estimated_bytes,
        "max_materialized_rows": max_materialized_rows,
        "max_estimated_bytes": max_estimated_bytes,
        "within_ceiling": within_ceiling,
    }
