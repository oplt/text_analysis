"""Parquet/Arrow (or JSONL/npz) persistence for large intermediate artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact

# Peak memory for streaming writers scales with this, not total row count.
DEFAULT_PARQUET_ROW_BATCH = 50_000


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str))
            handle.write("\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_row_batches(
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> Iterator[list[dict[str, Any]]]:
    """Yield bounded row batches without materializing the full iterable."""
    size = max(1, int(batch_size))
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(dict(row))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_column_slices(
    columns: Mapping[str, Sequence[Any]],
    *,
    batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> Iterator[dict[str, list[Any]]]:
    """Yield columnar slices of already-materialized columns."""
    if not columns:
        return
    size = max(1, int(batch_size))
    keys = list(columns)
    n_rows = len(columns[keys[0]])
    for key in keys[1:]:
        if len(columns[key]) != n_rows:
            raise ValueError("column lengths must match for parquet batching")
    for start in range(0, n_rows, size):
        end = min(start + size, n_rows)
        yield {key: list(columns[key][start:end]) for key in keys}


def iter_token_column_batches(
    unit_ids: Sequence[str],
    token_sequences: Sequence[Sequence[str]],
    *,
    batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> Iterator[dict[str, list[Any]]]:
    """Flatten tokens into bounded columnar batches (no full-column materialization)."""
    if len(unit_ids) != len(token_sequences):
        raise ValueError("unit_ids must align 1:1 with token_sequences")
    size = max(1, int(batch_size))
    out_unit_ids: list[str] = []
    positions: list[int] = []
    tokens: list[str] = []
    for unit_id, sequence in zip(unit_ids, token_sequences, strict=True):
        for position, token in enumerate(sequence):
            out_unit_ids.append(str(unit_id))
            positions.append(int(position))
            tokens.append(str(token))
            if len(tokens) >= size:
                yield {
                    "unit_id": out_unit_ids,
                    "token_position": positions,
                    "token": tokens,
                }
                out_unit_ids, positions, tokens = [], [], []
    if tokens:
        yield {
            "unit_id": out_unit_ids,
            "token_position": positions,
            "token": tokens,
        }


def write_parquet_column_batches(
    path: str | Path,
    batches: Iterable[Mapping[str, Sequence[Any]]],
    *,
    empty_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream columnar batches to Parquet via ``ParquetWriter``.

    Peak memory tracks the current batch only. Callers must not pass an
    iterable that internally materializes all rows (e.g. ``[full_table]``).
    """
    if not parquet_available():
        raise RuntimeError("pyarrow is required for streaming parquet writes")

    import pyarrow as pa
    import pyarrow.parquet as pq

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    target = resolved if resolved.suffix == ".parquet" else resolved.with_suffix(".parquet")

    writer: pq.ParquetWriter | None = None
    written = 0
    batch_count = 0
    try:
        for batch in batches:
            batch_count += 1
            table = pa.table({key: values for key, values in batch.items()})
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema)
            writer.write_table(table)
            written += table.num_rows
        if writer is None:
            if empty_schema is not None:
                empty = pa.table(
                    {name: pa.array([], type=dtype) for name, dtype in empty_schema.items()}
                )
                pq.write_table(empty, target)
            else:
                pq.write_table(pa.table({}), target)
            return {
                "format": "parquet",
                "path": str(target),
                "row_count": 0,
                "batch_count": 0,
                "streaming": True,
            }
    finally:
        if writer is not None:
            writer.close()

    return {
        "format": "parquet",
        "path": str(target),
        "row_count": written,
        "batch_count": batch_count,
        "streaming": True,
    }


def write_parquet_row_batches(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> dict[str, Any]:
    """Stream row mappings to Parquet without ``list(rows)``."""
    if not parquet_available():
        raise RuntimeError("pyarrow is required for streaming parquet writes")

    import pyarrow as pa
    import pyarrow.parquet as pq

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    target = resolved if resolved.suffix == ".parquet" else resolved.with_suffix(".parquet")

    writer: pq.ParquetWriter | None = None
    written = 0
    batch_count = 0
    try:
        for batch_rows in iter_row_batches(rows, batch_size=batch_size):
            batch_count += 1
            table = pa.Table.from_pylist(batch_rows)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema)
            writer.write_table(table)
            written += len(batch_rows)
        if writer is None:
            pq.write_table(pa.table({}), target)
            return {
                "format": "parquet",
                "path": str(target),
                "row_count": 0,
                "batch_count": 0,
                "streaming": True,
            }
    finally:
        if writer is not None:
            writer.close()

    return {
        "format": "parquet",
        "path": str(target),
        "row_count": written,
        "batch_count": batch_count,
        "streaming": True,
    }


def save_token_table(
    path: str | Path,
    unit_ids: Sequence[str],
    token_sequences: Sequence[Sequence[str]],
    *,
    batch_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> dict[str, Any]:
    """Persist flattened tokens with bounded-memory Parquet batches."""
    if parquet_available():
        import pyarrow as pa

        return write_parquet_column_batches(
            path,
            iter_token_column_batches(unit_ids, token_sequences, batch_size=batch_size),
            empty_schema={
                "unit_id": pa.string(),
                "token_position": pa.int64(),
                "token": pa.string(),
            },
        )

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    target = resolved if resolved.suffix == ".jsonl" else resolved.with_suffix(".jsonl")

    def _rows() -> Iterator[dict[str, Any]]:
        for unit_id, sequence in zip(unit_ids, token_sequences, strict=True):
            for position, token in enumerate(sequence):
                yield {
                    "unit_id": str(unit_id),
                    "token_position": int(position),
                    "token": str(token),
                }

    count = _write_jsonl(target, _rows())
    return {"format": "jsonl", "path": str(target), "row_count": count, "streaming": True}


def save_unit_table(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    columns: Mapping[str, Sequence[Any]] | None = None,
    chunk_size: int = DEFAULT_PARQUET_ROW_BATCH,
) -> dict[str, Any]:
    """Persist unit rows as parquet when available, otherwise JSONL.

    Prefer streaming:
    * pass an iterable of row mappings (never fully buffered here);
    * or use :func:`save_token_table` / :func:`write_parquet_column_batches`
      for token-scale tables.

    ``columns`` may still hold full arrays in the caller; when parquet is
    available they are written in ``chunk_size`` slices so the Arrow table is
    not built as one giant allocation.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    size = max(1, int(chunk_size))

    if parquet_available():
        if columns is not None:
            return write_parquet_column_batches(
                resolved,
                iter_column_slices(columns, batch_size=size),
                empty_schema=None,
            )
        return write_parquet_row_batches(
            resolved,
            rows or (),
            batch_size=size,
        )

    target = resolved if resolved.suffix == ".jsonl" else resolved.with_suffix(".jsonl")
    if columns is not None:
        keys = list(columns)
        n = len(next(iter(columns.values()), []))

        def _from_columns() -> Iterator[dict[str, Any]]:
            for i in range(n):
                yield {key: columns[key][i] for key in keys}

        count = _write_jsonl(target, _from_columns())
        return {"format": "jsonl", "path": str(target), "row_count": count}

    count = _write_jsonl(target, (dict(row) for row in (rows or ())))
    return {"format": "jsonl", "path": str(target), "row_count": count}


def load_unit_table(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if resolved.suffix == ".parquet" or resolved.with_suffix(".parquet").is_file():
        import pandas as pd

        parquet_path = (
            resolved if resolved.suffix == ".parquet" else resolved.with_suffix(".parquet")
        )
        frame = pd.read_parquet(parquet_path)
        return frame.to_dict(orient="records")

    jsonl_path = resolved if resolved.suffix == ".jsonl" else resolved.with_suffix(".jsonl")
    return _read_jsonl(jsonl_path)


def save_sparse_matrix(
    path_stem: str | Path,
    matrix: sparse.spmatrix,
    feature_names: list[str],
) -> dict[str, Any]:
    """Persist a sparse matrix with feature names (parquet COO or npz fallback)."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    coo = matrix.tocoo()

    if parquet_available():
        import pandas as pd

        target = stem.with_suffix(".parquet")
        frame = pd.DataFrame(
            {
                "row": coo.row.astype(np.int64),
                "col": coo.col.astype(np.int64),
                "data": coo.data,
            }
        )
        frame.to_parquet(target, index=False)
        meta_path = stem.with_suffix(".features.json")
        meta_path.write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "n_rows": int(matrix.shape[0]),
                    "n_cols": int(matrix.shape[1]),
                },
                sort_keys=True,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return {
            "format": "parquet_coo",
            "matrix_path": str(target),
            "features_path": str(meta_path),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        }

    matrix_path = stem.with_suffix(".npz")
    sparse.save_npz(matrix_path, matrix)
    features_path = stem.with_suffix(".features.json")
    features_path.write_text(
        json.dumps({"feature_names": feature_names}, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return {
        "format": "npz",
        "matrix_path": str(matrix_path),
        "features_path": str(features_path),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }


def load_sparse_matrix(path_stem: str | Path) -> tuple[sparse.spmatrix, list[str]]:
    stem = Path(path_stem)
    parquet_path = stem.with_suffix(".parquet")
    if parquet_path.is_file():
        import pandas as pd

        frame = pd.read_parquet(parquet_path)
        meta = json.loads(stem.with_suffix(".features.json").read_text(encoding="utf-8"))
        n_rows = int(meta["n_rows"])
        n_cols = int(meta["n_cols"])
        matrix = sparse.coo_matrix(
            (frame["data"].to_numpy(), (frame["row"].to_numpy(), frame["col"].to_numpy())),
            shape=(n_rows, n_cols),
        )
        return matrix, list(meta["feature_names"])

    matrix_path = stem.with_suffix(".npz")
    features_path = stem.with_suffix(".features.json")
    matrix = sparse.load_npz(matrix_path)
    meta = json.loads(features_path.read_text(encoding="utf-8"))
    return matrix, list(meta["feature_names"])


def save_prepared_artifact_summary(
    path: str | Path,
    prepared: PreparedCorpusArtifact,
    *,
    include_tokens: bool = False,
) -> dict[str, Any]:
    """Store a compact prepared-corpus summary (optionally including token sequences)."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    token_counts = [len(seq) for seq in prepared.token_sequences]
    summary: dict[str, Any] = {
        "unit_ids": list(prepared.unit_ids),
        "corpus_checksum": prepared.corpus_checksum,
        "pipeline_checksum": prepared.pipeline_checksum,
        "unit_count": len(prepared.unit_ids),
        "vocabulary_sample": list(prepared.vocabulary[:100]),
        "vocabulary_size": len(prepared.vocabulary),
        "token_counts": token_counts,
        "total_tokens": sum(token_counts),
    }
    if include_tokens:
        summary["token_sequences"] = [list(seq) for seq in prepared.token_sequences]

    if parquet_available():
        import pandas as pd

        target = resolved if resolved.suffix == ".parquet" else resolved.with_suffix(".parquet")
        pd.DataFrame([summary]).to_parquet(target, index=False)
        return {"format": "parquet", "path": str(target)}

    target = resolved if resolved.suffix == ".json" else resolved.with_suffix(".json")
    target.write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return {"format": "json", "path": str(target)}
