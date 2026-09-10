"""Parquet/Arrow (or JSONL/npz) persistence for large intermediate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from backend.modules.text_research.domain.prepared_corpus import PreparedCorpusArtifact


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_unit_table(
    path: str | Path,
    rows: list[dict[str, Any]] | None = None,
    *,
    columns: dict[str, list[Any]] | None = None,
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    """Persist unit rows as parquet when available, otherwise JSONL.

    Prefer ``columns`` (columnar arrays) to avoid materializing millions of
    per-token Python dicts. When ``rows`` is provided, writes in chunks via
    ``ParquetWriter`` so peak memory stays bounded.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if parquet_available():
        import pyarrow as pa
        import pyarrow.parquet as pq

        target = resolved if resolved.suffix == ".parquet" else resolved.with_suffix(".parquet")
        if columns is not None:
            table = pa.table(columns)
            pq.write_table(table, target)
            return {
                "format": "parquet",
                "path": str(target),
                "row_count": table.num_rows,
            }

        row_list = list(rows or [])
        if not row_list:
            pq.write_table(pa.table({}), target)
            return {"format": "parquet", "path": str(target), "row_count": 0}

        writer: pq.ParquetWriter | None = None
        written = 0
        try:
            for start in range(0, len(row_list), max(1, chunk_size)):
                batch_rows = row_list[start : start + chunk_size]
                batch_table = pa.Table.from_pylist(batch_rows)
                if writer is None:
                    writer = pq.ParquetWriter(target, batch_table.schema)
                writer.write_table(batch_table)
                written += len(batch_rows)
        finally:
            if writer is not None:
                writer.close()
        return {"format": "parquet", "path": str(target), "row_count": written}

    target = resolved if resolved.suffix == ".jsonl" else resolved.with_suffix(".jsonl")
    if columns is not None:
        keys = list(columns)
        n = len(next(iter(columns.values()), []))
        materialized = [{key: columns[key][i] for key in keys} for i in range(n)]
        _write_jsonl(target, materialized)
        return {"format": "jsonl", "path": str(target), "row_count": n}
    materialized_rows = list(rows or [])
    _write_jsonl(target, materialized_rows)
    return {"format": "jsonl", "path": str(target), "row_count": len(materialized_rows)}


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
