"""Canonical full-matrix identity for DFM scientific comparison.

Checksum is a SHA-256 over sorted semantic cells ``(unit_id, feature_name, value)``.
Count matrices use exact float formatting; weighted matrices may pass a tolerance
at comparison time without treating raw byte equality as scientific identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def format_matrix_value(value: float, *, float_tol: float = 0.0) -> str:
    """Canonical string form for a matrix value in the checksum stream."""
    if float_tol <= 0:
        # Exact scientific identity for counts / exact floats.
        if float(value).is_integer():
            return str(int(value))
        return repr(float(value))
    # Quantize to tolerance scale for future weighted matrices.
    scale = max(float_tol, 1e-15)
    quantized = round(float(value) / scale) * scale
    return f"{quantized:.12g}"


def iter_semantic_cells(
    *,
    unit_ids: Sequence[str],
    feature_names: Sequence[str],
    triples: Iterable[tuple[int, int, float]],
) -> list[tuple[str, str, float]]:
    """Map COO index triples onto sorted semantic ``(unit, feature, value)`` cells."""
    cells: list[tuple[str, str, float]] = []
    n_units = len(unit_ids)
    n_features = len(feature_names)
    for row, col, value in triples:
        if value == 0:
            continue
        unit = unit_ids[row] if 0 <= row < n_units else str(row)
        feature = feature_names[col] if 0 <= col < n_features else str(col)
        cells.append((str(unit), str(feature), float(value)))
    cells.sort(key=lambda item: (item[0], item[1]))
    return cells


def matrix_checksum_from_cells(
    cells: Sequence[tuple[str, str, float]],
    *,
    float_tol: float = 0.0,
) -> str:
    """SHA-256 over canonical newline-delimited ``unit\\tfeature\\tvalue`` rows."""
    digest = hashlib.sha256()
    for unit, feature, value in cells:
        line = f"{unit}\t{feature}\t{format_matrix_value(value, float_tol=float_tol)}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def matrix_checksum_from_coo(
    *,
    unit_ids: Sequence[str],
    feature_names: Sequence[str],
    triples: Iterable[tuple[int, int, float]],
    float_tol: float = 0.0,
) -> tuple[str, int]:
    """Return ``(checksum, cell_count)`` for a full COO triple stream."""
    cells = iter_semantic_cells(unit_ids=unit_ids, feature_names=feature_names, triples=triples)
    return matrix_checksum_from_cells(cells, float_tol=float_tol), len(cells)


def matrix_checksum_from_csr(
    matrix: Any,
    *,
    unit_ids: Sequence[str],
    feature_names: Sequence[str],
    float_tol: float = 0.0,
) -> tuple[str, int]:
    """Checksum a scipy CSR/COO matrix without relying on truncated JSON payloads."""
    coo = matrix.tocoo()
    triples = zip(
        (int(r) for r in coo.row.tolist()),
        (int(c) for c in coo.col.tolist()),
        (float(v) for v in coo.data.tolist()),
        strict=False,
    )
    return matrix_checksum_from_coo(
        unit_ids=unit_ids,
        feature_names=feature_names,
        triples=triples,
        float_tol=float_tol,
    )


def inline_sparse_is_complete(
    payload: Mapping[str, Any] | None, *, declared_nnz: int | None
) -> bool:
    """True only when inline COO is known to cover the full matrix."""
    if not isinstance(payload, dict):
        return False
    if payload.get("preview_only") is True or payload.get("truncated") is True:
        return False
    triples_len = _inline_triple_count(payload)
    if declared_nnz is not None and int(declared_nnz) != triples_len:
        return False
    exported = payload.get("nnz_exported")
    if exported is not None and int(exported) != triples_len:
        return False
    nnz_field = payload.get("nnz")
    if nnz_field is not None and int(nnz_field) != triples_len:
        # Payload nnz may mean full matrix nnz while rows are truncated.
        if payload.get("truncated") or payload.get("preview_only"):
            return False
        if int(nnz_field) > triples_len:
            return False
    return triples_len > 0 or declared_nnz == 0


def _inline_triple_count(payload: Mapping[str, Any]) -> int:
    if payload.get("rows") is not None:
        return len(payload.get("rows") or [])
    return len(payload.get("row") or [])


def extract_inline_coo_triples(payload: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    if payload.get("rows") is not None:
        rows = payload.get("rows") or []
        cols = payload.get("cols") or []
        values = payload.get("values") or []
    else:
        rows = payload.get("row") or []
        cols = payload.get("col") or []
        values = payload.get("data") or []
    return [
        (int(row), int(col), float(value))
        for row, col, value in zip(rows, cols, values, strict=False)
    ]


def dfm_side_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one engine's DFM payload into comparison-ready identity fields."""
    unit_ids = [str(uid) for uid in (payload.get("unit_ids") or [])]
    feature_names = [str(name) for name in (payload.get("feature_names") or [])]
    sparse = payload.get("sparse_coo") if isinstance(payload.get("sparse_coo"), dict) else None
    if sparse is None and isinstance(payload.get("sparse"), dict):
        sparse = payload.get("sparse")
    declared_nnz = payload.get("nnz")
    if declared_nnz is None and isinstance(payload.get("summary"), dict):
        declared_nnz = payload["summary"].get("nnz")
    if declared_nnz is None and isinstance(sparse, dict):
        declared_nnz = sparse.get("nnz")
    checksum = payload.get("matrix_checksum")
    checksum_complete = bool(payload.get("matrix_checksum_complete"))
    if checksum and payload.get("matrix_checksum_complete") is None:
        # Explicit checksum without a false completeness flag counts as complete.
        checksum_complete = True
    inline_complete = inline_sparse_is_complete(
        sparse if isinstance(sparse, dict) else None,
        declared_nnz=int(declared_nnz) if declared_nnz is not None else None,
    )
    return {
        "unit_ids": unit_ids,
        "feature_names": feature_names,
        "sparse": sparse if isinstance(sparse, dict) else {},
        "declared_nnz": int(declared_nnz) if declared_nnz is not None else None,
        "matrix_checksum": checksum if isinstance(checksum, str) else None,
        "matrix_checksum_complete": checksum_complete,
        "inline_complete": inline_complete,
        "dimensions": payload.get("dimensions"),
    }


def compare_dfm_identities(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    float_tol: float = 0.0,
    max_differences: int = 50,
) -> dict[str, Any]:
    """Full-matrix-aware DFM comparison; never claims equality from previews alone."""
    left_id = dfm_side_identity(left)
    right_id = dfm_side_identity(right)
    left_units = set(left_id["unit_ids"])
    right_units = set(right_id["unit_ids"])
    left_features = set(left_id["feature_names"])
    right_features = set(right_id["feature_names"])
    vocabulary_equal = left_features == right_features
    units_equal = left_units == right_units

    left_cs = left_id["matrix_checksum"]
    right_cs = right_id["matrix_checksum"]
    both_checksums = bool(
        left_cs
        and right_cs
        and left_id["matrix_checksum_complete"]
        and right_id["matrix_checksum_complete"]
    )
    cell_differences: dict[str, Any] = {}
    cells_compared = 0
    cells_equal: bool | None = None
    matrix_checksum_equal: bool | None = None
    comparison_complete = False
    comparison_status = "inconclusive"

    if both_checksums:
        comparison_complete = True
        matrix_checksum_equal = left_cs == right_cs
        cells_equal = matrix_checksum_equal
        cells_compared = max(
            int(left_id["declared_nnz"] or 0),
            int(right_id["declared_nnz"] or 0),
        )
        comparison_status = "equal" if cells_equal else "unequal"
        # Optional diagnostic cell diff when checksums diverge and inline is complete.
        if not cells_equal and left_id["inline_complete"] and right_id["inline_complete"]:
            cell_differences, cells_compared = _cell_diff(
                left_id, right_id, float_tol=float_tol, max_differences=max_differences
            )
    elif left_id["inline_complete"] and right_id["inline_complete"]:
        comparison_complete = True
        cell_differences, cells_compared = _cell_diff(
            left_id, right_id, float_tol=float_tol, max_differences=max_differences
        )
        cells_equal = not cell_differences
        left_computed, _ = matrix_checksum_from_coo(
            unit_ids=left_id["unit_ids"],
            feature_names=left_id["feature_names"],
            triples=extract_inline_coo_triples(left_id["sparse"]),
            float_tol=float_tol,
        )
        right_computed, _ = matrix_checksum_from_coo(
            unit_ids=right_id["unit_ids"],
            feature_names=right_id["feature_names"],
            triples=extract_inline_coo_triples(right_id["sparse"]),
            float_tol=float_tol,
        )
        matrix_checksum_equal = left_computed == right_computed
        left_cs = left_cs or left_computed
        right_cs = right_cs or right_computed
        comparison_status = "equal" if cells_equal else "unequal"
    else:
        comparison_complete = False
        cells_equal = None
        matrix_checksum_equal = None
        comparison_status = "inconclusive"

    return {
        "analysis_type": "dfm",
        "comparison_complete": comparison_complete,
        "comparison_status": comparison_status,
        "matrix_checksum_equal": matrix_checksum_equal,
        "python_matrix_checksum": left_cs,
        "r_matrix_checksum": right_cs,
        "cells_equal": cells_equal,
        "cells_compared": cells_compared,
        "python_dimensions": left_id["dimensions"],
        "r_dimensions": right_id["dimensions"],
        "vocabulary_equal": vocabulary_equal,
        "vocabulary_overlap": len(left_features & right_features),
        "units_equal": units_equal,
        "unit_overlap": len(left_units & right_units),
        "python_nnz": left_id["declared_nnz"],
        "r_nnz": right_id["declared_nnz"],
        "cell_difference_count": len(cell_differences),
        "cell_differences": cell_differences,
        "python_inline_complete": left_id["inline_complete"],
        "r_inline_complete": right_id["inline_complete"],
    }


def _cell_diff(
    left_id: dict[str, Any],
    right_id: dict[str, Any],
    *,
    float_tol: float,
    max_differences: int,
) -> tuple[dict[str, Any], int]:
    left_cells = {
        (unit, feature): value
        for unit, feature, value in iter_semantic_cells(
            unit_ids=left_id["unit_ids"],
            feature_names=left_id["feature_names"],
            triples=extract_inline_coo_triples(left_id["sparse"]),
        )
    }
    right_cells = {
        (unit, feature): value
        for unit, feature, value in iter_semantic_cells(
            unit_ids=right_id["unit_ids"],
            feature_names=right_id["feature_names"],
            triples=extract_inline_coo_triples(right_id["sparse"]),
        )
    }
    differences: dict[str, Any] = {}
    for unit, feature in sorted(set(left_cells) | set(right_cells)):
        left_val = left_cells.get((unit, feature))
        right_val = right_cells.get((unit, feature))
        if left_val is None or right_val is None:
            differences[f"{unit}|{feature}"] = {"python": left_val, "r": right_val}
        elif float_tol <= 0:
            if left_val != right_val:
                differences[f"{unit}|{feature}"] = {"python": left_val, "r": right_val}
        elif abs(left_val - right_val) > float_tol:
            differences[f"{unit}|{feature}"] = {"python": left_val, "r": right_val}
        if len(differences) >= max_differences:
            break
    compared = max(len(left_cells), len(right_cells))
    return dict(list(differences.items())[:max_differences]), compared


def attach_matrix_checksum_fields(
    payload: dict[str, Any],
    *,
    checksum: str,
    cells_compared: int,
) -> dict[str, Any]:
    """Return payload copy with canonical matrix identity fields."""
    updated = dict(payload)
    updated["matrix_checksum"] = checksum
    updated["matrix_checksum_complete"] = True
    updated["matrix_checksum_cells"] = cells_compared
    updated["matrix_checksum_algorithm"] = "sha256:unit\\tfeature\\tvalue"
    return updated


def dump_identity_debug(cells: Sequence[tuple[str, str, float]]) -> str:
    """Stable JSON for tests/debug."""
    return json.dumps(
        [{"unit": u, "feature": f, "value": v} for u, f, v in cells],
        sort_keys=True,
    )
