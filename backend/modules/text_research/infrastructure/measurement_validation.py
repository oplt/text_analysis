"""Compare user-selected measurement sources without equating them (§51).

The caller decides which series are conceptually comparable. This module only
computes agreement / correlation / confusion / prevalence stats on aligned
observations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from typing import Any

import numpy as np


def _aligned_pairs(
    a: list[Any],
    b: list[Any],
    *,
    ids: list[str] | None = None,
) -> tuple[list[Any], list[Any], list[str]]:
    if len(a) != len(b):
        raise ValueError("Measurement series must have equal length")
    if ids is not None and len(ids) != len(a):
        raise ValueError("ids must align 1:1 with measurements")
    kept_a: list[Any] = []
    kept_b: list[Any] = []
    kept_ids: list[str] = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x is None or y is None or x == "" or y == "":
            continue
        kept_a.append(x)
        kept_b.append(y)
        kept_ids.append(ids[i] if ids else str(i))
    return kept_a, kept_b, kept_ids


def agreement_rate(a: list[Any], b: list[Any]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def confusion_matrix(a: list[Hashable], b: list[Hashable]) -> dict[str, Any]:
    labels = sorted({*a, *b}, key=lambda v: str(v))
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for x, y in zip(a, b, strict=True):
        matrix[index[x]][index[y]] += 1
    return {"labels": [str(lab) for lab in labels], "matrix": matrix}


def prevalence(series: list[Any]) -> dict[str, float]:
    counts = Counter(series)
    n = len(series) or 1
    return {str(k): v / n for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def pearson_correlation(a: list[Any], b: list[Any]) -> dict[str, Any]:
    xa = np.asarray([float(v) for v in a], dtype=float)
    xb = np.asarray([float(v) for v in b], dtype=float)
    if len(xa) < 2 or np.std(xa) == 0 or np.std(xb) == 0:
        return {"r": None, "n": int(len(xa)), "note": "insufficient variance or n"}
    r = float(np.corrcoef(xa, xb)[0, 1])
    return {"r": r, "n": int(len(xa))}


def compare_measurements(
    source_a: str,
    values_a: list[Any],
    source_b: str,
    values_b: list[Any],
    *,
    ids: list[str] | None = None,
    value_kind: str = "categorical",
    subgroup: list[str] | None = None,
) -> dict[str, Any]:
    """Generic triangulation between two measurement columns.

    ``value_kind``:
      - categorical: agreement + confusion + prevalence
      - continuous: Pearson correlation + means
    """
    a, b, kept_ids = _aligned_pairs(values_a, values_b, ids=ids)
    result: dict[str, Any] = {
        "source_a": source_a,
        "source_b": source_b,
        "value_kind": value_kind,
        "n_paired": len(a),
        "n_dropped_missing": len(values_a) - len(a),
        "unit_ids": kept_ids[:500],
        "equated_concepts": False,
        "notes": [
            "User asserted comparability; the platform does not equate annotation, "
            "dictionary, classifier, or topic scores automatically."
        ],
    }
    if value_kind == "continuous":
        result["correlation"] = pearson_correlation(a, b)
        result["mean_a"] = float(np.mean([float(v) for v in a])) if a else None
        result["mean_b"] = float(np.mean([float(v) for v in b])) if b else None
    else:
        result["agreement_rate"] = agreement_rate(a, b)
        result["confusion"] = confusion_matrix(a, b)
        result["prevalence_a"] = prevalence(a)
        result["prevalence_b"] = prevalence(b)

    if subgroup is not None:
        if len(subgroup) != len(values_a):
            raise ValueError("subgroup must align with original series length")
        by_group: dict[str, dict[str, Any]] = {}
        for g in sorted({str(s) for s in subgroup if s is not None and s != ""}):
            mask = [str(s) == g for s in subgroup]
            ga = [values_a[i] for i, m in enumerate(mask) if m]
            gb = [values_b[i] for i, m in enumerate(mask) if m]
            by_group[g] = compare_measurements(
                source_a,
                ga,
                source_b,
                gb,
                value_kind=value_kind,
            )
            # Avoid recursive explosion of nested subgroup keys
            by_group[g].pop("notes", None)
            by_group[g]["equated_concepts"] = False
        result["by_subgroup"] = by_group

    return result
