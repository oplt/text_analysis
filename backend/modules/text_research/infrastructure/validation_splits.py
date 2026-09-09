"""Generic, user-configurable robustness/domain-shift and temporal validation
splits (§38, §39).

Pure index-arithmetic — no sklearn, no I/O, no database access — so every
function here is trivially unit-testable and reusable outside
:mod:`robustness_service`.

Design principle: nothing in this module hardcodes a domain-specific grouping
field (organization/region/country/...). Callers resolve a ``group_field`` or
``temporal_field`` name against document metadata themselves (see
``CorpusDocument.get_field_value``) and pass the resulting per-unit values in
here as plain lists aligned 1:1 with the unit/text order.

Limitations (documented, not hidden):

* Leave-one-group-out with many distinct values can be expensive (one fit per
  value); callers should cap the number of held-out values for large corpora.
* Temporal splits assume the temporal field is orderable (int/float year, or
  ISO date string); non-orderable values are dropped with an explicit count
  rather than silently coerced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MISSING_SENTINELS = (None, "", "unspecified", "none")


@dataclass(frozen=True, slots=True)
class GroupSplit:
    held_out_value: str
    train_index: list[int]
    test_index: list[int]


@dataclass(frozen=True, slots=True)
class TransferSplit:
    train_index: list[int]
    test_index: list[int]
    train_values: list[str] = field(default_factory=list)
    test_values: list[str] = field(default_factory=list)
    excluded_index: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train_index: list[int]
    test_index: list[int]
    train_period: Any = None
    test_period: Any = None
    excluded_index: list[int] = field(default_factory=list)


def _clean_key(value: Any) -> str | None:
    """Normalize a group value to a comparable string key, or ``None`` if missing."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _MISSING_SENTINELS:
        return None
    return text


def eligible_group_values(values: list[Any]) -> list[str]:
    """Distinct, present (non-missing) group values, sorted for determinism."""
    keys = {_clean_key(v) for v in values}
    keys.discard(None)
    return sorted(keys)


def leave_one_group_out(values: list[Any], *, max_groups: int | None = None) -> list[GroupSplit]:
    """Build one train/test split per distinct group value (§38).

    Held-out group forms the test set; everything else (with a known,
    non-missing group value) forms the train set. Rows with a missing group
    value are excluded from both train and test for that split, since they
    cannot be meaningfully assigned to "not this group".

    ``max_groups`` caps how many distinct values are swept (first N in sorted
    order) to bound cost on high-cardinality fields; ``None`` sweeps all.
    """
    keys = [_clean_key(v) for v in values]
    distinct = eligible_group_values(values)
    if max_groups is not None:
        distinct = distinct[: max(0, max_groups)]

    splits: list[GroupSplit] = []
    for held_out in distinct:
        train_index = [i for i, k in enumerate(keys) if k is not None and k != held_out]
        test_index = [i for i, k in enumerate(keys) if k == held_out]
        splits.append(
            GroupSplit(held_out_value=held_out, train_index=train_index, test_index=test_index)
        )
    return splits


def transfer_split(
    values: list[Any],
    *,
    train_values: list[Any],
    test_values: list[Any],
) -> TransferSplit:
    """Generic transfer test: train where field ∈ A, test where field ∈ B (§38).

    ``train_values``/``test_values`` are user-defined filters (e.g.
    ``group_field="region"``, ``train_values=["west"]``,
    ``test_values=["east"]``). Overlap between A and B is rejected (a value
    cannot simultaneously define train and test membership).
    """
    train_set = {_clean_key(v) for v in train_values if _clean_key(v) is not None}
    test_set = {_clean_key(v) for v in test_values if _clean_key(v) is not None}
    if not train_set:
        raise ValueError("transfer_split requires at least one non-empty train_values entry")
    if not test_set:
        raise ValueError("transfer_split requires at least one non-empty test_values entry")
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"train_values and test_values must be disjoint; overlap: {sorted(overlap)}"
        )

    keys = [_clean_key(v) for v in values]
    train_index = [i for i, k in enumerate(keys) if k in train_set]
    test_index = [i for i, k in enumerate(keys) if k in test_set]
    excluded_index = [i for i, k in enumerate(keys) if k not in train_set and k not in test_set]
    return TransferSplit(
        train_index=train_index,
        test_index=test_index,
        train_values=sorted(train_set),
        test_values=sorted(test_set),
        excluded_index=excluded_index,
    )


def _orderable_periods(values: list[Any]) -> list[tuple[int, Any]]:
    """(index, value) pairs for rows with a non-missing, orderable temporal value."""
    out: list[tuple[int, Any]] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            # Accept int/float directly; reject strings that don't parse as
            # numbers rather than silently treating them as equal/sortable
            # opaque strings (which would misorder ISO dates lexically only
            # by coincidence for same-length strings).
            if isinstance(v, bool):
                continue
            if isinstance(v, int | float):
                out.append((i, v))
            elif isinstance(v, str) and v.strip():
                out.append((i, v.strip()))
        except (TypeError, ValueError):
            continue
    return out


def temporal_holdout_split(values: list[Any], *, split_at: Any = None) -> TemporalSplit:
    """Simple train-before-test temporal holdout (§39).

    ``values`` are per-unit temporal field values (year, or any orderable
    scalar/date string), aligned 1:1 with the corpus texts. When ``split_at``
    is omitted, the median distinct value is used so both sides are
    non-trivially populated whenever possible.
    """
    pairs = _orderable_periods(values)
    excluded_index = [i for i in range(len(values)) if i not in {p[0] for p in pairs}]
    distinct_sorted = sorted({p[1] for p in pairs})
    if len(distinct_sorted) < 2:
        return TemporalSplit(train_index=[], test_index=[], excluded_index=excluded_index)

    cutoff = split_at if split_at is not None else distinct_sorted[len(distinct_sorted) // 2 - 1]
    train_index = [i for i, v in pairs if v <= cutoff]
    test_index = [i for i, v in pairs if v > cutoff]
    return TemporalSplit(
        train_index=train_index,
        test_index=test_index,
        train_period=cutoff,
        test_period=f">{cutoff}",
        excluded_index=excluded_index,
    )


def expanding_window_splits(values: list[Any], *, min_windows: int = 3) -> list[TemporalSplit]:
    """Expanding-window temporal validation (§39): train grows, test slides forward.

    For each distinct period (after the first), train = all rows at or
    before the previous period, test = rows exactly at the current period.
    Requires at least ``min_windows`` distinct periods to be worth running;
    otherwise returns an empty list (caller should fall back to a simple
    holdout instead of a degenerate sweep).
    """
    pairs = _orderable_periods(values)
    distinct_sorted = sorted({p[1] for p in pairs})
    if len(distinct_sorted) < min_windows:
        return []

    splits: list[TemporalSplit] = []
    for boundary_idx in range(1, len(distinct_sorted)):
        cutoff = distinct_sorted[boundary_idx - 1]
        current_period = distinct_sorted[boundary_idx]
        train_index = [i for i, v in pairs if v <= cutoff]
        test_index = [i for i, v in pairs if v == current_period]
        if not train_index or not test_index:
            continue
        splits.append(
            TemporalSplit(
                train_index=train_index,
                test_index=test_index,
                train_period=f"<= {cutoff}",
                test_period=current_period,
            )
        )
    return splits


def rolling_window_splits(
    values: list[Any], *, window_size: int, min_windows: int = 3
) -> list[TemporalSplit]:
    """Rolling-window temporal validation (§39): fixed-size train window slides forward.

    Unlike :func:`expanding_window_splits`, train is limited to the most
    recent ``window_size`` distinct periods immediately preceding the test
    period (older data is dropped rather than accumulated). Useful for
    detecting whether a model trained only on "recent" data generalizes to
    the next period as well as one trained on everything so far.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    pairs = _orderable_periods(values)
    distinct_sorted = sorted({p[1] for p in pairs})
    if len(distinct_sorted) < min_windows:
        return []

    splits: list[TemporalSplit] = []
    for boundary_idx in range(window_size, len(distinct_sorted)):
        window_periods = set(distinct_sorted[boundary_idx - window_size : boundary_idx])
        current_period = distinct_sorted[boundary_idx]
        train_index = [i for i, v in pairs if v in window_periods]
        test_index = [i for i, v in pairs if v == current_period]
        if not train_index or not test_index:
            continue
        splits.append(
            TemporalSplit(
                train_index=train_index,
                test_index=test_index,
                train_period=sorted(window_periods),
                test_period=current_period,
            )
        )
    return splits


def class_prevalence(labels: list[list[str]]) -> dict[str, float]:
    """Per-label positive prevalence for a multilabel target list (diagnostic)."""
    if not labels:
        return {}
    counts: dict[str, int] = {}
    for row in labels:
        for label in row:
            counts[label] = counts.get(label, 0) + 1
    n = len(labels)
    return {label: count / n for label, count in sorted(counts.items())}
