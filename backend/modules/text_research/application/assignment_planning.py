"""Pure helpers for annotation task assignment planning."""

from __future__ import annotations


def plan_annotation_assignment(
    unit_ids: list[str],
    annotator_ids: list[str],
    *,
    sample_size: int,
    strategy: str = "overlap",
    overlap_count: int | None = None,
    overlap_percent: float | None = None,
) -> dict[str, list[str]]:
    """Return annotator_id -> ordered unit ids to assign.

    Strategies:
    - ``shared``: every annotator receives the same sample
    - ``disjoint``: split the sample with no overlap
    - ``overlap``: ``overlap_count`` (or percent) units go to all annotators;
      remaining unique units are split evenly
    """
    if not annotator_ids:
        raise ValueError("At least one annotator is required")
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    pool = unit_ids[:sample_size]
    if not pool:
        return {annotator_id: [] for annotator_id in annotator_ids}

    strategy_key = strategy.strip().lower()
    if strategy_key == "shared":
        return {annotator_id: list(pool) for annotator_id in annotator_ids}

    if strategy_key == "disjoint":
        buckets: dict[str, list[str]] = {annotator_id: [] for annotator_id in annotator_ids}
        for index, unit_id in enumerate(pool):
            buckets[annotator_ids[index % len(annotator_ids)]].append(unit_id)
        return buckets

    if strategy_key != "overlap":
        raise ValueError(f"Unknown assignment strategy: {strategy}")

    if overlap_count is None:
        percent = 0.0 if overlap_percent is None else float(overlap_percent)
        overlap_count = int(round(len(pool) * (percent / 100.0)))
    overlap_count = max(0, min(overlap_count, len(pool)))

    shared = pool[:overlap_count]
    remainder = pool[overlap_count:]
    buckets = {annotator_id: list(shared) for annotator_id in annotator_ids}
    for index, unit_id in enumerate(remainder):
        buckets[annotator_ids[index % len(annotator_ids)]].append(unit_id)
    return buckets


def assignment_pairs(plan: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten plan into (text_unit_id, annotator_id) pairs."""
    pairs: list[tuple[str, str]] = []
    for annotator_id, unit_ids in plan.items():
        for unit_id in unit_ids:
            pairs.append((unit_id, annotator_id))
    return pairs
