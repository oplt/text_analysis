"""Pure logic for stratified / random annotation sampling (prompt.txt §24).

This module knows nothing about the database, FastAPI, or any particular
corpus schema. Callers hand it a flat list of :class:`SamplingItem` records
(one per candidate text unit, carrying whatever metadata fields it wants to
stratify on) and get back a JSON-serializable ``SamplingPlan`` dict.

Stratification is fully generic: callers pick any combination of metadata
field names to group by (e.g. ``["field_1", "field_2"]``). There is no
hardcoded geographic or organizational dimension anywhere in this module.

Sampling is always seeded: if no seed is supplied one is generated and
returned in the plan so the exact same selection can be reproduced later by
passing that seed back in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from backend.modules.text_research.domain.enums import SamplingLevel, StratumMode

STRATUM_MODE_PROPORTIONAL = StratumMode.PROPORTIONAL.value
STRATUM_MODE_EQUAL = StratumMode.EQUAL.value
STRATUM_MODES: frozenset[str] = frozenset(m.value for m in StratumMode)

SAMPLING_LEVEL_UNIT = SamplingLevel.UNIT.value
SAMPLING_LEVEL_DOCUMENT = SamplingLevel.DOCUMENT.value
SAMPLING_LEVELS: frozenset[str] = frozenset(m.value for m in SamplingLevel)

# Sentinel for grouping units whose stratify field is missing/empty so they
# still form a well-defined (rather than silently dropped) stratum.
_MISSING_STRATUM_VALUE = "\x00__missing__"

_MAX_SEED = 2**31 - 1


@dataclass(frozen=True, slots=True)
class SamplingItem:
    """One sampling candidate — normally a text unit.

    ``metadata`` holds whatever fields the caller may want to stratify on
    (e.g. document-level attributes such as organization/country, or custom
    keys pulled from a document's free-form metadata JSON). ``document_id``
    is used for document-level sampling and the max-units-per-document cap.
    """

    id: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_stratify_fields(stratify_by: list[str] | None) -> list[str]:
    if not stratify_by:
        return []
    return list(dict.fromkeys(str(f) for f in stratify_by if str(f).strip()))


def _stratum_key(item: SamplingItem, stratify_fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in stratify_fields:
        value = item.metadata.get(field_name)
        values.append(_MISSING_STRATUM_VALUE if value in (None, "") else str(value))
    return tuple(values)


def _allocate(sizes: list[int], total: int, mode: str) -> list[int]:
    """Split ``total`` across strata of the given ``sizes`` using ``mode``.

    Guarantees: each allocation is capped by its stratum size, the sum never
    exceeds ``total`` or the sum of ``sizes``, and any shortfall caused by a
    stratum running out of members is greedily redistributed to strata that
    still have spare capacity (in stable, deterministic order).
    """
    n = len(sizes)
    if n == 0:
        return []
    total_available = sum(sizes)
    target_total = min(total, total_available)

    if mode == STRATUM_MODE_EQUAL:
        base = target_total // n
        remainder = target_total - base * n
        raw = [base] * n
        for i in range(remainder):
            raw[i] += 1
    else:  # proportional
        if total_available == 0:
            raw = [0] * n
        else:
            exact = [target_total * size / total_available for size in sizes]
            raw = [int(x) for x in exact]  # floor
            remainder = target_total - sum(raw)
            # Largest-remainder method for deterministic, exact-sum rounding.
            order = sorted(range(n), key=lambda i: (exact[i] - raw[i], -i), reverse=True)
            for i in range(remainder):
                raw[order[i % n]] += 1

    allocation = [min(raw[i], sizes[i]) for i in range(n)]
    shortfall = target_total - sum(allocation)
    if shortfall > 0:
        capacity = [sizes[i] - allocation[i] for i in range(n)]
        idx = 0
        # Round-robin fill of any spare capacity, deterministic by index.
        guard = 0
        max_iterations = shortfall * n + n + 1
        while shortfall > 0 and any(c > 0 for c in capacity) and guard < max_iterations:
            i = idx % n
            if capacity[i] > 0:
                allocation[i] += 1
                capacity[i] -= 1
                shortfall -= 1
            idx += 1
            guard += 1
    return allocation


def build_sampling_plan(
    items: list[SamplingItem],
    *,
    sample_size: int,
    random_seed: int | None = None,
    stratify_by: list[str] | None = None,
    stratum_mode: str = STRATUM_MODE_PROPORTIONAL,
    sampling_level: str = SAMPLING_LEVEL_UNIT,
    max_units_per_document: int | None = None,
) -> dict[str, Any]:
    """Build a stratified/random sampling plan over ``items``.

    Returns a JSON-serializable dict (a ``SamplingPlan``) with the config
    used, the resolved random seed, the selected unit/document ids, and
    per-stratum counts — suitable for persisting alongside an annotation
    assignment or analysis run.

    Random sampling, seeded sampling, proportional stratification, and
    equal-per-stratum sampling are all the same algorithm: when
    ``stratify_by`` is empty, every item falls into one implicit stratum, so
    a plain seeded random sample of size ``sample_size`` falls out for free.
    """
    if not items:
        raise ValueError("No candidate units available for sampling")
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    mode = (stratum_mode or STRATUM_MODE_PROPORTIONAL).strip().lower()
    if mode not in STRATUM_MODES:
        raise ValueError(
            f"Unknown stratum_mode {stratum_mode!r}; expected one of {sorted(STRATUM_MODES)}"
        )

    level = (sampling_level or SAMPLING_LEVEL_UNIT).strip().lower()
    if level not in SAMPLING_LEVELS:
        raise ValueError(
            f"Unknown sampling_level {sampling_level!r}; expected one of {sorted(SAMPLING_LEVELS)}"
        )

    if max_units_per_document is not None and max_units_per_document < 1:
        raise ValueError("max_units_per_document must be >= 1")

    stratify_fields = _normalize_stratify_fields(stratify_by)

    if random_seed is None:
        random_seed = random.SystemRandom().randrange(1, _MAX_SEED)
    rng = random.Random(random_seed)

    by_document: dict[str, list[SamplingItem]] = {}
    for item in items:
        by_document.setdefault(item.document_id, []).append(item)

    if level == SAMPLING_LEVEL_DOCUMENT:
        # One atom per document; stratify on that document's own metadata.
        atoms = [
            SamplingItem(id=doc_id, document_id=doc_id, metadata=doc_items[0].metadata)
            for doc_id, doc_items in by_document.items()
        ]
    else:
        atoms = list(items)

    strata_map: dict[tuple[str, ...], list[SamplingItem]] = {}
    for atom in atoms:
        key = _stratum_key(atom, stratify_fields)
        strata_map.setdefault(key, []).append(atom)

    stratum_keys = sorted(strata_map.keys())
    sizes = [len(strata_map[key]) for key in stratum_keys]
    allocation = _allocate(sizes, sample_size, mode)

    selected_atoms: list[SamplingItem] = []
    doc_selected_counts: dict[str, int] = {}
    strata_info: list[dict[str, Any]] = []

    for key, target, size in zip(stratum_keys, allocation, sizes, strict=True):
        pool = list(strata_map[key])
        rng.shuffle(pool)

        chosen: list[SamplingItem] = []
        if level == SAMPLING_LEVEL_UNIT and max_units_per_document is not None:
            for atom in pool:
                if len(chosen) >= target:
                    break
                if doc_selected_counts.get(atom.document_id, 0) >= max_units_per_document:
                    continue
                chosen.append(atom)
                doc_selected_counts[atom.document_id] = (
                    doc_selected_counts.get(atom.document_id, 0) + 1
                )
        else:
            chosen = pool[:target]
            if level == SAMPLING_LEVEL_UNIT:
                for atom in chosen:
                    doc_selected_counts[atom.document_id] = (
                        doc_selected_counts.get(atom.document_id, 0) + 1
                    )

        selected_atoms.extend(chosen)
        stratum_labels = dict(zip(stratify_fields, key, strict=True)) if stratify_fields else {}
        # Restore the true missing marker to None for readability in output.
        stratum_labels = {
            field_name: (None if value == _MISSING_STRATUM_VALUE else value)
            for field_name, value in stratum_labels.items()
        }
        strata_info.append(
            {
                "stratum": stratum_labels,
                "available": size,
                "selected": len(chosen),
            }
        )

    if level == SAMPLING_LEVEL_DOCUMENT:
        selected_document_ids = [atom.id for atom in selected_atoms]
        selected_unit_ids: list[str] = []
        for doc_id in selected_document_ids:
            doc_units = list(by_document[doc_id])
            if max_units_per_document is not None and len(doc_units) > max_units_per_document:
                doc_units = rng.sample(doc_units, max_units_per_document)
            selected_unit_ids.extend(unit.id for unit in doc_units)
    else:
        selected_unit_ids = [atom.id for atom in selected_atoms]
        selected_document_ids = list(dict.fromkeys(atom.document_id for atom in selected_atoms))

    return {
        "sampling_config": {
            "sample_size": sample_size,
            "stratify_by": stratify_fields,
            "stratum_mode": mode,
            "sampling_level": level,
            "max_units_per_document": max_units_per_document,
        },
        "random_seed": random_seed,
        "selected_unit_ids": selected_unit_ids,
        "selected_document_ids": selected_document_ids,
        "strata": strata_info,
        "total_candidates": len(atoms),
        "total_selected": len(selected_atoms),
    }
