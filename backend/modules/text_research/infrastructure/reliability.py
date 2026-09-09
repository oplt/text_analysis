"""Inter-coder reliability statistics for text research.

Implements raw agreement, Cohen's kappa (two coders), Fleiss' kappa (three
or more coders, fixed-n design), and Krippendorff's alpha (nominal,
arbitrary number of coders, missing data supported) via the standard
coincidence-matrix formulation. No fabricated numbers: every value returned
is computed from the input labels.

Only the nominal measurement level is supported end-to-end (annotation
values in this module are unordered category labels such as "yes"/"no").
Ordinal weighted kappa is intentionally not implemented: there is no
ordinal label scale anywhere in the annotation pipeline, and applying a
weighted/ordinal statistic to nominal categories would silently misreport
disagreement severity.

References:
    - Cohen, J. (1960). A coefficient of agreement for nominal scales.
    - Fleiss, J. L. (1971). Measuring nominal scale agreement among many
      raters. Psychological Bulletin, 76(5), 378-382.
    - Krippendorff, K. (2004/2011). Computing Krippendorff's Alpha-Reliability.
      (coincidence-matrix formulation used here)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

Label = Hashable | None


def raw_agreement(labels_a: Sequence[Label], labels_b: Sequence[Label]) -> float:
    """Proportion of positions where ``labels_a`` and ``labels_b`` agree."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have equal length")
    if not labels_a:
        return 0.0
    matches = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    return matches / len(labels_a)


def cohens_kappa(labels_a: Sequence[Label], labels_b: Sequence[Label]) -> dict[str, Any]:
    """Cohen's kappa for two coders over the same set of units.

    Returns a dict with ``observed_agreement``, ``expected_agreement``,
    ``kappa``, and ``sample_size``.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have equal length")

    n = len(labels_a)
    if n == 0:
        return {
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "kappa": 0.0,
            "sample_size": 0,
        }

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    categories = set(count_a) | set(count_b)

    observed = raw_agreement(labels_a, labels_b)
    expected = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in categories)

    if math.isclose(expected, 1.0, abs_tol=1e-12):
        # Degenerate case: only one category exists across both coders.
        kappa = 1.0 if math.isclose(observed, 1.0, abs_tol=1e-12) else 0.0
    else:
        kappa = (observed - expected) / (1 - expected)

    return {
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "sample_size": n,
    }


def krippendorffs_alpha(
    reliability_data: list[list[Label]],
    level: str = "nominal",
) -> dict[str, Any]:
    """Krippendorff's alpha for an arbitrary number of coders, with missing data.

    Args:
        reliability_data: units x coders matrix. ``None`` marks a missing
            (not-annotated) value for a given unit/coder combination. Rows
            may have differing numbers of non-missing values.
        level: measurement level; only ``"nominal"`` is currently supported.

    Returns:
        Dict with ``alpha``, ``n_units``, ``n_coders``, ``missingness``.
    """
    if level != "nominal":
        raise ValueError(
            f"Unsupported measurement level: {level!r} (only 'nominal' is implemented)"
        )

    n_units = len(reliability_data)
    n_coders = max((len(row) for row in reliability_data), default=0)

    unit_values: list[list[Label]] = [[v for v in row if v is not None] for row in reliability_data]

    total_possible = n_units * n_coders
    present = sum(len(vals) for vals in unit_values)
    missingness = 1.0 - (present / total_possible) if total_possible else 0.0

    pairable_units = [vals for vals in unit_values if len(vals) >= 2]

    if not pairable_units:
        return {
            "alpha": None,
            "n_units": n_units,
            "n_coders": n_coders,
            "missingness": missingness,
        }

    categories = sorted({v for vals in unit_values for v in vals}, key=str)
    if len(categories) < 2:
        # Only one category ever used: agreement is total, but alpha is
        # conventionally undefined/1.0 depending on convention. We report
        # 1.0 since there is no possible disagreement in the data.
        return {
            "alpha": 1.0,
            "n_units": n_units,
            "n_coders": n_coders,
            "missingness": missingness,
        }

    index_of = {category: i for i, category in enumerate(categories)}
    k = len(categories)
    coincidence = [[0.0] * k for _ in range(k)]
    n = 0.0

    for vals in pairable_units:
        m = len(vals)
        weight = 1.0 / (m - 1)
        for i in range(m):
            ci = index_of[vals[i]]
            for j in range(m):
                if i == j:
                    continue
                cj = index_of[vals[j]]
                coincidence[ci][cj] += weight
        n += m

    category_totals = [sum(coincidence[c]) for c in range(k)]

    def delta_sq(c: int, other: int) -> float:
        # Nominal metric: 0 if the same category, 1 otherwise.
        return 0.0 if c == other else 1.0

    observed_disagreement = sum(
        coincidence[c][o] * delta_sq(c, o) for c in range(k) for o in range(k)
    )
    expected_disagreement = sum(
        category_totals[c] * category_totals[o] * delta_sq(c, o) for c in range(k) for o in range(k)
    )

    if expected_disagreement == 0:
        alpha = 1.0 if observed_disagreement == 0 else float("nan")
    else:
        alpha = 1.0 - ((n - 1) * observed_disagreement) / expected_disagreement

    return {
        "alpha": alpha,
        "n_units": n_units,
        "n_coders": n_coders,
        "missingness": missingness,
    }


def fleiss_kappa(reliability_data: list[list[Label]]) -> dict[str, Any]:
    """Fleiss' kappa for three or more coders rating nominal categories.

    Fleiss' original (1971) formula assumes every included unit was rated
    by the same fixed number of coders ``n``. To support the ragged
    assignment matrices produced by real annotation workflows (some units
    have missing ratings), this function:

    1. Drops ``None`` (missing) values from each unit's row.
    2. Determines ``n`` as the modal (most common) number of non-missing
       ratings among units with at least 2 ratings.
    3. Restricts the design to only the units that have exactly ``n``
       ratings, which is the standard fixed-raters-per-subject design
       Fleiss' kappa requires. Units with a different rater count are
       excluded and reported via ``n_units_excluded`` for transparency.

    Returns a dict with ``kappa`` (``None`` if not evaluable), ``n_raters``,
    ``n_units_included``, ``n_units_excluded``, ``n_categories``, and
    ``categories``.
    """
    non_missing = [[v for v in row if v is not None] for row in reliability_data]
    rater_counts = Counter(len(row) for row in non_missing if len(row) >= 2)

    if not rater_counts:
        return {
            "kappa": None,
            "n_raters": 0,
            "n_units_included": 0,
            "n_units_excluded": len(reliability_data),
            "n_categories": 0,
            "categories": [],
            "reason": "No unit has ratings from at least 2 coders.",
        }

    n_raters = rater_counts.most_common(1)[0][0]

    if n_raters < 3:
        return {
            "kappa": None,
            "n_raters": n_raters,
            "n_units_included": 0,
            "n_units_excluded": len(reliability_data),
            "n_categories": 0,
            "categories": [],
            "reason": (
                "Fleiss' kappa requires 3+ coders per unit in the fixed-n design; "
                f"the modal design here has only {n_raters} coder(s) per unit. "
                "Use Cohen's kappa for a two-coder design."
            ),
        }

    included_rows = [row for row in non_missing if len(row) == n_raters]
    n_units_included = len(included_rows)
    n_units_excluded = len(reliability_data) - n_units_included

    if n_units_included < 2:
        return {
            "kappa": None,
            "n_raters": n_raters,
            "n_units_included": n_units_included,
            "n_units_excluded": n_units_excluded,
            "n_categories": 0,
            "categories": [],
            "reason": f"Only {n_units_included} unit(s) have the fixed {n_raters}-coder design.",
        }

    categories = sorted({v for row in included_rows for v in row}, key=str)
    k = len(categories)
    index_of = {category: i for i, category in enumerate(categories)}

    table = [[0] * k for _ in range(n_units_included)]
    for i, row in enumerate(included_rows):
        for value in row:
            table[i][index_of[value]] += 1

    n = n_raters
    big_n = n_units_included

    category_totals = [sum(table[i][j] for i in range(big_n)) for j in range(k)]
    p_j = [total / (big_n * n) for total in category_totals]

    p_i = [
        (sum(count * count for count in table[i]) - n) / (n * (n - 1)) for i in range(big_n)
    ]
    p_bar = sum(p_i) / big_n
    p_e_bar = sum(p * p for p in p_j)

    if math.isclose(p_e_bar, 1.0, abs_tol=1e-12):
        kappa = 1.0 if math.isclose(p_bar, 1.0, abs_tol=1e-12) else 0.0
    else:
        kappa = (p_bar - p_e_bar) / (1 - p_e_bar)

    return {
        "kappa": kappa,
        "n_raters": n_raters,
        "n_units_included": n_units_included,
        "n_units_excluded": n_units_excluded,
        "n_categories": k,
        "categories": categories,
    }


def reliability_metadata(
    reliability_data: list[list[Label]],
    *,
    scale: str = "nominal",
    statistics_used: Sequence[str] = (),
) -> dict[str, Any]:
    """Descriptive metadata for a reliability computation.

    Args:
        reliability_data: units x coders matrix (``None`` = missing), same
            convention as :func:`krippendorffs_alpha`.
        scale: measurement scale of the underlying annotation values.
            Only ``"nominal"`` is supported end-to-end by this module;
            ordinal/weighted statistics are not implemented anywhere in the
            pipeline, so this is recorded for transparency rather than to
            select a different code path.
        statistics_used: names of the statistics actually reported for this
            computation (e.g. ``["raw_agreement", "cohens_kappa"]``).

    Returns:
        Dict with ``scale``, ``n_coders`` (max coders assigned to the
        label), ``n_units``, ``pairable_units`` (units with >= 2
        non-missing ratings), ``missing_values`` (count of missing
        unit/coder cells), and ``statistics_used``.
    """
    n_units = len(reliability_data)
    n_coders = max((len(row) for row in reliability_data), default=0)
    non_missing = [[v for v in row if v is not None] for row in reliability_data]
    pairable_units = sum(1 for row in non_missing if len(row) >= 2)
    total_possible = n_units * n_coders
    present = sum(len(row) for row in non_missing)
    missing_values = total_possible - present

    return {
        "scale": scale,
        "n_coders": n_coders,
        "n_units": n_units,
        "pairable_units": pairable_units,
        "missing_values": missing_values,
        "statistics_used": list(statistics_used),
    }


def agreement_matrix(coder_labels: dict[str, dict[str, Label]]) -> dict[str, Any]:
    """Pairwise raw-agreement matrix across coders.

    Args:
        coder_labels: ``{coder_id: {unit_id: label}}``.

    Returns:
        A dict with ``coders`` (sorted coder ids) and ``matrix``, where
        ``matrix[a][b]`` is ``{"agreement": float | None, "n_common_units": int}``.
    """
    coders = sorted(coder_labels.keys())
    matrix: dict[str, dict[str, dict[str, Any]]] = {}

    for coder_a in coders:
        matrix[coder_a] = {}
        units_a = coder_labels[coder_a]
        for coder_b in coders:
            units_b = coder_labels[coder_b]
            common_units = sorted(set(units_a) & set(units_b))

            if coder_a == coder_b:
                matrix[coder_a][coder_b] = {
                    "agreement": 1.0 if units_a else None,
                    "n_common_units": len(units_a),
                }
                continue

            if not common_units:
                matrix[coder_a][coder_b] = {"agreement": None, "n_common_units": 0}
                continue

            labels_a = [units_a[unit] for unit in common_units]
            labels_b = [units_b[unit] for unit in common_units]
            matrix[coder_a][coder_b] = {
                "agreement": raw_agreement(labels_a, labels_b),
                "n_common_units": len(common_units),
            }

    return {"coders": coders, "matrix": matrix}


def reliability_by_label(
    label_reliability_data: dict[str, list[list[Label]]],
    level: str = "nominal",
) -> dict[str, dict[str, Any]]:
    """Compute reliability statistics independently for each label.

    Args:
        label_reliability_data: ``{label_name: units_x_coders_matrix}`` where
            each matrix has the same shape convention as
            :func:`krippendorffs_alpha`.
        level: measurement level passed through to :func:`krippendorffs_alpha`.

    Returns:
        ``{label_name: {"alpha": {...}, "raw_agreement": float, "cohens_kappa": {...}}}``
        The two-coder keys (``raw_agreement``/``cohens_kappa``) are only
        included when every pairable unit for that label has exactly two
        coders (i.e. a clean two-coder design).
    """
    results: dict[str, dict[str, Any]] = {}

    for label, data in label_reliability_data.items():
        entry: dict[str, Any] = {"alpha": krippendorffs_alpha(data, level=level)}

        two_coder_pairs = [row for row in data if len(row) == 2]
        fully_paired = [row for row in two_coder_pairs if row[0] is not None and row[1] is not None]

        if two_coder_pairs and len(fully_paired) == len(two_coder_pairs) and fully_paired:
            labels_a = [row[0] for row in fully_paired]
            labels_b = [row[1] for row in fully_paired]
            entry["raw_agreement"] = raw_agreement(labels_a, labels_b)
            entry["cohens_kappa"] = cohens_kappa(labels_a, labels_b)

        results[label] = entry

    return results


def disagreement_units(
    values_by_coder: dict[str, dict[str, Label]],
) -> list[dict[str, Any]]:
    """Return units where at least two coders assigned different labels.

    Args:
        values_by_coder: ``{coder_id: {unit_id: label}}``.

    Returns:
        Sorted list of ``{"text_unit_id": str, "judgements": {coder_id: label}}``.
    """
    unit_judgements: dict[str, dict[str, Label]] = {}
    for coder_id, units in values_by_coder.items():
        for unit_id, value in units.items():
            unit_judgements.setdefault(unit_id, {})[coder_id] = value

    disagreements: list[dict[str, Any]] = []
    for unit_id in sorted(unit_judgements):
        judgements = unit_judgements[unit_id]
        if len(judgements) < 2:
            continue
        if len(set(judgements.values())) > 1:
            disagreements.append({"text_unit_id": unit_id, "judgements": judgements})
    return disagreements


def coder_pair_agreement_matrix(
    values_by_coder: dict[str, dict[str, Label]],
) -> dict[str, Any]:
    """Alias for :func:`agreement_matrix` used by reliability workflows."""
    return agreement_matrix(values_by_coder)


def krippendorff_alpha_nominal(reliability_data: list[list[Label]]) -> dict[str, Any]:
    """Alias for :func:`krippendorffs_alpha` at nominal measurement level."""
    return krippendorffs_alpha(reliability_data, level="nominal")
