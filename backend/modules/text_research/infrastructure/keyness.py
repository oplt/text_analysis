"""Keyness / group comparison statistics for research text analysis.

Comparison groups are always supplied by the caller from corpus metadata —
this module never hardcodes research categories or domain labels.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from scipy import stats

KEYNESS_METHODS: frozenset[str] = frozenset(
    {
        "log_likelihood",
        "g2",
        "chi_square",
        "chi2",
        "fisher",
    }
)

CORRECTIONS: frozenset[str] = frozenset({"none", "bh", "fdr", "benjamini_hochberg"})


def normalize_keyness_method(method: str | None) -> str:
    key = (method or "log_likelihood").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "g2": "log_likelihood",
        "ll": "log_likelihood",
        "loglikelihood": "log_likelihood",
        "likelihood": "log_likelihood",
        "chi2": "chi_square",
        "chisquare": "chi_square",
        "chi_squared": "chi_square",
        "fisher_exact": "fisher",
    }
    canonical = aliases.get(key, key)
    if canonical not in {"log_likelihood", "chi_square", "fisher"}:
        raise ValueError(
            f"Unsupported keyness method {method!r}; expected one of "
            f"log_likelihood, chi_square, fisher"
        )
    return canonical


def normalize_correction(correction: str | None) -> str | None:
    if correction is None:
        return "bh"
    key = str(correction).strip().lower().replace("-", "_")
    aliases = {"fdr": "bh", "benjamini_hochberg": "bh", "bh_fdr": "bh"}
    key = aliases.get(key, key)
    if key in {"", "none", "off", "false"}:
        return None
    if key != "bh":
        raise ValueError(
            f"Unsupported multiple-testing correction {correction!r}; "
            f"expected 'bh' (Benjamini–Hochberg) or 'none'"
        )
    return "bh"


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini–Hochberg FDR adjustment (independent / positive regression)."""
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and math.isfinite(p)]
    n = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    if n == 0:
        return adjusted

    indexed.sort(key=lambda item: item[1])
    # Working array in rank order (ascending p).
    bh = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        idx_in_sorted = rank - 1
        raw = indexed[idx_in_sorted][1]
        val = min(prev, raw * n / rank)
        bh[idx_in_sorted] = val
        prev = val
    for (orig_i, _), adj in zip(indexed, bh, strict=True):
        adjusted[orig_i] = min(1.0, adj)
    return adjusted


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


def _g2_statistic(a: int, b: int, total_a: int, total_b: int) -> float:
    """Dunning's log-likelihood ratio (G²) for a 2×2 frequency table."""
    grand = total_a + total_b
    if grand <= 0 or (a == 0 and b == 0):
        return 0.0
    # Also include the complement cells for a proper 2×2 G².
    a_not = total_a - a
    b_not = total_b - b
    row_term = a + b
    row_other = a_not + b_not
    if row_term == 0 or row_other == 0 or total_a == 0 or total_b == 0:
        # Fall back to 1×2 style used historically when a term is absent in one corpus.
        expected_a = total_a * row_term / grand if grand else 0.0
        expected_b = total_b * row_term / grand if grand else 0.0
        g2 = 0.0
        if a > 0 and expected_a > 0:
            g2 += a * math.log(a / expected_a)
        if b > 0 and expected_b > 0:
            g2 += b * math.log(b / expected_b)
        return 2.0 * g2

    cells = (
        (a, total_a * row_term / grand),
        (b, total_b * row_term / grand),
        (a_not, total_a * row_other / grand),
        (b_not, total_b * row_other / grand),
    )
    g2 = 0.0
    for observed, expected in cells:
        if observed > 0 and expected > 0:
            g2 += observed * math.log(observed / expected)
    return 2.0 * g2


def _chi_square_and_p(a: int, b: int, total_a: int, total_b: int) -> tuple[float, float, float]:
    """Pearson χ², p-value, and phi effect size for the 2×2 table."""
    table = [[a, total_a - a], [b, total_b - b]]
    # Guard empty rows/cols
    if total_a <= 0 or total_b <= 0 or (a + b) == 0 and (total_a - a + total_b - b) == 0:
        return 0.0, 1.0, 0.0
    try:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    except ValueError:
        return 0.0, 1.0, 0.0
    n = total_a + total_b
    phi = math.sqrt(chi2 / n) if n > 0 and chi2 >= 0 else 0.0
    return float(chi2), float(p), float(phi)


def _fisher(a: int, b: int, total_a: int, total_b: int) -> tuple[float, float]:
    """Fisher's exact test → (odds_ratio, two-sided p)."""
    table = [[a, total_a - a], [b, total_b - b]]
    if min(total_a, total_b) <= 0:
        return float("nan"), 1.0
    try:
        odds, p = stats.fisher_exact(table, alternative="two-sided")
    except ValueError:
        return float("nan"), 1.0
    return float(odds), float(p)


def _log_ratio(a: int, b: int, total_a: int, total_b: int) -> float:
    """Hardie-style binary log ratio with +0.5 smoothing."""
    if total_a <= 0 or total_b <= 0:
        return 0.0
    rate_a = (a + 0.5) / total_a
    rate_b = (b + 0.5) / total_b
    return math.log2(rate_a / rate_b)


def _odds_ratio(a: int, b: int, total_a: int, total_b: int) -> float:
    """Haldane–Anscombe corrected odds ratio."""
    a_c = a + 0.5
    b_c = b + 0.5
    a_not = (total_a - a) + 0.5
    b_not = (total_b - b) + 0.5
    return (a_c / a_not) / (b_c / b_not)


def _effect_direction(rate_a: float, rate_b: float) -> str:
    if rate_a > rate_b:
        return "a"
    if rate_b > rate_a:
        return "b"
    return "neutral"


def _expected_min(a: int, b: int, total_a: int, total_b: int) -> float:
    grand = total_a + total_b
    if grand <= 0:
        return 0.0
    row_term = a + b
    return min(
        total_a * row_term / grand if grand else 0.0,
        total_b * row_term / grand if grand else 0.0,
        total_a * (grand - row_term) / grand if grand else 0.0,
        total_b * (grand - row_term) / grand if grand else 0.0,
    )


def keyness_report(
    tokenized_a: list[list[str]],
    tokenized_b: list[list[str]],
    *,
    method: str = "log_likelihood",
    top_n: int = 50,
    min_frequency: int = 1,
    correction: str | None = "bh",
    fisher_expected_threshold: float = 5.0,
    group_a_label: str | None = None,
    group_b_label: str | None = None,
    group_field: str | None = None,
) -> dict[str, Any]:
    """Compare two token groups with G² / χ² / Fisher, effect sizes, and FDR.

    Groups must be chosen by the caller from available metadata; no comparison
    categories are hardcoded here.
    """
    method_name = normalize_keyness_method(method)
    correction_name = normalize_correction(correction)
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if min_frequency < 0:
        raise ValueError("min_frequency must be >= 0")

    counts_a: Counter[str] = Counter(t for tokens in tokenized_a for t in tokens)
    counts_b: Counter[str] = Counter(t for tokens in tokenized_b for t in tokens)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    vocabulary = set(counts_a) | set(counts_b)

    rows: list[dict[str, Any]] = []
    for term in vocabulary:
        a = int(counts_a.get(term, 0))
        b = int(counts_b.get(term, 0))
        if a + b < min_frequency:
            continue

        rate_a = (a / total_a) if total_a else 0.0
        rate_b = (b / total_b) if total_b else 0.0
        g2 = _g2_statistic(a, b, total_a, total_b)
        chi2, chi2_p, phi = _chi_square_and_p(a, b, total_a, total_b)
        fisher_or, fisher_p = _fisher(a, b, total_a, total_b)
        if not math.isfinite(fisher_or):
            fisher_or = None  # type: ignore[assignment]
        g2_p = float(stats.chi2.sf(g2, df=1)) if g2 > 0 else 1.0
        log_ratio = _log_ratio(a, b, total_a, total_b)
        odds_ratio = _odds_ratio(a, b, total_a, total_b)
        expected_min = _expected_min(a, b, total_a, total_b)
        prefer_fisher = expected_min < fisher_expected_threshold

        if method_name == "log_likelihood":
            statistic = g2
            p_value = g2_p
        elif method_name == "chi_square":
            statistic = chi2
            p_value = chi2_p
        else:
            statistic = -math.log10(max(fisher_p, 1e-300))
            p_value = fisher_p

        rows.append(
            {
                "feature": term,
                "freq_a": a,
                "freq_b": b,
                "rate_a": rate_a,
                "rate_b": rate_b,
                "keyness_statistic": float(statistic),
                "g2": float(g2),
                "g2_p_value": float(g2_p),
                "chi_square": float(chi2),
                "chi_square_p_value": float(chi2_p),
                "phi": float(phi),
                "fisher_odds_ratio": fisher_or,
                "fisher_p_value": float(fisher_p),
                "p_value": float(p_value),
                "effect_direction": _effect_direction(rate_a, rate_b),
                "log_ratio": float(log_ratio),
                "odds_ratio": float(odds_ratio),
                "effect_size": float(log_ratio),
                "expected_min": float(expected_min),
                "fisher_recommended": prefer_fisher,
                "method": method_name,
            }
        )

    # Sort by primary statistic (desc); for fisher already -log10(p).
    rows.sort(key=lambda row: (-row["keyness_statistic"], row["feature"]))

    # Apply BH to the full ranked candidate set before truncating, using the
    # p-values of all tested features (more honest than adjusting only top_n).
    p_adjusted_all = (
        benjamini_hochberg([row["p_value"] for row in rows]) if correction_name == "bh" else None
    )
    if p_adjusted_all is not None:
        for row, adj in zip(rows, p_adjusted_all, strict=True):
            row["p_adjusted"] = adj
            row["correction"] = "bh"
    else:
        for row in rows:
            row["p_adjusted"] = row["p_value"]
            row["correction"] = "none"

    limited = rows[:top_n]
    return {
        "method": method_name,
        "correction": correction_name or "none",
        "min_frequency": min_frequency,
        "top_n": top_n,
        "features_tested": len(rows),
        "features_returned": len(limited),
        "group_field": group_field,
        "group_a": {
            "label": group_a_label,
            "unit_count": len(tokenized_a),
            "token_count": total_a,
        },
        "group_b": {
            "label": group_b_label,
            "unit_count": len(tokenized_b),
            "token_count": total_b,
        },
        "features": limited,
        # Transparency: significance alone is insufficient for large corpora.
        "notes": [
            "Effect sizes (log_ratio, odds_ratio) are reported alongside p-values.",
            "Benjamini–Hochberg FDR is applied across all tested features when correction='bh'.",
            "Fisher is flagged when any expected cell is below the configured threshold.",
            "Comparison groups must be chosen from corpus metadata; none are hardcoded.",
        ],
    }


def keyness(
    tokenized_a: list[list[str]],
    tokenized_b: list[list[str]],
    top_n: int = 50,
    *,
    method: str = "log_likelihood",
    min_frequency: int = 1,
    correction: str | None = "bh",
    group_a_label: str | None = None,
    group_b_label: str | None = None,
    group_field: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible list of keyness feature rows (enriched)."""
    report = keyness_report(
        tokenized_a,
        tokenized_b,
        method=method,
        top_n=top_n,
        min_frequency=min_frequency,
        correction=correction,
        group_a_label=group_a_label,
        group_b_label=group_b_label,
        group_field=group_field,
    )
    return report["features"]


def describe_keyness_capabilities() -> dict[str, Any]:
    return {
        "methods": ["log_likelihood", "chi_square", "fisher"],
        "corrections": ["bh", "none"],
        "effect_sizes": ["log_ratio", "odds_ratio", "phi"],
        "group_selection": "caller-supplied metadata filters only (no hardcoded categories)",
    }
