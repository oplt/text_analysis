"""Collocation / co-occurrence association statistics.

Association method is always caller-selected — nothing is hardcoded as the sole
score. Window size, directionality, and frequency thresholds are configurable.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Literal

ASSOCIATION_METHODS: frozenset[str] = frozenset(
    {
        "count",
        "pmi",
        "npmi",
        "dice",
        "log_dice",
        "logdice",
        "t_score",
        "tscore",
    }
)

DirectionMode = Literal["undirected", "directional"]


def normalize_association_method(method: str | None) -> str:
    key = (method or "pmi").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "raw": "count",
        "freq": "count",
        "frequency": "count",
        "logdice": "log_dice",
        "tscore": "t_score",
        "t": "t_score",
    }
    canonical = aliases.get(key, key)
    if canonical not in {"count", "pmi", "npmi", "dice", "log_dice", "t_score"}:
        raise ValueError(
            f"Unsupported association method {method!r}; expected one of "
            f"count, pmi, npmi, dice, log_dice, t_score"
        )
    return canonical


def normalize_direction(directional: bool | str | None) -> DirectionMode:
    if directional is None or directional is False:
        return "undirected"
    if directional is True:
        return "directional"
    key = str(directional).strip().lower().replace("-", "_")
    if key in {"undirected", "non_directional", "nondirectional", "symmetric", "false", "0"}:
        return "undirected"
    if key in {"directional", "directed", "asymmetric", "true", "1"}:
        return "directional"
    raise ValueError(
        f"Unsupported direction mode {directional!r}; expected undirected or directional"
    )


def _pmi(count: float, freq_a: float, freq_b: float, n: float) -> float:
    if n <= 0 or count <= 0 or freq_a <= 0 or freq_b <= 0:
        return 0.0
    p_ab = count / n
    p_a = freq_a / n
    p_b = freq_b / n
    return math.log(p_ab / (p_a * p_b))


def _npmi(count: float, freq_a: float, freq_b: float, n: float) -> float:
    if n <= 0 or count <= 0:
        return 0.0
    p_ab = count / n
    pmi = _pmi(count, freq_a, freq_b, n)
    denom = -math.log(p_ab)
    if denom <= 0:
        return 0.0
    return pmi / denom


def _dice(count: float, freq_a: float, freq_b: float) -> float:
    denom = freq_a + freq_b
    if denom <= 0 or count <= 0:
        return 0.0
    return (2.0 * count) / denom


def _log_dice(count: float, freq_a: float, freq_b: float) -> float:
    """Sketch Engine–style logDice: 14 + log2(2f / (f1 + f2))."""
    dice = _dice(count, freq_a, freq_b)
    if dice <= 0:
        return 0.0
    return 14.0 + math.log2(dice)


def _t_score(count: float, freq_a: float, freq_b: float, n: float) -> float:
    if n <= 0 or count <= 0:
        return 0.0
    expected = (freq_a * freq_b) / n
    return (count - expected) / math.sqrt(count)


def score_pair(
    *,
    count: int,
    freq_a: int,
    freq_b: int,
    n_tokens: int,
) -> dict[str, float]:
    c, fa, fb, n = float(count), float(freq_a), float(freq_b), float(n_tokens)
    return {
        "count": float(count),
        "pmi": _pmi(c, fa, fb, n),
        "npmi": _npmi(c, fa, fb, n),
        "dice": _dice(c, fa, fb),
        "log_dice": _log_dice(c, fa, fb),
        "t_score": _t_score(c, fa, fb, n),
    }


def _collect_pairs(
    tokenized: list[list[str]],
    *,
    window: int,
    direction: DirectionMode,
) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    term_counts: Counter[str] = Counter()
    total_tokens = 0

    for tokens in tokenized:
        total_tokens += len(tokens)
        term_counts.update(tokens)
        n = len(tokens)
        for i in range(n):
            if direction == "undirected":
                for j in range(i + 1, min(i + 1 + window, n)):
                    a, b = tokens[i], tokens[j]
                    if a == b:
                        continue
                    pair_counts[tuple(sorted((a, b)))] += 1
            else:
                # Forward window only: (focus, later collocate) — order preserved.
                for j in range(i + 1, min(i + 1 + window, n)):
                    a, b = tokens[i], tokens[j]
                    if a == b:
                        continue
                    pair_counts[(a, b)] += 1
    return pair_counts, term_counts, total_tokens


def collocation_report(
    tokenized: list[list[str]],
    *,
    window: int = 5,
    top_n: int = 100,
    association_method: str = "pmi",
    directional: bool | str | None = False,
    min_frequency: int = 1,
    min_count: int = 1,
    include_network: bool = True,
    network_weight_field: str | None = None,
) -> dict[str, Any]:
    """Compute co-occurrence pairs with selectable association statistics."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if min_frequency < 0:
        raise ValueError("min_frequency must be >= 0")
    if min_count < 0:
        raise ValueError("min_count must be >= 0")

    method = normalize_association_method(association_method)
    direction = normalize_direction(directional)
    pair_counts, term_counts, total_tokens = _collect_pairs(
        tokenized, window=window, direction=direction
    )

    rows: list[dict[str, Any]] = []
    for (term_a, term_b), count in pair_counts.items():
        if count < min_count:
            continue
        freq_a = int(term_counts[term_a])
        freq_b = int(term_counts[term_b])
        if freq_a < min_frequency or freq_b < min_frequency:
            continue
        scores = score_pair(
            count=count, freq_a=freq_a, freq_b=freq_b, n_tokens=total_tokens
        )
        association_score = float(scores[method if method != "count" else "count"])
        if method == "count":
            association_score = float(count)
        rows.append(
            {
                "term_a": term_a,
                "term_b": term_b,
                "count": count,
                "freq_a": freq_a,
                "freq_b": freq_b,
                "association_method": method,
                "association_score": association_score,
                "pmi": scores["pmi"],
                "npmi": scores["npmi"],
                "dice": scores["dice"],
                "log_dice": scores["log_dice"],
                "t_score": scores["t_score"],
                "directional": direction == "directional",
                "window": window,
            }
        )

    if method == "count":
        rows.sort(key=lambda row: (-row["count"], row["term_a"], row["term_b"]))
    else:
        rows.sort(
            key=lambda row: (
                -row["association_score"],
                -row["count"],
                row["term_a"],
                row["term_b"],
            )
        )

    limited = rows[:top_n]
    report: dict[str, Any] = {
        "association_method": method,
        "window": window,
        "direction": direction,
        "directional": direction == "directional",
        "min_frequency": min_frequency,
        "min_count": min_count,
        "top_n": top_n,
        "n_tokens": total_tokens,
        "pairs_tested": len(rows),
        "pairs_returned": len(limited),
        "pairs": limited,
        "available_methods": [
            "count",
            "pmi",
            "npmi",
            "dice",
            "log_dice",
            "t_score",
        ],
    }
    if include_network:
        from backend.modules.text_research.infrastructure.association_network import (
            build_association_network,
        )

        weight_field = network_weight_field or (
            "count" if method == "count" else "association_score"
        )
        report["network"] = build_association_network(
            limited,
            directed=direction == "directional",
            weight_field=weight_field,
        )
    return report


def cooccurrence(
    tokenized: list[list[str]],
    window: int = 5,
    top_n: int = 100,
    *,
    association_method: str = "pmi",
    directional: bool | str | None = False,
    min_frequency: int = 1,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Backward-compatible list of co-occurrence rows."""
    return collocation_report(
        tokenized,
        window=window,
        top_n=top_n,
        association_method=association_method,
        directional=directional,
        min_frequency=min_frequency,
        min_count=min_count,
    )["pairs"]


def describe_collocation_capabilities() -> dict[str, Any]:
    return {
        "association_methods": [
            "count",
            "pmi",
            "npmi",
            "dice",
            "log_dice",
            "t_score",
        ],
        "direction_modes": ["undirected", "directional"],
        "thresholds": ["min_frequency", "min_count"],
        "network": {
            "optional": True,
            "outputs": ["nodes", "edges", "weights", "association_statistics"],
            "semantic_interpretation": False,
        },
        "notes": [
            "association_method selects the ranking / association_score field.",
            "All listed statistics are still attached to each pair row.",
            "include_network=True adds a graph-ready nodes/edges payload.",
        ],
    }
