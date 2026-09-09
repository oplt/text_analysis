"""Graph-ready association network payloads from co-occurrence pairs.

Backend returns nodes / edges / weights / association statistics for downstream
visualization. No semantic labels or domain interpretations are inferred.
"""

from __future__ import annotations

from typing import Any


def build_association_network(
    pairs: list[dict[str, Any]],
    *,
    directed: bool = False,
    weight_field: str = "association_score",
) -> dict[str, Any]:
    """Convert co-occurrence pair rows into a visualization-ready graph.

    Parameters
    ----------
    pairs:
        Rows from :func:`collocation.collocation_report` (or equivalent) with at
        least ``term_a``, ``term_b``, and a weight field.
    directed:
        When True, edges keep ``(term_a → term_b)`` orientation.
    weight_field:
        Field used as the primary edge ``weight`` (typically the selected
        association method score or ``count``).
    """
    if weight_field not in {
        "association_score",
        "count",
        "pmi",
        "npmi",
        "dice",
        "log_dice",
        "t_score",
    }:
        raise ValueError(
            f"Unsupported weight_field {weight_field!r}; expected association_score, "
            f"count, pmi, npmi, dice, log_dice, or t_score"
        )

    degree: dict[str, int] = {}
    weighted_degree: dict[str, float] = {}
    frequency: dict[str, int] = {}
    edges: list[dict[str, Any]] = []

    for pair in pairs:
        term_a = str(pair.get("term_a") or "")
        term_b = str(pair.get("term_b") or "")
        if not term_a or not term_b:
            continue
        count = int(pair.get("count") or 0)
        weight = float(pair.get(weight_field, pair.get("association_score", count)) or 0.0)

        freq_a = pair.get("freq_a")
        freq_b = pair.get("freq_b")
        if isinstance(freq_a, (int, float)):
            frequency[term_a] = max(frequency.get(term_a, 0), int(freq_a))
        if isinstance(freq_b, (int, float)):
            frequency[term_b] = max(frequency.get(term_b, 0), int(freq_b))

        degree[term_a] = degree.get(term_a, 0) + 1
        degree[term_b] = degree.get(term_b, 0) + 1
        weighted_degree[term_a] = weighted_degree.get(term_a, 0.0) + abs(weight)
        weighted_degree[term_b] = weighted_degree.get(term_b, 0.0) + abs(weight)

        statistics = {
            "count": count,
            "association_score": float(pair.get("association_score") or 0.0),
            "pmi": float(pair.get("pmi") or 0.0),
            "npmi": float(pair.get("npmi") or 0.0),
            "dice": float(pair.get("dice") or 0.0),
            "log_dice": float(pair.get("log_dice") or 0.0),
            "t_score": float(pair.get("t_score") or 0.0),
        }
        edge: dict[str, Any] = {
            "source": term_a,
            "target": term_b,
            "weight": weight,
            "weight_field": weight_field,
            "count": count,
            "association_score": statistics["association_score"],
            "association_statistics": statistics,
            "directed": directed,
        }
        # Convenience aliases used by some frontends.
        edge["term_a"] = term_a
        edge["term_b"] = term_b
        edges.append(edge)

    nodes = [
        {
            "id": term,
            "label": term,
            "frequency": frequency.get(term),
            "degree": degree.get(term, 0),
            "weighted_degree": weighted_degree.get(term, 0.0),
        }
        for term in sorted(degree.keys())
    ]

    return {
        "directed": directed,
        "weight_field": weight_field,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        # Explicitly non-interpretive — consumers must not treat ids as theories.
        "interpretation": None,
        "notes": [
            "Structural word/feature association graph derived from co-occurrence pairs.",
            "No semantic categories or domain meanings are assigned by the backend.",
        ],
    }


def describe_association_network_capabilities() -> dict[str, Any]:
    return {
        "outputs": ["nodes", "edges", "weights", "association_statistics"],
        "weight_fields": [
            "association_score",
            "count",
            "pmi",
            "npmi",
            "dice",
            "log_dice",
            "t_score",
        ],
        "semantic_interpretation": False,
    }
