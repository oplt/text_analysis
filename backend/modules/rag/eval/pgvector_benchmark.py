"""Small helpers for comparing filtered ANN output to exact pgvector output."""

from __future__ import annotations

from typing import Any, Callable


def recall_at_k(exact_ids: list[str], ann_ids: list[str], *, k: int) -> float:
    """Return ANN recall against an exact result set under identical filters."""
    expected = set(exact_ids[:k])
    return len(expected.intersection(ann_ids[:k])) / len(expected) if expected else 1.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def exact_filtered_topk(
    query: list[float],
    vectors: dict[str, list[float]],
    *,
    allow_list: set[str] | None,
    k: int,
) -> list[str]:
    """In-memory exact search with an optional allow-list filter."""
    scored = []
    for chunk_id, vector in vectors.items():
        if allow_list is not None and chunk_id not in allow_list:
            continue
        scored.append((cosine_similarity(query, vector), chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _score, chunk_id in scored[:k]]


def filtered_ann_vs_exact(
    *,
    query: list[float],
    vectors: dict[str, list[float]],
    allow_list: set[str] | None,
    k: int,
    ann_search: Callable[[list[float], set[str] | None, int], list[str]] | None = None,
) -> dict[str, Any]:
    """Compare filtered ANN candidates to exact in-memory neighbors.

    When ``ann_search`` is omitted, ANN is approximated by exact search so the
    harness remains runnable without a live pgvector database.
    """
    exact_ids = exact_filtered_topk(query, vectors, allow_list=allow_list, k=k)
    if ann_search is None:
        ann_ids = list(exact_ids)
        mode = "in_memory_exact_as_ann_placeholder"
    else:
        ann_ids = list(ann_search(query, allow_list, k))
        mode = "caller_ann"
    return {
        "schema_version": 1,
        "mode": mode,
        "k": k,
        "allow_list_size": None if allow_list is None else len(allow_list),
        "exact_ids": exact_ids,
        "ann_ids": ann_ids,
        "recall_at_k": recall_at_k(exact_ids, ann_ids, k=k),
        "empty_allow_list_contract": allow_list == set(),
    }


def explain_pgvector_query_plan(
    *,
    db_available: bool = False,
    explain_runner: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Placeholder for live EXPLAIN; skips cleanly when no DB is available."""
    if not db_available or explain_runner is None:
        return {
            "schema_version": 1,
            "skipped": True,
            "reason": "pgvector_unavailable",
            "status": "NEEDS_LIVE_MEASUREMENT",
        }
    plan = explain_runner()
    return {"schema_version": 1, "skipped": False, "plan": plan}
