"""Token-bounded, deterministic packing for corpus-synthesis reduction nodes."""

from __future__ import annotations

from typing import Any

from backend.lib.vectors import estimate_tokens
from backend.modules.text_research.domain.models import dumps

REDUCTION_PROMPT_OVERHEAD_TOKENS = 256


def pack_reduction_nodes(
    nodes: list[dict[str, Any]],
    *,
    max_tokens: int,
    max_items: int,
) -> list[list[dict[str, Any]]]:
    """Pack ordered nodes without ever constructing an over-budget prompt payload."""
    payload_budget = max_tokens - REDUCTION_PROMPT_OVERHEAD_TOKENS
    if payload_budget <= 0:
        raise ValueError("synthesis reduction token budget is too small")

    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    used = 0
    for node in nodes:
        cost = estimate_tokens(dumps(node))
        if cost > payload_budget:
            raise ValueError("A synthesis reduction node exceeds the configured token budget")
        if batch and (used + cost > payload_budget or len(batch) >= max_items):
            batches.append(batch)
            batch = []
            used = 0
        batch.append(node)
        used += cost
    if batch:
        batches.append(batch)
    return batches


def checkpoint_key(*, level: int, batch: int) -> str:
    """Stable JSON-dict key for hierarchical reduce checkpoints."""
    return f"level-{int(level)}/batch-{int(batch)}"


def write_reduce_checkpoint(
    store: dict[str, Any] | None,
    *,
    level: int,
    batch: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist a reduction checkpoint keyed by level/batch (JSON-serializable)."""
    checkpoints = dict(store or {})
    checkpoints[checkpoint_key(level=level, batch=batch)] = {
        "level": int(level),
        "batch": int(batch),
        **payload,
    }
    return checkpoints


def read_reduce_checkpoint(
    store: dict[str, Any] | None, *, level: int, batch: int
) -> dict[str, Any] | None:
    if not store:
        return None
    value = store.get(checkpoint_key(level=level, batch=batch))
    return dict(value) if isinstance(value, dict) else None


def filter_claims_to_supporting_chunks(
    claims: list[Any],
    supporting_chunk_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject claim chunk IDs that escape the supporting map/reduce allow-list.

    Returns (kept_claims, rejected_claims). Claims whose every chunk_id escapes
    are dropped entirely; partial escapes keep only the allowed IDs.
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for claim in claims:
        if isinstance(claim, dict):
            text = claim.get("text", "")
            chunk_ids = list(claim.get("chunk_ids") or [])
            citation_numbers = list(claim.get("citation_numbers") or [])
        else:
            text = getattr(claim, "text", "")
            chunk_ids = list(getattr(claim, "chunk_ids", None) or [])
            citation_numbers = list(getattr(claim, "citation_numbers", None) or [])
        allowed = [chunk_id for chunk_id in chunk_ids if chunk_id in supporting_chunk_ids]
        escaped = [chunk_id for chunk_id in chunk_ids if chunk_id not in supporting_chunk_ids]
        if escaped:
            rejected.append(
                {
                    "text": text,
                    "escaped_chunk_ids": escaped,
                    "original_chunk_ids": chunk_ids,
                }
            )
        if chunk_ids and not allowed:
            continue
        kept.append(
            {
                "text": text,
                "chunk_ids": allowed,
                "citation_numbers": citation_numbers,
            }
        )
    return kept, rejected
