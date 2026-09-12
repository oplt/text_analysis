"""Centralized workload estimation for research analysis dispatch.

Used by :mod:`analysis_executor` to decide inline ``asyncio.to_thread`` vs
Celery enqueue. Thresholds live on settings (``RESEARCH_ASYNC_*``).

Pair/cost models mirror the algorithms after TASK-007:
- pairwise + top_k → bounded heap/block work (``min(all_pairs, N·top_k)``)
- query → N comparisons
- group_centroid → G(G-1)/2
- co-occurrence → token-window ops (caller passes ``estimated_pairs``)
- topic/model sweeps → models × corpus scale
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadEstimate:
    """Dimensions used to classify a research job as small or large."""

    analysis_type: str
    n_units: int = 0
    n_documents: int = 0
    estimated_tokens: int = 0
    estimated_features: int = 0
    requested_top_k: int = 0
    n_clusters: int = 0
    n_topic_models: int = 0
    n_seed_runs: int = 0
    estimated_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_token_count(texts: Sequence[str] | None, *, chars_per_token: float = 4.0) -> int:
    """Cheap token estimate from character length (no tokenizer dependency)."""
    if not texts:
        return 0
    total_chars = sum(len(text or "") for text in texts)
    if total_chars <= 0:
        return 0
    return max(1, int(total_chars / max(chars_per_token, 1.0)))


def estimate_pair_count(
    n_units: int,
    *,
    mode: str = "pairwise",
    top_k: int = 0,
    n_groups: int = 0,
    window_size: int = 0,
    n_models: int = 0,
) -> int:
    """Pair/operation-count estimate calibrated to analysis algorithm cost.

    Modes:
    - ``pairwise`` / ``all_pairs`` / ``duplicate``: undirected pairs; when
      ``top_k > 0`` use bounded ``min(all_pairs, N·top_k)`` (TASK-007 path).
    - ``query`` / ``top_k``: one score per candidate (N).
    - ``group_centroid``: undirected group pairs.
    - ``cooccurrence``: sliding-window token ops ≈ N · window.
    - ``sweep`` / ``model_sweep``: models × corpus scale.
    """
    n = max(0, int(n_units))
    k = max(0, int(top_k))
    key = (mode or "pairwise").strip().lower().replace("-", "_")

    if key in {"query", "top_k"}:
        return n

    if key in {"group_centroid", "centroid", "between_groups"}:
        g = max(0, int(n_groups))
        if g < 2:
            # Fall back to unit count when caller did not pass group cardinality.
            g = n
        if g < 2:
            return 0
        return g * (g - 1) // 2

    if key in {"cooccurrence", "collocation", "window"}:
        return n * max(1, int(window_size) if window_size else 1)

    if key in {"sweep", "model_sweep", "topic_sweep"}:
        models = max(1, int(n_models) if n_models else 1)
        return models * max(n, 1)

    if n < 2:
        return 0

    all_pairs = n * (n - 1) // 2
    if key in {"pairwise", "all_pairs", "duplicate"} and k > 0:
        # Blocked top-K still compares broadly, but returned/retained work is
        # bounded; use min(all_pairs, N·k) as the async cost proxy.
        return min(all_pairs, n * k)
    if key in {"pairwise", "all_pairs", "duplicate"}:
        return all_pairs
    return all_pairs


def estimate_workload(
    *,
    analysis_type: str,
    n_units: int = 0,
    n_documents: int = 0,
    texts: Sequence[str] | None = None,
    estimated_tokens: int | None = None,
    estimated_features: int = 0,
    requested_top_k: int = 0,
    n_clusters: int = 0,
    n_topic_models: int = 0,
    n_seed_runs: int = 0,
    estimated_pairs: int | None = None,
    pair_mode: str = "pairwise",
    n_groups: int = 0,
    window_size: int = 0,
) -> WorkloadEstimate:
    """Build a :class:`WorkloadEstimate` from available request dimensions."""
    tokens = int(estimated_tokens) if estimated_tokens is not None else estimate_token_count(texts)
    analysis = (analysis_type or "").strip().lower()
    if estimated_pairs is not None:
        pairs = int(estimated_pairs)
    elif analysis in {"similarity", "duplicate_detection", "cooccurrence"}:
        mode = pair_mode
        if analysis == "cooccurrence" and pair_mode in {"pairwise", "all_pairs"}:
            mode = "cooccurrence"
        if analysis == "duplicate_detection" and pair_mode == "pairwise":
            mode = "duplicate"
        pairs = estimate_pair_count(
            n_units,
            mode=mode,
            top_k=requested_top_k,
            n_groups=n_groups,
            window_size=window_size,
            n_models=n_topic_models,
        )
    elif analysis in {"topic_k_sweep", "topic_seed_stability", "robustness"}:
        pairs = estimate_pair_count(
            n_units,
            mode="sweep",
            n_models=max(n_topic_models, n_seed_runs, 1),
        )
    else:
        pairs = 0
    return WorkloadEstimate(
        analysis_type=analysis_type,
        n_units=max(0, int(n_units)),
        n_documents=max(0, int(n_documents)),
        estimated_tokens=max(0, tokens),
        estimated_features=max(0, int(estimated_features)),
        requested_top_k=max(0, int(requested_top_k)),
        n_clusters=max(0, int(n_clusters)),
        n_topic_models=max(0, int(n_topic_models)),
        n_seed_runs=max(0, int(n_seed_runs)),
        estimated_pairs=max(0, pairs),
    )


def async_thresholds() -> dict[str, int]:
    """Load configured async thresholds (with safe defaults)."""
    try:
        from backend.core.config import settings

        return {
            "unit": int(settings.RESEARCH_ASYNC_UNIT_THRESHOLD),
            "token": int(settings.RESEARCH_ASYNC_TOKEN_THRESHOLD),
            "pair": int(settings.RESEARCH_ASYNC_PAIR_THRESHOLD),
        }
    except Exception:
        return {"unit": 5_000, "token": 500_000, "pair": 2_000_000}


def exceeds_async_threshold(estimate: WorkloadEstimate) -> bool:
    """True when any primary dimension exceeds its configured threshold."""
    thresholds = async_thresholds()
    if estimate.n_units >= thresholds["unit"]:
        return True
    if estimate.estimated_tokens >= thresholds["token"]:
        return True
    if estimate.estimated_pairs >= thresholds["pair"]:
        return True
    # Multiplicative CPU amplifiers even below unit/token caps.
    if estimate.n_topic_models >= 5 or estimate.n_seed_runs >= 5:
        return True
    return estimate.n_clusters >= 50 and estimate.n_units >= max(100, thresholds["unit"] // 10)
