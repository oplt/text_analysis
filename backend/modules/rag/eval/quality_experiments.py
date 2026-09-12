"""Deterministic experiment matrix for measured RAG quality/default selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalExperiment:
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    dense_candidates: int
    lexical_candidates: int
    rrf_k: int
    rerank_depth: int
    parent_expansion: bool
    max_chunks_per_document: int


def experiment_matrix() -> list[RetrievalExperiment]:
    """Return the target matrix; execution is intentionally opt-in and benchmark-backed."""
    return [
        RetrievalExperiment(*values)
        for values in product(
            (256, 384, 512),
            (0, 32, 64),
            (20, 40, 80),
            (20, 40, 80),
            (30, 60, 90),
            (0, 20, 40),
            (False, True),
            (2, 3, 5),
        )
    ]


def experiment_report(
    experiment: RetrievalExperiment,
    metrics: dict[str, float],
    *,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Serialize an auditable candidate result without selecting a global default."""
    required = {
        "recall_at_5", "recall_at_10", "precision_at_k", "mrr", "ndcg_at_10",
        "source_diversity", "duplicate_ratio", "citation_precision", "citation_recall",
        "citation_structural_validity", "unsupported_claim_rate", "context_tokens",
        "retrieval_p50_ms", "retrieval_p95_ms", "retrieval_p99_ms",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Experiment metrics are incomplete: {sorted(missing)}")
    return {
        "schema_version": 1,
        "experiment": asdict(experiment),
        "metrics": metrics,
        "environment": environment,
        "default_selection": "manual_review_required",
    }
