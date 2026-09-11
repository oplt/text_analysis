"""Retrieval intent → planner parameters (backend-owned; not hardcoded in React)."""

from __future__ import annotations

from dataclasses import dataclass

from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.infrastructure.rag_config import RagConfig


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    intent: RetrievalIntent
    top_k: int
    dense_candidates: int
    lexical_candidates: int
    diversify: bool
    max_per_document: int
    multi_query: bool


def plan_retrieval(
    intent: RetrievalIntent | str | None,
    config: RagConfig,
    *,
    top_k: int | None = None,
    retrieval_mode: str | None = None,
) -> RetrievalPlan:
    resolved_intent = (
        RetrievalIntent(intent)
        if isinstance(intent, str) and intent
        else (intent or RetrievalIntent.EVIDENCE)
    )
    if not isinstance(resolved_intent, RetrievalIntent):
        resolved_intent = RetrievalIntent.EVIDENCE

    base_k = top_k or getattr(config, "top_k", 5)
    evidence_top_k = getattr(config, "evidence_top_k", max(base_k, 12))
    dense_candidates = getattr(config, "dense_candidates", max(base_k * 3, 40))
    lexical_candidates = getattr(config, "lexical_candidates", max(base_k * 3, 40))
    max_per_document = getattr(config, "source_max_chunks_per_document", 3)

    diversify = resolved_intent in {
        RetrievalIntent.COMPARISON,
        RetrievalIntent.SYNTHESIS,
        RetrievalIntent.EVIDENCE,
        RetrievalIntent.CONTRADICTION,
    }
    multi_query = resolved_intent == RetrievalIntent.CONTRADICTION

    if resolved_intent in {RetrievalIntent.FACT, RetrievalIntent.DEFINITION}:
        final_k = min(base_k, 5)
        dense = min(dense_candidates, max(final_k * 3, final_k))
        lexical = min(lexical_candidates, max(final_k * 3, final_k))
    elif resolved_intent == RetrievalIntent.SEMANTIC_SEARCH:
        # Dense-primary; hybrid unless retrieval_mode overrides
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates
    elif resolved_intent in {RetrievalIntent.EVIDENCE, RetrievalIntent.COMPARISON}:
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates
    elif resolved_intent == RetrievalIntent.SYNTHESIS:
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates
    else:  # contradiction
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates

    mode = (retrieval_mode or "hybrid").strip().lower()
    if mode == "dense" or mode == "semantic":
        lexical = 0
    elif mode == "lexical":
        dense = 0
    # hybrid / unknown → keep both

    return RetrievalPlan(
        intent=resolved_intent,
        top_k=final_k,
        dense_candidates=dense,
        lexical_candidates=lexical,
        diversify=diversify,
        max_per_document=max_per_document,
        multi_query=multi_query,
    )
