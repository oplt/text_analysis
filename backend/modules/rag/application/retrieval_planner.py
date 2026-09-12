"""Retrieval intent → planner parameters (backend-owned; not hardcoded in React)."""

from __future__ import annotations

from dataclasses import dataclass

from backend.modules.rag.domain.enums import RetrievalIntent
from backend.modules.rag.infrastructure.rag_config import RagConfig


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """Bounded retrieval settings selected by deterministic query analysis."""

    intent: RetrievalIntent
    top_k: int
    dense_candidates: int
    lexical_candidates: int
    diversify: bool
    max_per_document: int
    multi_query: bool
    lexical_phrase_boost: bool = False
    rerank_depth: int = 0
    rerank_method: str = "noop"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "top_k": self.top_k,
            "dense_candidates": self.dense_candidates,
            "lexical_candidates": self.lexical_candidates,
            "diversify": self.diversify,
            "max_per_document": self.max_per_document,
            "multi_query": self.multi_query,
            "lexical_phrase_boost": self.lexical_phrase_boost,
            "rerank_depth": self.rerank_depth,
            "rerank_method": self.rerank_method,
        }


# Existing callers import RetrievalPlan; retain that public name during the profile rollout.
RetrievalPlan = RetrievalProfile


def plan_retrieval(
    intent: RetrievalIntent | str | None,
    config: RagConfig,
    *,
    top_k: int | None = None,
    retrieval_mode: str | None = None,
    lexical_phrase_boost: bool = False,
) -> RetrievalProfile:
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
    rerank_cap = max(0, int(getattr(config, "rerank_max_depth", 40)))
    rerank_enabled = bool(getattr(config, "rerank_enabled", False))
    rerank_method = (
        "unicode_heuristic"
        if bool(getattr(config, "rerank_heuristic_enabled", False))
        else "noop"
    )

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
    elif resolved_intent in {
        RetrievalIntent.EVIDENCE,
        RetrievalIntent.COMPARISON,
        RetrievalIntent.SYNTHESIS,
    }:
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates
    else:  # contradiction
        final_k = max(base_k, evidence_top_k)
        dense = dense_candidates
        lexical = lexical_candidates

    rerank_depth = 0 if resolved_intent == RetrievalIntent.FACT else rerank_cap
    overrides = getattr(config, "rerank_intent_depth_overrides", None) or {}
    if isinstance(overrides, dict) and resolved_intent.value in overrides:
        rerank_depth = max(0, int(overrides[resolved_intent.value]))
    if not rerank_enabled:
        rerank_depth = 0
    if resolved_intent == RetrievalIntent.FACT:
        rerank_depth = 0

    mode = (retrieval_mode or "hybrid").strip().lower()
    if mode == "dense" or mode == "semantic":
        lexical = 0
    elif mode == "lexical":
        dense = 0
    # hybrid / unknown → keep both

    return RetrievalProfile(
        intent=resolved_intent,
        top_k=final_k,
        dense_candidates=dense,
        lexical_candidates=lexical,
        diversify=diversify,
        max_per_document=max_per_document,
        multi_query=multi_query,
        lexical_phrase_boost=lexical_phrase_boost,
        rerank_depth=rerank_depth,
        rerank_method=rerank_method,
    )
