"""Offline retrieval quality comparison harness (no paid API calls).

Compares ranking strategies over fixture qrels:

  dense-only (score order)
  legacy hybrid rerank (token overlap on dense candidates)
  RRF fusion of two ranked lists
  RRF + source diversify

Usage:
  backend/.venv/bin/python -m backend.modules.rag.eval.retrieval_eval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.modules.rag.application.retrieval_fusion import reciprocal_rank_fusion
from backend.modules.rag.application.retrieval_ranker import HybridRetrievalRanker
from backend.modules.rag.application.source_diversifier import diversify_by_document
from backend.modules.rag.domain.models import RetrievedChunk

FIXTURES = Path(__file__).with_name("fixtures") / "retrieval_qrels.json"


@dataclass(slots=True)
class Metrics:
    recall_at_k: float
    mrr: float
    precision_at_k: float


def _chunk(chunk_id: str, document_id: str, score: float, content: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"text for {chunk_id}",
        score=score,
        filename=f"{document_id}.txt",
        chunk_index=0,
    )


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / len(relevant)


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / k


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for index, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / index
    return 0.0


def evaluate_ranking(ranked_ids: list[str], relevant: set[str], k: int = 5) -> Metrics:
    return Metrics(
        recall_at_k=recall_at_k(ranked_ids, relevant, k),
        mrr=mrr(ranked_ids, relevant),
        precision_at_k=precision_at_k(ranked_ids, relevant, k),
    )


def run_strategies(case: dict) -> dict[str, Metrics]:
    dense = [
        _chunk(c["chunk_id"], c["document_id"], c["score"], c.get("content", ""))
        for c in case["dense"]
    ]
    lexical = [
        _chunk(c["chunk_id"], c["document_id"], c["score"], c.get("content", ""))
        for c in case["lexical"]
    ]
    relevant = set(case["relevant_chunk_ids"])
    query = case["question"]
    k = int(case.get("k", 5))

    dense_ids = [c.chunk_id for c in dense[:k]]
    legacy = HybridRetrievalRanker().rerank(query, dense, limit=k)
    legacy_ids = [c.chunk_id for c in legacy]
    fused = reciprocal_rank_fusion([dense, lexical], k=60, limit=k * 2)
    fused_ids = [c.chunk_id for c in fused[:k]]
    diversified, _ = diversify_by_document(
        fused,
        limit=k,
        max_per_document=2,
        documents_in_scope=len({c.document_id for c in dense + lexical}),
    )
    diversify_ids = [c.chunk_id for c in diversified]

    return {
        "dense_only": evaluate_ranking(dense_ids, relevant, k),
        "legacy_rerank": evaluate_ranking(legacy_ids, relevant, k),
        "rrf_fusion": evaluate_ranking(fused_ids, relevant, k),
        "rrf_diversify": evaluate_ranking(diversify_ids, relevant, k),
    }


def main() -> None:
    payload = json.loads(FIXTURES.read_text())
    for case in payload["cases"]:
        print(f"\n== {case['id']}: {case['question']}")
        for name, metrics in run_strategies(case).items():
            print(
                f"  {name:16} recall@{case.get('k', 5)}="
                f"{metrics.recall_at_k:.3f}  mrr={metrics.mrr:.3f}  "
                f"p@{case.get('k', 5)}={metrics.precision_at_k:.3f}"
            )


if __name__ == "__main__":
    main()
