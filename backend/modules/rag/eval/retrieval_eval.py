"""Offline retrieval + citation quality harness (deterministic metrics).

Compares ranking strategies and citation validity over fixture qrels.

Usage:
  backend/.venv/bin/python -m backend.modules.rag.eval.retrieval_eval
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from backend.modules.rag.application.citation_validation_service import CitationValidationService
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
    ndcg_at_k: float = 0.0


@dataclass(slots=True)
class CitationMetrics:
    citation_precision: float
    citation_recall: float
    citation_validity: float
    unsupported_claim_rate: float
    groundedness: float


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


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    def dcg(ids: list[str]) -> float:
        total = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            rel = 1.0 if cid in relevant else 0.0
            total += rel / math.log2(i + 1)
        return total

    ideal = dcg(sorted(relevant, key=lambda x: x)[:k])  # binary relevance ideal
    # Better ideal: put all relevant first
    ideal_ids = list(relevant) + [cid for cid in ranked_ids if cid not in relevant]
    ideal = dcg(ideal_ids)
    if ideal <= 0:
        return 0.0
    return dcg(ranked_ids) / ideal


def evaluate_ranking(ranked_ids: list[str], relevant: set[str], k: int = 5) -> Metrics:
    return Metrics(
        recall_at_k=recall_at_k(ranked_ids, relevant, k),
        mrr=mrr(ranked_ids, relevant),
        precision_at_k=precision_at_k(ranked_ids, relevant, k),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevant, k),
    )


def evaluate_citations(
    *,
    raw_output: str,
    retrieved: list[RetrievedChunk],
    gold_chunk_ids: set[str],
    allowed_document_ids: list[str] | None,
) -> CitationMetrics:
    validated = CitationValidationService().validate(
        raw_output=raw_output,
        retrieved_chunks=retrieved,
        allowed_document_ids=allowed_document_ids,
    )
    cited = {c.chunk_id for c in validated.citations if c.used_in_answer}
    if validated.citation_validation_status == "unstructured":
        return CitationMetrics(0.0, 0.0, 0.0, 1.0, 0.0)

    precision = len(cited & gold_chunk_ids) / len(cited) if cited else 0.0
    recall = len(cited & gold_chunk_ids) / len(gold_chunk_ids) if gold_chunk_ids else 0.0
    validity = 1.0 if validated.citation_validation_status == "valid" else (
        0.5 if validated.citation_validation_status == "partial" else 0.0
    )
    claims = validated.claims
    unsupported = sum(1 for claim in claims if not claim.chunk_ids)
    unsupported_rate = unsupported / len(claims) if claims else (
        1.0 if validated.citation_validation_status != "valid" else 0.0
    )
    grounded = 1.0 - unsupported_rate if claims else (
        1.0 if validated.citation_validation_status == "valid" and not cited else 0.0
    )
    return CitationMetrics(
        citation_precision=precision,
        citation_recall=recall,
        citation_validity=validity,
        unsupported_claim_rate=unsupported_rate,
        groundedness=grounded,
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


def run_ci_regression() -> dict:
    """Tiny CI regression over fixture corpus — fails if RRF recall collapses."""
    payload = json.loads(FIXTURES.read_text())
    summaries: dict[str, float] = {}
    for case in payload["cases"]:
        metrics = run_strategies(case)
        summaries[case["id"]] = metrics["rrf_fusion"].recall_at_k
        if "citation_raw" in case:
            retrieved = [
                _chunk(c["chunk_id"], c["document_id"], c["score"], c.get("content", ""))
                for c in case["dense"] + case["lexical"]
            ]
            # Deduplicate by chunk_id
            by_id = {c.chunk_id: c for c in retrieved}
            cite = evaluate_citations(
                raw_output=case["citation_raw"],
                retrieved=list(by_id.values()),
                gold_chunk_ids=set(case["relevant_chunk_ids"]),
                allowed_document_ids=case.get("scoped_document_ids"),
            )
            summaries[f"{case['id']}_cite_validity"] = cite.citation_validity
    return summaries


def main() -> None:
    payload = json.loads(FIXTURES.read_text())
    for case in payload["cases"]:
        print(f"\n== {case['id']}: {case['question']}")
        for name, metrics in run_strategies(case).items():
            print(
                f"  {name:16} recall@{case.get('k', 5)}="
                f"{metrics.recall_at_k:.3f}  mrr={metrics.mrr:.3f}  "
                f"p@{case.get('k', 5)}={metrics.precision_at_k:.3f}  "
                f"ndcg={metrics.ndcg_at_k:.3f}"
            )
    print("\nCI regression snapshot:", run_ci_regression())


if __name__ == "__main__":
    main()
