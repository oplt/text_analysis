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
from typing import Any

from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.application.retrieval_fusion import reciprocal_rank_fusion
from backend.modules.rag.application.retrieval_ranker import HybridRetrievalRanker
from backend.modules.rag.application.source_diversifier import diversify_by_document
from backend.modules.rag.domain.models import RetrievedChunk

FIXTURES = Path(__file__).with_name("fixtures") / "retrieval_qrels.json"
REQUIRED_CATEGORIES = {
    "exact lexical matches",
    "semantic paraphrases",
    "multilingual queries",
    "fact lookup",
    "evidence search",
    "comparison",
    "contradiction/counter-evidence",
    "sparse evidence",
    "no evidence",
    "source diversity",
    "parent context",
    "citation fabrication",
    "citation omission",
}


@dataclass(slots=True)
class Metrics:
    recall_at_k: float
    mrr: float
    precision_at_k: float
    ndcg_at_k: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    document_coverage: float = 0.0


@dataclass(slots=True)
class CitationMetrics:
    citation_precision: float
    citation_recall: float
    citation_validity: float
    unsupported_claim_rate: float
    groundedness: float

    @property
    def citation_structural_validity(self) -> float:
        return self.citation_validity


def _chunk(
    chunk_id: str,
    document_id: str,
    score: float,
    content: str = "",
    parent_context_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"text for {chunk_id}",
        score=score,
        filename=f"{document_id}.txt",
        chunk_index=0,
        parent_context_id=parent_context_id,
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

    ideal_ids = sorted(relevant) + [cid for cid in ranked_ids if cid not in relevant]
    ideal = dcg(ideal_ids)
    if ideal <= 0:
        return 0.0
    return dcg(ranked_ids) / ideal


def document_coverage_at_k(
    ranked_ids: list[str], relevant: set[str], chunk_to_document: dict[str, str], k: int
) -> float:
    """Fraction of relevant documents represented in the top-k result set."""
    if not relevant:
        return 0.0
    retrieved_documents = {
        chunk_to_document[chunk_id] for chunk_id in ranked_ids[:k] if chunk_id in chunk_to_document
    }
    return len(retrieved_documents & relevant) / len(relevant)


def evaluate_ranking(
    ranked_ids: list[str],
    relevant: set[str],
    k: int = 5,
    *,
    relevant_document_ids: set[str] | None = None,
    chunk_to_document: dict[str, str] | None = None,
) -> Metrics:
    chunk_to_document = chunk_to_document or {}
    document_coverage = (
        document_coverage_at_k(
            ranked_ids,
            relevant_document_ids or set(),
            chunk_to_document,
            10,
        )
        if relevant_document_ids is not None
        else 0.0
    )
    return Metrics(
        recall_at_k=recall_at_k(ranked_ids, relevant, k),
        mrr=mrr(ranked_ids, relevant),
        precision_at_k=precision_at_k(ranked_ids, relevant, k),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevant, k),
        recall_at_5=recall_at_k(ranked_ids, relevant, 5),
        recall_at_10=recall_at_k(ranked_ids, relevant, 10),
        ndcg_at_10=ndcg_at_k(ranked_ids, relevant, 10),
        document_coverage=document_coverage,
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
    validity = (
        1.0
        if validated.citation_validation_status in {"valid", "no_evidence"}
        else (0.5 if validated.citation_validation_status == "partial" else 0.0)
    )
    claims = validated.claims
    unsupported = sum(1 for claim in claims if not claim.chunk_ids)
    unsupported_rate = (
        unsupported / len(claims)
        if claims
        else (1.0 if validated.citation_validation_status != "valid" else 0.0)
    )
    grounded = (
        1.0 - unsupported_rate
        if claims
        else (1.0 if validated.citation_validation_status == "valid" and not cited else 0.0)
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
        _chunk(
            c["chunk_id"],
            c["document_id"],
            c["score"],
            c.get("content", ""),
            c.get("parent_context_id"),
        )
        for c in case["dense"]
    ]
    lexical = [
        _chunk(
            c["chunk_id"],
            c["document_id"],
            c["score"],
            c.get("content", ""),
            c.get("parent_context_id"),
        )
        for c in case["lexical"]
    ]
    relevant = set(case["relevant_chunk_ids"])
    relevant_documents = set(case.get("relevant_document_ids", []))
    by_chunk = {chunk.chunk_id: chunk.document_id for chunk in dense + lexical}
    if not relevant_documents:
        relevant_documents = {by_chunk[cid] for cid in relevant if cid in by_chunk}
    query = case["question"]
    k = int(case.get("k", 5))

    dense_ids = [c.chunk_id for c in dense[: max(k, 10)]]
    lexical_ids = [c.chunk_id for c in lexical[: max(k, 10)]]
    legacy = HybridRetrievalRanker().rerank(query, dense, limit=max(k, 10))
    legacy_ids = [c.chunk_id for c in legacy]
    fused = reciprocal_rank_fusion([dense, lexical], k=60, limit=max(k, 10))
    fused_ids = [c.chunk_id for c in fused]
    diversified, _ = diversify_by_document(
        fused,
        limit=max(k, 10),
        max_per_document=2,
        documents_in_scope=len({c.document_id for c in dense + lexical}),
    )
    diversify_ids = [c.chunk_id for c in diversified]

    return {
        "dense_only": evaluate_ranking(
            dense_ids,
            relevant,
            k,
            relevant_document_ids=relevant_documents,
            chunk_to_document=by_chunk,
        ),
        "lexical_only": evaluate_ranking(
            lexical_ids,
            relevant,
            k,
            relevant_document_ids=relevant_documents,
            chunk_to_document=by_chunk,
        ),
        "rrf_hybrid": evaluate_ranking(
            fused_ids,
            relevant,
            k,
            relevant_document_ids=relevant_documents,
            chunk_to_document=by_chunk,
        ),
        # Retained aliases keep existing consumers and CI snapshots compatible.
        "legacy_rerank": evaluate_ranking(legacy_ids, relevant, k),
        "rrf_fusion": evaluate_ranking(
            fused_ids,
            relevant,
            k,
            relevant_document_ids=relevant_documents,
            chunk_to_document=by_chunk,
        ),
        "rrf_diversify": evaluate_ranking(diversify_ids, relevant, k),
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return all ranking and citation metrics for one deterministic case."""
    strategies = run_strategies(case)
    result: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category"),
        "strategies": {
            name: {
                "recall_at_5": metrics.recall_at_5,
                "recall_at_10": metrics.recall_at_10,
                "mrr": metrics.mrr,
                "ndcg_at_10": metrics.ndcg_at_10,
                "document_coverage": metrics.document_coverage,
            }
            for name, metrics in strategies.items()
            if name in {"dense_only", "lexical_only", "rrf_hybrid"}
        },
    }
    if "citation_raw" in case:
        retrieved = [
            _chunk(
                c["chunk_id"],
                c["document_id"],
                c["score"],
                c.get("content", ""),
                c.get("parent_context_id"),
            )
            for c in case["dense"] + case["lexical"]
        ]
        by_id = {chunk.chunk_id: chunk for chunk in retrieved}
        citation = evaluate_citations(
            raw_output=case["citation_raw"],
            retrieved=list(by_id.values()),
            gold_chunk_ids=set(case.get("citation_gold_chunk_ids", case["relevant_chunk_ids"])),
            allowed_document_ids=case.get("scoped_document_ids"),
        )
        result["citation"] = {
            "structural_validity": citation.citation_structural_validity,
            "citation_precision": citation.citation_precision,
            "citation_recall": citation.citation_recall,
            "unsupported_claim_rate": citation.unsupported_claim_rate,
        }
    return result


def run_benchmark(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run every fixture case and aggregate metrics by strategy."""
    payload = payload or json.loads(FIXTURES.read_text())
    cases = [evaluate_case(case) for case in payload["cases"]]
    strategy_names = ("dense_only", "lexical_only", "rrf_hybrid")
    ranking: dict[str, dict[str, float]] = {}
    for strategy in strategy_names:
        values = [case["strategies"][strategy] for case in cases]
        ranking[strategy] = {
            metric: sum(value[metric] for value in values) / len(values) for metric in values[0]
        }
    citation_cases = [case["citation"] for case in cases if "citation" in case]
    citation = (
        {
            metric: sum(value[metric] for value in citation_cases) / len(citation_cases)
            for metric in citation_cases[0]
        }
        if citation_cases
        else {}
    )
    categories = {case.get("category") for case in payload["cases"]}
    missing_categories = sorted(REQUIRED_CATEGORIES - categories)
    if missing_categories:
        raise ValueError(f"Retrieval benchmark is missing categories: {missing_categories}")
    return {
        "schema_version": 2,
        "case_count": len(cases),
        "categories": sorted(categories),
        "ranking": ranking,
        "citation": citation,
        "cases": cases,
    }


def run_ci_regression() -> dict:
    """Tiny CI regression over fixture corpus — fails if RRF recall collapses."""
    payload = json.loads(FIXTURES.read_text())
    summaries: dict[str, float] = {}
    for case in payload["cases"]:
        metrics = run_strategies(case)
        summaries[case["id"]] = metrics["rrf_hybrid"].recall_at_5
        summaries[f"{case['id']}_rrf_recall_at_10"] = metrics["rrf_hybrid"].recall_at_10
        if "citation_raw" in case:
            retrieved = [
                _chunk(
                    c["chunk_id"],
                    c["document_id"],
                    c["score"],
                    c.get("content", ""),
                    c.get("parent_context_id"),
                )
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
            if name not in {"dense_only", "lexical_only", "rrf_hybrid"}:
                continue
            print(
                f"  {name:16} recall@5={metrics.recall_at_5:.3f}  "
                f"recall@10={metrics.recall_at_10:.3f}  mrr={metrics.mrr:.3f}  "
                f"ndcg@10={metrics.ndcg_at_10:.3f}  "
                f"doc_coverage={metrics.document_coverage:.3f}"
            )
    report = run_benchmark(payload)
    print("\nAggregate ranking:", json.dumps(report["ranking"], sort_keys=True))
    print("Aggregate citation:", json.dumps(report["citation"], sort_keys=True))
    print("\nCI regression snapshot:", run_ci_regression())


if __name__ == "__main__":
    main()
