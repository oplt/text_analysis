"""Integration evaluation contract for real RAG ingestion and SQL retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.models import RetrievalOutcome
from backend.modules.rag.eval.retrieval_eval import evaluate_ranking

FIXTURES = Path(__file__).with_name("fixtures") / "end_to_end_qrels.json"
REQUIRED_CATEGORIES = {
    "keyword",
    "semantic",
    "multilingual",
    "comparison",
    "contradiction",
    "synthesis",
    "no_answer",
}


@dataclass(frozen=True, slots=True)
class EndToEndCase:
    id: str
    query: str
    intent: str | None
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    expected_source_ids: tuple[str, ...]
    no_answer: bool
    category: str


class RetrievalRunner(Protocol):
    async def retrieve(self, query: str, **kwargs: Any) -> RetrievalOutcome: ...


def load_end_to_end_qrels(path: Path = FIXTURES) -> list[EndToEndCase]:
    payload = json.loads(path.read_text())
    cases = [
        EndToEndCase(
            id=item["id"],
            query=item["query"],
            intent=item.get("intent"),
            expected_document_ids=tuple(item.get("expected_document_ids", [])),
            expected_chunk_ids=tuple(item.get("expected_chunk_ids", [])),
            expected_source_ids=tuple(
                item.get("expected_source_ids", item.get("expected_document_ids", []))
            ),
            no_answer=bool(item.get("no_answer", False)),
            category=item["category"],
        )
        for item in payload["cases"]
    ]
    missing = REQUIRED_CATEGORIES - {case.category for case in cases}
    if missing:
        raise ValueError(f"End-to-end qrels are missing categories: {sorted(missing)}")
    return cases


def structural_citation_validity(
    *,
    claim_chunk_ids: list[str],
    retrieved_chunk_ids: set[str],
) -> dict[str, float]:
    """Placeholder structural citation checks (no entailment model)."""
    if not claim_chunk_ids:
        return {
            "citation_structural_validity": 1.0,
            "unsupported_claim_rate": 0.0,
        }
    supported = [chunk_id for chunk_id in claim_chunk_ids if chunk_id in retrieved_chunk_ids]
    unsupported = len(claim_chunk_ids) - len(supported)
    return {
        "citation_structural_validity": len(supported) / len(claim_chunk_ids),
        "unsupported_claim_rate": unsupported / len(claim_chunk_ids),
    }


async def run_end_to_end_benchmark(
    retrieval: RetrievalRunner | RetrievalService,
    *,
    user_id: str,
    project_id: str | None,
    document_ids: list[str],
    cases: list[EndToEndCase] | None = None,
    claim_chunk_ids_by_case: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Evaluate the actual retrieval service against an already-ingested fixture corpus."""
    reports: list[dict[str, Any]] = []
    claim_map = claim_chunk_ids_by_case or {}
    for case in cases or load_end_to_end_qrels():
        started = perf_counter()
        outcome = await retrieval.retrieve(
            case.query,
            user_id=user_id,
            project_id=project_id,
            filters={"document_ids": document_ids, "owner_scoped": False},
            intent=case.intent,
        )
        latency_ms = int((perf_counter() - started) * 1000)
        chunk_ids = [chunk.chunk_id for chunk in outcome.chunks]
        document_ids_found = [chunk.document_id for chunk in outcome.chunks]
        metrics = evaluate_ranking(
            chunk_ids,
            set(case.expected_chunk_ids),
            k=5,
            relevant_document_ids=set(case.expected_document_ids),
            chunk_to_document={chunk.chunk_id: chunk.document_id for chunk in outcome.chunks},
        )
        duplicate_ratio = 1 - (len(set(chunk_ids)) / len(chunk_ids)) if chunk_ids else 0.0
        citation_metrics = structural_citation_validity(
            claim_chunk_ids=claim_map.get(case.id, list(case.expected_chunk_ids)),
            retrieved_chunk_ids=set(chunk_ids) | set(case.expected_chunk_ids),
        )
        reports.append(
            {
                "id": case.id,
                "category": case.category,
                "latency_ms": latency_ms,
                "no_answer_expected": case.no_answer,
                "no_matches": outcome.no_matches,
                "recall_at_5": metrics.recall_at_5,
                "mrr": metrics.mrr,
                "ndcg_at_10": metrics.ndcg_at_10,
                "source_diversity": len(set(document_ids_found)),
                "duplicate_ratio": duplicate_ratio,
                "expected_source_ids": list(case.expected_source_ids),
                **citation_metrics,
            }
        )
    latencies = sorted(item["latency_ms"] for item in reports)
    return {
        "schema_version": 1,
        "case_count": len(reports),
        "cases": reports,
        "latency": {
            "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "p95_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0,
        },
    }
