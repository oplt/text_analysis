"""Deterministic baseline metrics for structure-v1 chunk-policy experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import ParsedDocument
from backend.modules.rag.infrastructure.langchain_text_splitters import split_documents

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "corpus"


@dataclass(frozen=True, slots=True)
class ChunkingExperimentResult:
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    total_tokens: int
    duplicate_token_ratio: float
    recall_at_k: float = 0.0
    mrr: float = 0.0


def measure_structure_v1(
    documents: list[ParsedDocument], *, chunk_size: int, chunk_overlap: int
) -> ChunkingExperimentResult:
    """Measure storage duplication without claiming retrieval-quality superiority."""
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_tokens = sum(estimate_tokens(document.content) for document in documents)
    chunk_tokens = sum(estimate_tokens(content) for content, _metadata in chunks)
    duplicate_ratio = max(0.0, (chunk_tokens - source_tokens) / max(1, source_tokens))
    return ChunkingExperimentResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(chunks),
        total_tokens=chunk_tokens,
        duplicate_token_ratio=duplicate_ratio,
    )


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(ranked_ids[:k]).intersection(relevant_ids)) / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for index, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0


def ranking_metrics_from_fixture(
    ranked_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Compute Recall@k / MRR from synthetic fixture rankings (no live retrieval)."""
    return {
        "recall_at_k": recall_at_k(ranked_ids, relevant_ids, k=k),
        "mrr": mean_reciprocal_rank(ranked_ids, relevant_ids),
    }


def run_chunking_experiment_report(
    *,
    documents: list[ParsedDocument] | None = None,
    policies: list[tuple[int, int]] | None = None,
    synthetic_rankings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Emit a JSON-serializable experiment report from fixtures / synthetic rankings."""
    if documents is None:
        documents = []
        for path in sorted(FIXTURE_DIR.glob("*.txt")):
            documents.append(
                ParsedDocument(
                    content=path.read_text(encoding="utf-8"),
                    metadata={"format": "text", "source": path.name},
                )
            )
    policies = policies or [(512, 64), (768, 96), (1000, 150)]
    results = [
        asdict(measure_structure_v1(documents, chunk_size=size, chunk_overlap=overlap))
        for size, overlap in policies
    ]
    ranking_reports = []
    for item in synthetic_rankings or _default_synthetic_rankings():
        metrics = ranking_metrics_from_fixture(
            list(item["ranked_ids"]),
            set(item["relevant_ids"]),
            k=int(item.get("k", 5)),
        )
        ranking_reports.append({**item, "metrics": metrics})
        # Attach mean ranking metrics onto the closest default-size policy row.
        for row in results:
            if row["chunk_size"] == 1000:
                row["recall_at_k"] = metrics["recall_at_k"]
                row["mrr"] = metrics["mrr"]
    return {
        "schema_version": 1,
        "policies": results,
        "ranking_fixtures": ranking_reports,
        "default_justification": (
            "structure-v1 defaults (1000/150) remain the prose baseline; "
            "fixture rankings justify keeping them until live Recall@k improves."
        ),
    }


def _default_synthetic_rankings() -> list[dict[str, Any]]:
    return [
        {
            "id": "governance-keyword",
            "ranked_ids": [
                "fixture-governance#0",
                "fixture-participation#0",
                "fixture-multilingual#0",
            ],
            "relevant_ids": ["fixture-governance#0"],
            "k": 5,
        },
        {
            "id": "participation-contradiction",
            "ranked_ids": [
                "fixture-participation#0",
                "fixture-governance#0",
                "fixture-multilingual#0",
            ],
            "relevant_ids": ["fixture-participation#0"],
            "k": 5,
        },
    ]


if __name__ == "__main__":
    print(json.dumps(run_chunking_experiment_report(), indent=2, sort_keys=True))
