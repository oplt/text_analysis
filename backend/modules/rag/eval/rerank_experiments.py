"""Offline rerank depth comparison (0 / 20 / 40) over fixture lists.

Default justification: FACT skips rerank (depth 0); evidence caps at 20; broader
intents may use up to 40. Live ANN/LLM-rerank quality is NEEDS_LIVE_MEASUREMENT.
"""

from __future__ import annotations

import json
from typing import Any

from backend.modules.rag.application.rerank_policy import apply_rerank_policy
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.eval.chunking_experiments import ranking_metrics_from_fixture


class _IdentityReranker:
    """Deterministic fixture ranker: reverse the leading window to simulate change."""

    name = "fixture-reverse"
    version = "v1"

    def rerank(self, query, chunks, *, limit):
        del query
        window = list(chunks[:limit])
        window.reverse()
        return window


def _chunks_from_ids(ranked_ids: list[str]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=chunk_id.split("#", 1)[0],
            content=chunk_id,
            score=1.0 - (index * 0.01),
            filename=f"{chunk_id}.txt",
            chunk_index=index,
        )
        for index, chunk_id in enumerate(ranked_ids)
    ]


def compare_rerank_depths(
    *,
    ranked_ids: list[str],
    relevant_ids: set[str],
    depths: tuple[int, ...] = (0, 20, 40),
    query: str = "fixture query",
) -> dict[str, Any]:
    ranker = _IdentityReranker()
    baseline = _chunks_from_ids(ranked_ids)
    comparisons = []
    for depth in depths:
        result = apply_rerank_policy(ranker, query, baseline, depth=depth)
        ordered_ids = [chunk.chunk_id for chunk in result.chunks]
        metrics = ranking_metrics_from_fixture(ordered_ids, relevant_ids, k=5)
        comparisons.append(
            {
                "depth": depth,
                "ordered_ids": ordered_ids,
                "failed": result.failed,
                "metrics": metrics,
            }
        )
    return {
        "schema_version": 1,
        "query": query,
        "relevant_ids": sorted(relevant_ids),
        "depths": comparisons,
        "default_justification": (
            "depth 0 for FACT; evidence override ~20; max_depth 40 for broad intents. "
            "Fixture reverse-ranker is not a production quality claim."
        ),
    }


def run_rerank_experiment_report() -> dict[str, Any]:
    fixtures = [
        {
            "id": "governance",
            "ranked_ids": [
                "fixture-governance#0",
                "fixture-participation#0",
                "fixture-multilingual#0",
            ]
            + [f"noise-{i}" for i in range(25)],
            "relevant_ids": {"fixture-governance#0"},
        },
        {
            "id": "participation",
            "ranked_ids": [
                "fixture-participation#0",
                "fixture-governance#0",
            ]
            + [f"noise-{i}" for i in range(45)],
            "relevant_ids": {"fixture-participation#0"},
        },
    ]
    return {
        "schema_version": 1,
        "cases": [
            {
                "id": item["id"],
                **compare_rerank_depths(
                    ranked_ids=item["ranked_ids"],
                    relevant_ids=item["relevant_ids"],
                ),
            }
            for item in fixtures
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_rerank_experiment_report(), indent=2, sort_keys=True))
