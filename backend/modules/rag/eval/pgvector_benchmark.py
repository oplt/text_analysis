"""Small helpers for comparing filtered ANN output to exact pgvector output."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.lib.vectors import vector_literal
from backend.modules.rag.infrastructure.repositories import RagRepository


def recall_at_k(exact_ids: list[str], ann_ids: list[str], *, k: int) -> float:
    """Return ANN recall against an exact result set under identical filters."""
    expected = set(exact_ids[:k])
    return len(expected.intersection(ann_ids[:k])) / len(expected) if expected else 1.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def exact_filtered_topk(
    query: list[float],
    vectors: dict[str, list[float]],
    *,
    allow_list: set[str] | None,
    k: int,
) -> list[str]:
    """In-memory exact search with an optional allow-list filter."""
    scored = []
    for chunk_id, vector in vectors.items():
        if allow_list is not None and chunk_id not in allow_list:
            continue
        scored.append((cosine_similarity(query, vector), chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _score, chunk_id in scored[:k]]


def filtered_ann_vs_exact(
    *,
    query: list[float],
    vectors: dict[str, list[float]],
    allow_list: set[str] | None,
    k: int,
    ann_search: Callable[[list[float], set[str] | None, int], list[str]] | None = None,
) -> dict[str, Any]:
    """Compare filtered ANN candidates to exact in-memory neighbors.

    When ``ann_search`` is omitted, ANN is approximated by exact search so the
    harness remains runnable without a live pgvector database.
    """
    exact_ids = exact_filtered_topk(query, vectors, allow_list=allow_list, k=k)
    if ann_search is None:
        ann_ids = list(exact_ids)
        mode = "in_memory_exact_as_ann_placeholder"
    else:
        ann_ids = list(ann_search(query, allow_list, k))
        mode = "caller_ann"
    return {
        "schema_version": 1,
        "mode": mode,
        "measurement": "SYNTHETIC" if ann_search is None else "CALLER_SUPPLIED",
        "k": k,
        "allow_list_size": None if allow_list is None else len(allow_list),
        "exact_ids": exact_ids,
        "ann_ids": ann_ids,
        "recall_at_k": recall_at_k(exact_ids, ann_ids, k=k),
        "empty_allow_list_contract": allow_list == set(),
    }


def explain_pgvector_query_plan(
    *,
    db_available: bool = False,
    explain_runner: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Placeholder for live EXPLAIN; skips cleanly when no DB is available."""
    if not db_available or explain_runner is None:
        return {
            "schema_version": 1,
            "skipped": True,
            "reason": "pgvector_unavailable",
            "status": "NEEDS_LIVE_MEASUREMENT",
        }
    plan = explain_runner()
    return {"schema_version": 1, "skipped": False, "plan": plan}


async def measure_pgvector_search(
    db: AsyncSession,
    *,
    queries: list[list[float]],
    user_id: str,
    document_ids: list[str] | None,
    index_revision_ids: list[str] | None,
    project_id: str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Measure the production filtered query against forced exact scans.

    Use a dedicated benchmark session: planner changes are transaction-local.
    Input vectors must come from a reproducible fixture or a declared model.
    """
    if not queries or k < 10:
        raise ValueError("Provide queries and k >= 10 for Recall@10")
    repo = RagRepository(db)
    samples = []
    plans = []
    options = dict(
        user_id=user_id,
        project_id=project_id,
        document_ids=document_ids,
        index_revision_ids=index_revision_ids,
        top_k=k,
        score_threshold=-1.0,
        owner_scoped=False,
    )
    for query in queries:
        async with db.begin_nested():
            prior = {
                name: (await db.execute(text(f"SHOW {name}"))).scalar_one()
                for name in (
                    "enable_indexscan",
                    "enable_bitmapscan",
                    "enable_seqscan",
                    "enable_sort",
                )
            }
            try:
                await db.execute(text("SET LOCAL enable_seqscan = on"))
                await db.execute(text("SET LOCAL enable_sort = on"))
                await db.execute(text("SET LOCAL enable_indexscan = off"))
                await db.execute(text("SET LOCAL enable_bitmapscan = off"))
                started = perf_counter()
                exact = await repo.similarity_search_indexed(query_embedding=query, **options)
                exact_elapsed = (perf_counter() - started) * 1000
            finally:
                for name, value in prior.items():
                    await db.execute(
                        text("SELECT set_config(:name, :value, true)"),
                        {"name": name, "value": value},
                    )
            started = perf_counter()
            ann = await repo.similarity_search_indexed(query_embedding=query, **options)
            elapsed = (perf_counter() - started) * 1000
            if exact is None or ann is None:
                raise RuntimeError(
                    "Live pgvector measurement unavailable; refusing fallback metrics"
                )
            exact_ids = [chunk.chunk_id for chunk in exact]
            ann_ids = [chunk.chunk_id for chunk in ann]
            samples.append(
                {
                    "latency_ms": elapsed,
                    "exact_latency_ms": exact_elapsed,
                    "recall_at_5": recall_at_k(exact_ids, ann_ids, k=5),
                    "recall_at_10": recall_at_k(exact_ids, ann_ids, k=10),
                    "candidate_count": len(ann_ids),
                }
            )
            if len(plans) < 3:
                params = {"query_vec": vector_literal(query), "top_k": k, "score_threshold": -1.0}
                filters = repo._retrieval_scope_filters(
                    user_id=user_id,
                    project_id=project_id,
                    document_ids=document_ids,
                    owner_scoped=False,
                    params=params,
                    index_revision_ids=index_revision_ids,
                )
                where = " AND ".join(
                    [
                        *(filters if filters is not None else ["FALSE"]),
                        "c.embedding IS NOT NULL",
                        "(1 - (c.embedding <=> CAST(:query_vec AS vector))) >= :score_threshold",
                    ]
                )
                plan = await db.execute(
                    text(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                        "SELECT c.id FROM rag_chunks c "
                        "JOIN rag_documents d ON d.id=c.document_id "
                        f"WHERE {where} "
                        "ORDER BY c.embedding <=> CAST(:query_vec AS vector) "
                        "LIMIT :top_k"
                    ),
                    params,
                )
                plans.append(plan.scalar_one())
    latencies = sorted(sample["latency_ms"] for sample in samples)
    sizes = (
        (
            await db.execute(
                text(
                    "SELECT s.indexrelname, pg_relation_size(s.indexrelid) AS bytes, am.amname "
                    "FROM pg_stat_user_indexes s JOIN pg_class c ON c.oid=s.indexrelid "
                    "JOIN pg_am am ON am.oid=c.relam WHERE s.relname='rag_chunks'"
                )
            )
        )
        .mappings()
        .all()
    )
    ann_indexes = {row["indexrelname"] for row in sizes if row["amname"] in {"hnsw", "ivfflat"}}
    used_indexes = set()

    def inspect_plan(node):
        if isinstance(node, dict):
            if "Index Name" in node:
                used_indexes.add(node["Index Name"])
            for child in node.values():
                inspect_plan(child)
        elif isinstance(node, list):
            for child in node:
                inspect_plan(child)

    inspect_plan(plans)
    exact_latencies = sorted(sample["exact_latency_ms"] for sample in samples)

    def percentile(values, percent):
        return (
            values[math.ceil(len(values) * percent / 100) - 1]
            if (values and (percent != 99 or len(values) >= 100))
            else None
        )

    ann_used = bool(ann_indexes & used_indexes)
    return {
        "measurement": "MEASURED",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "timestamp": datetime.now(UTC).isoformat(),
        "database_version": (await db.execute(text("SHOW server_version"))).scalar_one(),
        "pgvector_version": (
            await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        ).scalar_one(),
        "embedding_dimensions": len(queries[0]),
        "ann_index_used": ann_used,
        "search_path": "ANN" if ann_used else "EXACT_PLANNER_CHOICE",
        "index_type": sorted(
            {row["amname"] for row in sizes if row["indexrelname"] in ann_indexes}
        ),
        "index_size_bytes": sum(
            row["bytes"] for row in sizes if row["indexrelname"] in ann_indexes
        ),
        "exact_p50_ms": percentile(exact_latencies, 50),
        "exact_p95_ms": percentile(exact_latencies, 95),
        "production_p50_ms": percentile(latencies, 50),
        "production_p95_ms": percentile(latencies, 95),
        "ann_p50_ms": percentile(latencies, 50) if ann_used else None,
        "ann_p95_ms": percentile(latencies, 95) if ann_used else None,
        "ann_p99_ms": percentile(latencies, 99) if ann_used else None,
        "recall_at_5": sum(sample["recall_at_5"] for sample in samples) / len(samples),
        "recall_at_10": sum(sample["recall_at_10"] for sample in samples) / len(samples),
        "sample_count": len(samples),
        "samples": samples,
        "plans": plans,
        "latency": {
            f"p{percent}_ms": latencies[math.ceil(len(latencies) * percent / 100) - 1]
            if latencies and (percent != 99 or len(latencies) >= 100)
            else None
            for percent in (50, 95, 99)
        },
        "index_sizes": [dict(row) for row in sizes],
        "document_ids": document_ids,
        "index_revision_ids": index_revision_ids,
    }
