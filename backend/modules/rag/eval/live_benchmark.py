"""Live production-table load benchmark. Requires an empty, migrated disposable DB."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.modules.identity_access.models import User
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.eval.load_benchmark import DEFAULT_SCALES
from backend.modules.rag.eval.pgvector_benchmark import measure_pgvector_search
from backend.modules.rag.infrastructure.models import RagChunk  # noqa: F401
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository


def seeded_vector(seed: int) -> list[float]:
    """Declared synthetic vectors: 32 independent dimensions, padded to production size."""
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(32)] + [0.0] * 1504


async def run(scale_index: int, output: Path, samples: int = 20) -> dict:
    url = os.environ["RAG_INTEGRATION_DATABASE_URL"]
    engine = create_async_engine(url)
    scale = DEFAULT_SCALES[scale_index]
    report = {"measurement": "NOT_RUN"}
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            if (await db.execute(text("SELECT count(*) FROM rag_documents"))).scalar_one():
                raise ValueError("Benchmark requires an empty disposable RAG database")
            user = User(email=f"benchmark-{uuid4()}@example.invalid", password_hash="disabled")
            db.add(user)
            await db.flush()
            repo = RagRepository(db)
            documents, revisions = [], []
            for index in range(scale.documents):
                doc = await repo.create_document(
                    user_id=user.id,
                    filename=f"{index}.txt",
                    original_filename=f"{index}.txt",
                    content_type="text/plain",
                    storage_path=None,
                    project_id=None,
                    organization_id=None,
                    source_type="benchmark",
                    metadata={},
                )
                revision = str(uuid4())
                await db.execute(
                    text(
                        "INSERT INTO rag_document_revisions (id, document_id, created_at) "
                        "VALUES (:revision, :document, now())"
                    ),
                    {"revision": revision, "document": doc.id},
                )
                doc.current_revision_id = revision
                doc.status = "indexed"
                documents.append(doc.id)
                revisions.append(revision)
            await db.flush()
            started = perf_counter()
            for start in range(0, scale.chunks, 256):
                rows = []
                for index in range(start, min(start + 256, scale.chunks)):
                    document = index % scale.documents
                    rows.append(
                        {
                            "id": str(uuid4()),
                            "document": documents[document],
                            "user": user.id,
                            "revision": revisions[document],
                            "index": index // scale.documents,
                            "vector": str(seeded_vector(index)),
                            "content": f"research evidence {index}",
                        }
                    )
                await db.execute(
                    text(
                        "INSERT INTO rag_chunks "
                        "(id, document_id, user_id, revision_id, chunk_index, "
                        "content, token_count, embedding, created_at, updated_at) VALUES "
                        "(:id, :document, :user, :revision, :index, :content, 4, "
                        "CAST(:vector AS vector), now(), now())"
                    ),
                    rows,
                )
            await db.commit()
            seed_seconds = perf_counter() - started
            # Measures a real rebuild of only this disposable benchmark's vector index.
            indexes = (
                (
                    await db.execute(
                        text(
                            "SELECT c.relname FROM pg_index i "
                            "JOIN pg_class c ON c.oid=i.indexrelid "
                            "JOIN pg_am a ON a.oid=c.relam WHERE i.indrelid='rag_chunks'::regclass "
                            "AND a.amname IN ('hnsw', 'ivfflat')"
                        )
                    )
                )
                .scalars()
                .all()
            )
            started = perf_counter()
            for index in indexes:
                quoted = db.bind.dialect.identifier_preparer.quote(index)
                await db.execute(text(f"REINDEX INDEX {quoted}"))
            await db.commit()
            index_seconds = perf_counter() - started
            await db.execute(text("ANALYZE rag_chunks"))
            await db.execute(text("ANALYZE rag_documents"))
            queries = [seeded_vector(scale.chunks + index) for index in range(samples)]
            report = await measure_pgvector_search(
                db,
                queries=queries,
                user_id=user.id,
                document_ids=None,
                index_revision_ids=None,
            )
            filtered = await measure_pgvector_search(
                db,
                queries=queries,
                user_id=user.id,
                document_ids=documents[: max(1, scale.documents // 10)],
                index_revision_ids=revisions[: max(1, scale.documents // 10)],
            )
            hybrid_latencies = []
            retrieval = RetrievalService(db, replace(RagConfig.from_settings(), enabled=True))

            async def embed_texts(texts):
                return [
                    seeded_vector(int.from_bytes(hashlib.sha256(value.encode()).digest()[:4]))
                    for value in texts
                ]

            retrieval.embeddings = SimpleNamespace(embed_texts=embed_texts)
            for index in range(samples):
                started = perf_counter()
                outcome = await retrieval.retrieve(
                    f"research evidence {scale.chunks + index}",
                    user_id=user.id,
                    project_id=None,
                    top_k=10,
                    filters={"document_ids": documents, "index_revision_ids": revisions},
                )
                if outcome.degraded:
                    raise AssertionError("Hybrid measurement degraded; refusing latency report")
                hybrid_latencies.append((perf_counter() - started) * 1000)
            # Keep the default planner measurement intact. This separate diagnostic
            # measures real ANN, without claiming that PostgreSQL selected it by default.
            async with db.begin_nested() as transaction:
                await db.execute(text("SET LOCAL enable_seqscan=off"))
                await db.execute(text("SET LOCAL enable_sort=off"))
                forced = await measure_pgvector_search(
                    db,
                    queries=queries,
                    user_id=user.id,
                    document_ids=None,
                    index_revision_ids=revisions,
                )
                forced["filtered"] = await measure_pgvector_search(
                    db,
                    queries=queries,
                    user_id=user.id,
                    document_ids=documents[: max(1, scale.documents // 10)],
                    index_revision_ids=revisions[: max(1, scale.documents // 10)],
                )
                forced["filtered_recall_at_10"] = forced["filtered"]["recall_at_10"]
                forced["planner_overrides"] = {"enable_seqscan": False, "enable_sort": False}
                await transaction.rollback()
            report.update(
                {
                    "embedding_measurement": "SYNTHETIC_32D_PADDED_1536",
                    "documents": scale.documents,
                    "chunks": scale.chunks,
                    "seed_seconds": seed_seconds,
                    "index_build_seconds": index_seconds,
                    "filtered": filtered,
                    "filtered_recall_at_10": filtered["recall_at_10"],
                    "hybrid_p95_ms": sorted(hybrid_latencies)[math.ceil(samples * 0.95) - 1],
                    "hybrid_status": "MEASURED_SYNTHETIC_EMBEDDINGS",
                    "ann_diagnostic": forced,
                    "production_plan_warning": None if report["ann_index_used"] else "exact_scan",
                }
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2))
            if not forced["ann_index_used"] or forced["recall_at_10"] < 0.5:
                raise AssertionError(f"ANN diagnostic failed; inspect {output}")
            if report["recall_at_10"] < 0.5 or filtered["recall_at_10"] < 0.5:
                raise AssertionError(f"Recall below conservative 0.5 threshold; inspect {output}")
            return report
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--output", type=Path, default=Path("var/benchmarks/rag-live.json"))
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(run(("small", "medium", "large").index(args.scale), args.output, args.samples))
