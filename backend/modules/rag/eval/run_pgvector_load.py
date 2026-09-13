"""Seed declared synthetic vectors and measure production pgvector search at scale."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.modules.identity_access.models import User
from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.rag.eval.load_benchmark import DEFAULT_SCALES, LoadScale
from backend.modules.rag.eval.pgvector_benchmark import measure_pgvector_search
from backend.modules.rag.infrastructure.repositories import RagRepository


def seed_vector(index: int) -> list[float]:
    # Reproducible, nonzero, distinct vectors; no learned-model quality claim.
    return [math.sin((index + 1) * (axis + 1) * 1.61803398875) for axis in range(32)] + [0.0] * 1504


async def run(scale: LoadScale, report_path: Path, *, samples: int = 100) -> dict:
    engine = create_async_engine(os.environ["RAG_INTEGRATION_DATABASE_URL"])
    started = perf_counter()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            user = User(email=f"pgvector-load-{uuid4()}@example.invalid", password_hash="disabled")
            db.add(user)
            await db.commit()
            repo = RagRepository(db)
            documents, revisions = [], []
            chunk_index = 0
            for document_index in range(scale.documents):
                document = await repo.create_document(
                    user_id=user.id,
                    filename=f"load-{document_index}.txt",
                    original_filename=f"load-{document_index}.txt",
                    content_type="text/plain",
                    storage_path=None,
                    project_id=None,
                    organization_id=None,
                    source_type="benchmark",
                    metadata={"measurement": "synthetic_vectors"},
                )
                revision = await repo.create_document_revision(
                    document,
                    source_content_hash=None,
                    parser_version="benchmark-seed-v1",
                    chunker_version="benchmark-seed-v1",
                    index_version="pgvector-fts-v1",
                    embedding_provider="synthetic",
                    embedding_model="sine-32-padded-1536",
                    embedding_model_version="v1",
                    embedding_dimensions=1536,
                    embedding_preprocessing_version="none",
                )
                rows = []
                for local_index in range(scale.chunks // scale.documents):
                    rows.append(
                        {
                            "id": str(uuid4()),
                            "chunk_index": local_index,
                            "content": f"Research load fixture {chunk_index}",
                            "token_count": 8,
                            "embedding": seed_vector(chunk_index),
                            "metadata": {},
                        }
                    )
                    chunk_index += 1
                await repo.replace_chunks(
                    document, rows, revision_id=revision.id, expected_embedding_dimensions=1536
                )
                await repo.activate_document_revision(document, revision)
                await repo.update_document_status(document, DocumentStatus.INDEXED)
                await db.commit()
                documents.append(document.id)
                revisions.append(revision.id)
            seed_seconds = perf_counter() - started
            await db.execute(text("ANALYZE rag_chunks"))
            await db.execute(text("ANALYZE rag_documents"))
            queries = [
                seed_vector(index * max(1, scale.chunks // samples)) for index in range(samples)
            ]
            report = {
                "measurement": "MEASURED",
                "vectors": "SYNTHETIC",
                "documents": len(documents),
                "chunks": chunk_index,
                "seed_seconds": seed_seconds,
                "chunks_per_second": chunk_index / seed_seconds,
                "postgres_version": (await db.execute(text("SELECT version()"))).scalar_one(),
            }
            for name, selected in (
                ("full", documents),
                ("filtered", documents[:10]),
                ("empty", []),
            ):
                report[name] = await measure_pgvector_search(
                    db,
                    queries=queries if name == "full" else queries[:10],
                    user_id=user.id,
                    document_ids=selected,
                    index_revision_ids=revisions,
                )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in ("measurement", "chunks", "seed_seconds", "chunks_per_second")
                    }
                )
            )
            return report
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("10k", "100k", "1M"), default="10k")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(
        run(
            dict(zip(("10k", "100k", "1M"), DEFAULT_SCALES, strict=True))[args.scale],
            args.report,
            samples=args.samples,
        )
    )
