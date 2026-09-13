"""Run production ingestion, retrieval and optional generation in a benchmark database."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.modules.identity_access.models import User
from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.eval.end_to_end_eval import load_end_to_end_qrels, run_end_to_end_benchmark
from backend.modules.rag.eval.fixture_ingestion import ingest_fixture_corpus
from backend.modules.rag.infrastructure.rag_config import RagConfig


async def run(report_path: Path, *, generate: bool) -> dict:
    url = os.environ["RAG_INTEGRATION_DATABASE_URL"]
    engine = create_async_engine(url)
    config = replace(RagConfig.from_settings(), enabled=True)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            user = User(email=f"rag-benchmark-{uuid4()}@example.invalid", password_hash="disabled")
            db.add(user)
            await db.commit()
            index = await ingest_fixture_corpus(
                DocumentIngestionService(db, config), user_id=user.id
            )
            documents = list(index.document_ids.values())
            answers = RagAnswerService(db, config)

            async def answer(case, outcome):
                return await answers.answer_from_retrieval(
                    case.query,
                    outcome=outcome,
                    user=user,
                    project_id=None,
                    document_ids=documents,
                    include_memory=False,
                )

            report = await run_end_to_end_benchmark(
                RetrievalService(db, config),
                user_id=user.id,
                project_id=None,
                document_ids=documents,
                index_revision_ids=list(index.revision_ids),
                cases=index.bind(load_end_to_end_qrels()),
                answer_runner=answer if generate else None,
                max_context_tokens=config.max_context_tokens,
            )
            report.update(
                measurement="MEASURED",
                embedding_provider=config.embedding_provider,
                embedding_model=config.embedding_model,
                fixture_mapping=index.source_coordinates,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            return report
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--generate", action="store_true", help="Use the configured generation provider"
    )
    args = parser.parse_args()
    asyncio.run(run(args.report, generate=args.generate))
