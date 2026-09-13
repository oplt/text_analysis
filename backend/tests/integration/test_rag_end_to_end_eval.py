"""Real ingestion/pgvector pipeline with explicitly synthetic offline embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.modules.identity_access.models import User
from backend.modules.rag.application import rag_answer_service, trace_context
from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.rag_answer_service import RagAnswerService
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.eval.end_to_end_eval import load_end_to_end_qrels, run_end_to_end_benchmark
from backend.modules.rag.eval.fixture_ingestion import ingest_fixture_corpus
from backend.modules.rag.eval.multilingual_benchmark import benchmark_turkish_lexical
from backend.modules.rag.eval.pgvector_benchmark import measure_pgvector_search
from backend.modules.rag.infrastructure.rag_config import RagConfig


async def synthetic_embeddings(self, texts):
    """Stable token hashing tests pipeline wiring, not learned semantic quality."""
    vectors = []
    for content in texts:
        vector = [0.0] * self.config.embedding_dimensions
        for token in re.findall(r"\w+", content.casefold()):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % len(vector)
            vector[index] += 1.0
        vectors.append(vector)
    return vectors


@pytest.mark.asyncio
async def test_real_fixture_ingestion_and_retrieval(monkeypatch, tmp_path):
    url = os.getenv("RAG_INTEGRATION_DATABASE_URL")
    if not url:
        pytest.skip("Set RAG_INTEGRATION_DATABASE_URL to a migrated disposable pgvector database")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("rag-evaluation")
    monkeypatch.setattr(trace_context, "tracer", tracer)
    monkeypatch.setattr(rag_answer_service, "tracer", tracer)
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: tracer)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    config = replace(
        RagConfig.from_settings(),
        enabled=True,
        embedding_dimensions=1536,
        rerank_heuristic_enabled=True,
    )
    monkeypatch.setattr(EmbeddingService, "embed_texts", synthetic_embeddings)
    try:
        async with factory() as db:
            assert (
                await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
            ).scalar()
            user = User(email=f"rag-eval-{uuid4()}@example.invalid", password_hash="disabled")
            db.add(user)
            await db.commit()
            index = await ingest_fixture_corpus(
                DocumentIngestionService(db, config), user_id=user.id
            )
            assert len(index.document_ids) == 3
            assert len(index.chunk_ids) >= 3
            answers = RagAnswerService(db, config)

            async def answer_runner(case, outcome):
                # The offline generator sees only actual selected evidence, never qrels.
                async def generate(*args, **kwargs):
                    ids = kwargs["retrieved_chunk_ids"]
                    claims = [
                        {"text": chunk.content, "chunk_ids": [chunk.chunk_id]}
                        for chunk in outcome.chunks
                        if chunk.chunk_id in ids
                    ]
                    return SimpleNamespace(
                        id=None,
                        model_name="synthetic-extractive",
                        output_text=json.dumps({"claims": claims, "no_evidence": not claims}),
                    )

                answers.generation = SimpleNamespace(run_rag_answer=generate)
                return await answers.answer_from_retrieval(
                    case.query,
                    outcome=outcome,
                    user=user,
                    project_id=None,
                    document_ids=list(index.document_ids.values()),
                    include_memory=False,
                )

            with tracer.start_as_current_span("evaluation.request"):
                report = await run_end_to_end_benchmark(
                    RetrievalService(db, config),
                    user_id=user.id,
                    project_id=None,
                    document_ids=list(index.document_ids.values()),
                    cases=index.bind(load_end_to_end_qrels()),
                    answer_runner=answer_runner,
                    index_revision_ids=list(index.revision_ids),
                )
            report["measurement"] = "MEASURED_DATABASE_SYNTHETIC_EMBEDDINGS"
            spans = exporter.get_finished_spans()
            assert len({span.context.trace_id for span in spans}) == 1
            assert {
                "rag.retrieval",
                "rag.generation",
                "rag.context_selection",
                "rag.citation_validation",
                "rag.answer.persistence",
            } <= {s.name for s in spans}
            assert all(not ({"query", "content", "api_key"} & set(s.attributes)) for s in spans)
            report["trace_stages"] = sorted({span.name for span in spans})
            report["fixture_mapping"] = index.source_coordinates
            report["pgvector"] = await measure_pgvector_search(
                db,
                queries=await synthetic_embeddings(
                    EmbeddingService(config),
                    ["institutional accountability", "participation", "hesap verebilirlik"],
                ),
                user_id=user.id,
                document_ids=list(index.document_ids.values()),
                index_revision_ids=list(index.revision_ids),
            )
            report["turkish_lexical"] = await benchmark_turkish_lexical(db)
            output = Path(os.getenv("RAG_EVAL_REPORT_PATH", str(tmp_path / "rag-end-to-end.json")))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            cases = {case["id"]: case for case in report["cases"]}
            assert cases["keyword"]["recall_at_5"] >= 0.5, str(output)
            assert cases["multilingual"]["recall_at_5"] >= 0.5, str(output)
            assert all(case["context_token_count"] <= 4000 for case in report["cases"])
            assert all(case["citation_structural_validity"] == 1.0 for case in report["cases"])
            assert all(case["citation_validation_status"] == "valid" for case in report["cases"])
            assert all(value["source_spans"] for value in index.source_coordinates.values())
    finally:
        await engine.dispose()
        provider.shutdown()
