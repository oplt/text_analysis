from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.config import settings
from backend.modules.rag.infrastructure.rag_config import (
    RagConfig,
    validate_rag_config,
    validate_rag_index_health,
)


def _health_db(
    *,
    extension: bool = True,
    embedding_type: str | None = "vector(1536)",
    indexes=None,
):
    schema_result = MagicMock()
    schema_result.mappings.return_value.one.return_value = {
        "vector_extension": extension,
        "embedding_type": embedding_type,
    }
    index_result = MagicMock()
    index_result.scalars.return_value.all.return_value = indexes or []
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[schema_result, index_result])
    return db


def _health_config(**changes):
    return replace(RagConfig.from_settings(), **changes)


def test_validate_rag_config_accepts_pgvector():
    validate_rag_config(
        RagConfig(
            enabled=True,
            vector_backend="pgvector",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            chunk_size=1000,
            chunk_overlap=150,
            top_k=5,
            score_threshold=0.0,
            max_context_tokens=4000,
            allowed_file_types=("pdf", "txt"),
            max_file_bytes=10_000_000,
        )
    )


def test_validate_rag_config_skips_when_rag_disabled():
    validate_rag_config(
        RagConfig(
            enabled=False,
            vector_backend="qdrant",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            chunk_size=1000,
            chunk_overlap=150,
            top_k=5,
            score_threshold=0.0,
            max_context_tokens=4000,
            allowed_file_types=("pdf", "txt"),
            max_file_bytes=10_000_000,
        )
    )


def test_validate_rag_config_rejects_unimplemented_backend():
    with pytest.raises(RuntimeError, match="qdrant"):
        validate_rag_config(
            RagConfig(
                enabled=True,
                vector_backend="qdrant",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
                chunk_size=1000,
                chunk_overlap=150,
                top_k=5,
                score_threshold=0.0,
                max_context_tokens=4000,
                allowed_file_types=("pdf", "txt"),
                max_file_bytes=10_000_000,
            )
        )


def test_validate_rag_config_rejects_unknown_pdf_parser():
    with pytest.raises(RuntimeError, match="RAG_PDF_PARSER"):
        validate_rag_config(
            RagConfig(
                enabled=True,
                vector_backend="pgvector",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
                chunk_size=1000,
                chunk_overlap=150,
                top_k=5,
                score_threshold=0.0,
                max_context_tokens=4000,
                allowed_file_types=("pdf", "txt"),
                max_file_bytes=10_000_000,
                pdf_parser="invalid",
            )
        )


def test_validate_rag_index_health_accepts_cosine_ann_index():
    db = _health_db(
        indexes=[
            "CREATE INDEX ix_rag_chunks_embedding_hnsw ON public.rag_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        ]
    )

    asyncio.run(
        validate_rag_index_health(
            db,
            config=_health_config(require_ann_index=False),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("db_kwargs", "message"),
    [
        ({"extension": False}, "vector extension"),
        ({"embedding_type": None}, "embedding column"),
        ({"embedding_type": "vector(768)"}, "dimension"),
        ({"indexes": []}, "ANN index"),
        (
            {
                "indexes": [
                    "CREATE INDEX ix_rag_chunks_embedding_hnsw ON rag_chunks "
                    "USING hnsw (embedding vector_l2_ops)"
                ]
            },
            "operator class",
        ),
    ],
)
def test_validate_rag_index_health_rejects_invalid_production_schema(db_kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            validate_rag_index_health(
                _health_db(**db_kwargs),
                config=_health_config(require_ann_index=False),
                strict=True,
            )
        )


def test_validate_rag_index_health_is_required_in_production(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "production")
    db = _health_db(indexes=[])

    with pytest.raises(RuntimeError, match="ANN index"):
        asyncio.run(
            validate_rag_index_health(
                db,
                config=_health_config(require_ann_index=False),
            )
        )
