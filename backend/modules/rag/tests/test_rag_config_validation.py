from __future__ import annotations

import pytest

from backend.modules.rag.infrastructure.rag_config import RagConfig, validate_rag_config


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
