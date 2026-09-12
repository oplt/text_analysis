from __future__ import annotations

import re
from dataclasses import dataclass

from backend.core.config import settings

SUPPORTED_VECTOR_BACKENDS = frozenset({"pgvector"})
SUPPORTED_FUSION_METHODS = frozenset({"rrf"})
SUPPORTED_PDF_PARSERS = frozenset({"auto", "pymupdf", "pypdf"})


@dataclass(frozen=True, slots=True)
class RagConfig:
    enabled: bool
    vector_backend: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    max_context_tokens: int
    allowed_file_types: tuple[str, ...]
    max_file_bytes: int
    rerank_enabled: bool = False
    rerank_heuristic_enabled: bool = False
    rerank_candidate_multiplier: int = 3
    dense_candidates: int = 40
    lexical_candidates: int = 40
    fusion_method: str = "rrf"
    rrf_k: int = 60
    source_max_chunks_per_document: int = 3
    evidence_top_k: int = 12
    synthesis_max_documents: int = 25
    synthesis_batch_size: int = 8
    synthesis_async_document_threshold: int = 8
    synthesis_passages_per_document: int = 3
    parent_context_enabled: bool = False
    pdf_parser: str = "auto"
    retrieval_algorithm_version: str = "hybrid-rrf-v1"
    chunker_version: str = "structure-v1"
    parser_version: str = "pdf-auto-v1"
    index_version: str = "pgvector-fts-v1"
    expected_vector_dimensions: int = 1536
    require_ann_index: bool = False

    @classmethod
    def from_settings(cls) -> RagConfig:
        raw_types = settings.RAG_ALLOWED_FILE_TYPES.strip()
        allowed = tuple(t.strip().lower() for t in raw_types.split(",") if t.strip())
        embedding_provider = (
            settings.RAG_EMBEDDING_PROVIDER.strip() or settings.AI_EMBEDDING_PROVIDER
        ).lower()
        return cls(
            enabled=settings.RAG_ENABLED,
            vector_backend=settings.RAG_VECTOR_BACKEND.lower(),
            embedding_provider=embedding_provider,
            embedding_model=settings.RAG_EMBEDDING_MODEL,
            embedding_dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            top_k=settings.RAG_TOP_K,
            score_threshold=settings.RAG_SCORE_THRESHOLD,
            rerank_enabled=settings.RAG_RERANK_ENABLED,
            rerank_heuristic_enabled=settings.RAG_RERANK_HEURISTIC_ENABLED,
            rerank_candidate_multiplier=max(1, settings.RAG_RERANK_CANDIDATE_MULTIPLIER),
            max_context_tokens=settings.RAG_MAX_CONTEXT_TOKENS,
            allowed_file_types=allowed or ("pdf", "txt", "md", "docx", "csv"),
            max_file_bytes=settings.RAG_MAX_FILE_BYTES,
            dense_candidates=max(1, settings.RAG_DENSE_CANDIDATES),
            lexical_candidates=max(1, settings.RAG_LEXICAL_CANDIDATES),
            fusion_method=settings.RAG_FUSION_METHOD.lower().strip() or "rrf",
            rrf_k=max(1, settings.RAG_RRF_K),
            source_max_chunks_per_document=max(1, settings.RAG_SOURCE_MAX_CHUNKS_PER_DOCUMENT),
            evidence_top_k=max(1, settings.RAG_EVIDENCE_TOP_K),
            synthesis_max_documents=max(1, settings.RAG_SYNTHESIS_MAX_DOCUMENTS),
            synthesis_batch_size=max(1, settings.RAG_SYNTHESIS_BATCH_SIZE),
            synthesis_async_document_threshold=max(
                1, settings.RAG_SYNTHESIS_ASYNC_DOCUMENT_THRESHOLD
            ),
            synthesis_passages_per_document=max(1, settings.RAG_SYNTHESIS_PASSAGES_PER_DOCUMENT),
            parent_context_enabled=settings.RAG_PARENT_CONTEXT_ENABLED,
            pdf_parser=settings.RAG_PDF_PARSER.strip().lower() or "auto",
            retrieval_algorithm_version=settings.RAG_RETRIEVAL_ALGORITHM_VERSION,
            chunker_version=settings.RAG_CHUNKER_VERSION,
            parser_version=settings.RAG_PARSER_VERSION,
            index_version=settings.RAG_INDEX_VERSION,
            expected_vector_dimensions=settings.RAG_EXPECTED_VECTOR_DIMENSIONS,
            require_ann_index=settings.RAG_REQUIRE_ANN_INDEX,
        )


def validate_rag_config(config: RagConfig | None = None) -> None:
    """Fail fast when RAG is enabled with an unsupported vector backend."""
    resolved = config or RagConfig.from_settings()
    if not resolved.enabled:
        return
    if resolved.vector_backend not in SUPPORTED_VECTOR_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_VECTOR_BACKENDS))
        raise RuntimeError(
            f"RAG_VECTOR_BACKEND={resolved.vector_backend!r} is not implemented. "
            f"Supported backends: {supported}."
        )
    if resolved.fusion_method not in SUPPORTED_FUSION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_FUSION_METHODS))
        raise RuntimeError(
            f"RAG_FUSION_METHOD={resolved.fusion_method!r} is not supported. "
            f"Supported methods: {supported}."
        )
    if resolved.pdf_parser not in SUPPORTED_PDF_PARSERS:
        supported = ", ".join(sorted(SUPPORTED_PDF_PARSERS))
        raise RuntimeError(
            f"RAG_PDF_PARSER={resolved.pdf_parser!r} is not supported. "
            f"Supported parsers: {supported}."
        )
    if resolved.dense_candidates < resolved.top_k:
        raise RuntimeError("RAG_DENSE_CANDIDATES must be >= RAG_TOP_K")
    if resolved.lexical_candidates < 1:
        raise RuntimeError("RAG_LEXICAL_CANDIDATES must be >= 1")
    if resolved.embedding_dimensions != resolved.expected_vector_dimensions:
        raise RuntimeError(
            f"RAG_EMBEDDING_DIMENSIONS={resolved.embedding_dimensions} does not match "
            f"schema contract RAG_EXPECTED_VECTOR_DIMENSIONS="
            f"{resolved.expected_vector_dimensions}. "
            "Changing embedding dimensionality requires an explicit re-index migration."
        )


async def validate_rag_index_health(
    db,
    *,
    config: RagConfig | None = None,
    strict: bool | None = None,
) -> None:
    """Validate the pgvector schema when ANN health is required."""
    config = config or RagConfig.from_settings()
    if not config.enabled:
        return
    validate_rag_config(config)
    enforce_health = (
        strict if strict is not None else config.require_ann_index or settings.is_production
    )
    if not enforce_health:
        return
    from sqlalchemy import text

    schema_result = await db.execute(
        text(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                ) AS vector_extension,
                (
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                      AND c.relname = 'rag_chunks'
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                ) AS embedding_type
            """
        )
    )
    schema = schema_result.mappings().one()
    if not schema["vector_extension"]:
        raise RuntimeError(
            "RAG pgvector health check failed: the vector extension is missing. "
            "Install pgvector and run the RAG migrations."
        )

    embedding_type = str(schema["embedding_type"] or "").strip().lower()
    dimension_match = re.fullmatch(r"vector\((\d+)\)", embedding_type)
    if dimension_match is None:
        raise RuntimeError(
            "RAG pgvector health check failed: rag_chunks.embedding column is missing or "
            "is not a typed vector column."
        )
    actual_dimensions = int(dimension_match.group(1))
    if actual_dimensions != config.expected_vector_dimensions:
        raise RuntimeError(
            "RAG pgvector health check failed: rag_chunks.embedding has dimension "
            f"{actual_dimensions}, expected {config.expected_vector_dimensions}."
        )

    index_result = await db.execute(
        text(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'rag_chunks'
            """
        )
    )
    index_definitions = [str(indexdef).lower() for indexdef in index_result.scalars().all()]
    ann_embedding_indexes = [
        indexdef
        for indexdef in index_definitions
        if (" using hnsw " in indexdef or " using ivfflat " in indexdef)
        and "(embedding" in indexdef
    ]
    if not ann_embedding_indexes:
        raise RuntimeError(
            "RAG pgvector health check failed: no HNSW or IVFFlat ANN index "
            "on rag_chunks.embedding was found. Run the RAG migrations."
        )
    if not any("vector_cosine_ops" in indexdef for indexdef in ann_embedding_indexes):
        raise RuntimeError(
            "RAG pgvector health check failed: the ANN index on rag_chunks.embedding "
            "does not use the required vector_cosine_ops operator class."
        )
