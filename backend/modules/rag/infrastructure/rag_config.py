from __future__ import annotations

from dataclasses import dataclass

from backend.core.config import settings

SUPPORTED_VECTOR_BACKENDS = frozenset({"pgvector"})
SUPPORTED_FUSION_METHODS = frozenset({"rrf"})


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
    synthesis_passages_per_document: int = 3
    parent_context_enabled: bool = False
    retrieval_algorithm_version: str = "hybrid-rrf-v1"
    chunker_version: str = "structure-v1"
    parser_version: str = "pypdf-v1"
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
            synthesis_passages_per_document=max(1, settings.RAG_SYNTHESIS_PASSAGES_PER_DOCUMENT),
            parent_context_enabled=settings.RAG_PARENT_CONTEXT_ENABLED,
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


async def validate_rag_index_health(db) -> None:
    """Ensure ANN index exists when required; always check embedding dim contract."""
    config = RagConfig.from_settings()
    if not config.enabled:
        return
    validate_rag_config(config)
    if not config.require_ann_index:
        return
    from sqlalchemy import text

    result = await db.execute(
        text(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'rag_chunks'
              AND (
                indexdef ILIKE '%hnsw%'
                OR indexdef ILIKE '%ivfflat%'
                OR indexname ILIKE '%embedding%'
              )
            """
        )
    )
    rows = result.fetchall()
    if not rows:
        raise RuntimeError(
            "RAG requires an HNSW or IVFFlat index on rag_chunks.embedding; "
            "none found. Refusing to start with RAG_REQUIRE_ANN_INDEX=true."
        )
