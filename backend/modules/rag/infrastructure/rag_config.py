from __future__ import annotations

import re
from dataclasses import dataclass

from backend.core.config import settings
from backend.modules.rag.infrastructure.config_models import (
    ChunkingConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    ParsingConfig,
    RerankingConfig,
    RetrievalConfig,
    SynthesisConfig,
)

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
    embedding_model_version: str | None = None
    embedding_preprocessing_version: str = "unicode-nfc-v1"
    rerank_enabled: bool = False
    rerank_heuristic_enabled: bool = False
    rerank_candidate_multiplier: int = 3
    rerank_max_depth: int = 40
    context_overlap_dedupe_threshold: float = 0.8
    context_ordering_policy: str = "relevance"
    pdf_ocr_enabled: bool = False
    pdf_ocr_min_text_chars: int = 40
    pdf_text_quality_threshold: float = 0.65
    pdf_table_extraction_enabled: bool = False
    pdf_header_footer_suppression: bool = True
    text_decode_max_replacement_ratio: float = 0.02
    unicode_normalization: str = "NFC"
    rerank_intent_depth_overrides: dict[str, int] | None = None
    dense_candidates: int = 40
    lexical_candidates: int = 40
    fusion_method: str = "rrf"
    rrf_k: int = 60
    source_max_chunks_per_document: int = 3
    evidence_top_k: int = 12
    synthesis_max_documents: int = 25
    synthesis_reduce_token_budget: int = 6000
    synthesis_batch_size: int = 8
    synthesis_async_document_threshold: int = 8
    synthesis_passages_per_document: int = 3
    parent_context_enabled: bool = False
    retrieval_branch_concurrency: int = 2
    query_variant_concurrency: int = 3
    pdf_parser: str = "auto"
    retrieval_algorithm_version: str = "hybrid-rrf-v1"
    chunker_version: str = "structure-v1"
    chunk_policy_version: str = "structure-v1"
    parent_window_size: int = 3
    parser_version: str = "pdf-auto-v1"
    index_version: str = "pgvector-fts-v1"
    expected_vector_dimensions: int = 1536
    require_ann_index: bool = False
    exact_vector_fallback_max_rows: int = 5000
    retrieval_cache_artifact_version: str = "v2"

    @property
    def parsing(self) -> ParsingConfig:
        return ParsingConfig(
            self.parser_version,
            self.pdf_parser,
            unicode_normalization=self.unicode_normalization,
            text_decode_max_replacement_ratio=self.text_decode_max_replacement_ratio,
            pdf_ocr_enabled=self.pdf_ocr_enabled,
            pdf_ocr_min_text_chars=self.pdf_ocr_min_text_chars,
            pdf_text_quality_threshold=self.pdf_text_quality_threshold,
            pdf_table_extraction_enabled=self.pdf_table_extraction_enabled,
            pdf_header_footer_suppression=self.pdf_header_footer_suppression,
        )

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            self.chunker_version,
            self.chunk_policy_version,
            self.chunk_size,
            self.chunk_overlap,
            self.parent_window_size,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            self.embedding_provider,
            self.embedding_model,
            self.embedding_model_version,
            self.embedding_dimensions,
            self.embedding_preprocessing_version,
        )

    @property
    def retrieval(self) -> RetrievalConfig:
        return RetrievalConfig(
            self.retrieval_algorithm_version,
            self.index_version,
            self.fusion_method,
            self.rrf_k,
            self.dense_candidates,
            self.lexical_candidates,
            self.top_k,
            self.score_threshold,
            self.parent_context_enabled,
            self.retrieval_branch_concurrency,
            self.query_variant_concurrency,
        )

    @property
    def reranking(self) -> RerankingConfig:
        return RerankingConfig(
            self.rerank_enabled,
            self.rerank_heuristic_enabled,
            self.rerank_max_depth,
            intent_depth_overrides=self.rerank_intent_depth_overrides,
        )

    @property
    def generation(self) -> GenerationConfig:
        return GenerationConfig(
            self.max_context_tokens,
            self.context_overlap_dedupe_threshold,
            context_ordering_policy=self.context_ordering_policy,
        )

    @property
    def synthesis(self) -> SynthesisConfig:
        return SynthesisConfig(
            self.synthesis_passages_per_document,
            self.synthesis_reduce_token_budget,
            self.synthesis_batch_size,
        )

    @property
    def evaluation(self) -> EvaluationConfig:
        return EvaluationConfig()

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
            embedding_model_version=settings.RAG_EMBEDDING_MODEL_VERSION or None,
            embedding_dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
            embedding_preprocessing_version=settings.RAG_EMBEDDING_PREPROCESSING_VERSION,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            top_k=settings.RAG_TOP_K,
            score_threshold=settings.RAG_SCORE_THRESHOLD,
            rerank_enabled=settings.RAG_RERANK_ENABLED,
            rerank_heuristic_enabled=settings.RAG_RERANK_HEURISTIC_ENABLED,
            rerank_candidate_multiplier=max(1, settings.RAG_RERANK_CANDIDATE_MULTIPLIER),
            rerank_max_depth=max(0, settings.RAG_RERANK_MAX_DEPTH),
            context_overlap_dedupe_threshold=min(
                1.0, max(0.0, settings.RAG_CONTEXT_OVERLAP_DEDUPE_THRESHOLD)
            ),
            context_ordering_policy=(
                getattr(settings, "RAG_CONTEXT_ORDERING_POLICY", "relevance") or "relevance"
            )
            .strip()
            .lower(),
            pdf_ocr_enabled=bool(settings.RAG_PDF_OCR_ENABLED),
            pdf_ocr_min_text_chars=max(1, settings.RAG_PDF_OCR_MIN_TEXT_CHARS),
            pdf_text_quality_threshold=float(settings.RAG_PDF_TEXT_QUALITY_THRESHOLD),
            pdf_table_extraction_enabled=bool(settings.RAG_PDF_TABLE_EXTRACTION_ENABLED),
            pdf_header_footer_suppression=bool(
                getattr(settings, "RAG_PDF_HEADER_FOOTER_SUPPRESSION", True)
            ),
            text_decode_max_replacement_ratio=float(settings.RAG_TEXT_DECODE_MAX_REPLACEMENT_RATIO),
            unicode_normalization="NFC",
            rerank_intent_depth_overrides={
                "fact": 0,
                "definition": 0,
                "evidence": min(20, max(0, settings.RAG_RERANK_MAX_DEPTH)),
                "semantic_search": min(40, max(0, settings.RAG_RERANK_MAX_DEPTH)),
                "comparison": min(40, max(0, settings.RAG_RERANK_MAX_DEPTH)),
                "synthesis": min(40, max(0, settings.RAG_RERANK_MAX_DEPTH)),
                "contradiction": min(40, max(0, settings.RAG_RERANK_MAX_DEPTH)),
            },
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
            synthesis_reduce_token_budget=max(512, settings.RAG_SYNTHESIS_REDUCE_TOKEN_BUDGET),
            synthesis_batch_size=max(1, settings.RAG_SYNTHESIS_BATCH_SIZE),
            synthesis_async_document_threshold=max(
                1, settings.RAG_SYNTHESIS_ASYNC_DOCUMENT_THRESHOLD
            ),
            synthesis_passages_per_document=max(1, settings.RAG_SYNTHESIS_PASSAGES_PER_DOCUMENT),
            parent_context_enabled=settings.RAG_PARENT_CONTEXT_ENABLED,
            retrieval_branch_concurrency=max(1, settings.RAG_RETRIEVAL_BRANCH_CONCURRENCY),
            query_variant_concurrency=max(1, settings.RAG_QUERY_VARIANT_CONCURRENCY),
            pdf_parser=settings.RAG_PDF_PARSER.strip().lower() or "auto",
            retrieval_algorithm_version=settings.RAG_RETRIEVAL_ALGORITHM_VERSION,
            chunker_version=settings.RAG_CHUNKER_VERSION,
            chunk_policy_version=settings.RAG_CHUNK_POLICY_VERSION,
            parent_window_size=max(2, settings.RAG_PARENT_WINDOW_SIZE),
            parser_version=settings.RAG_PARSER_VERSION,
            index_version=settings.RAG_INDEX_VERSION,
            expected_vector_dimensions=settings.RAG_EXPECTED_VECTOR_DIMENSIONS,
            require_ann_index=settings.RAG_REQUIRE_ANN_INDEX,
            exact_vector_fallback_max_rows=max(1, settings.RAG_EXACT_VECTOR_FALLBACK_MAX_ROWS),
            retrieval_cache_artifact_version=settings.RAG_RETRIEVAL_CACHE_ARTIFACT_VERSION,
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
