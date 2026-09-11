from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.modules.rag.domain.enums import DocumentStatus, IngestionJobStatus, RetrievalIntent, SourceType


@dataclass(slots=True)
class ParsedDocument:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None


@dataclass(slots=True)
class DocumentChunk:
    document_id: str
    user_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    organization_id: str | None = None
    project_id: str | None = None
    embedding: list[float] | None = None
    id: str | None = None
    vector_external_id: str | None = None
    parent_chunk_id: str | None = None
    content_hash: str | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    filename: str
    chunk_index: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    used_in_answer: bool = False
    retrieval_sources: tuple[str, ...] = ()


@dataclass(slots=True)
class RetrievalCoverage:
    documents_in_scope: int = 0
    documents_with_retrieved_evidence: int = 0
    retrieved_passage_count: int = 0
    coverage_ratio: float = 0.0


@dataclass(slots=True)
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    degraded: bool = False
    degradation_reason: str | None = None
    no_matches: bool = False
    injection_chunks_filtered: int = 0
    intent: RetrievalIntent | None = None
    fusion_method: str | None = None
    coverage: RetrievalCoverage | None = None
    dense_candidate_count: int = 0
    lexical_candidate_count: int = 0
    fused_candidate_count: int = 0
    retrieval_trace_id: str | None = None
    scope_hash: str | None = None


@dataclass(slots=True)
class ClaimCitation:
    text: str
    chunk_ids: list[str]
    citation_numbers: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    document_id: str
    chunk_id: str
    filename: str
    score: float
    snippet: str
    page_number: int | None = None
    chunk_index: int | None = None
    citation_number: int | None = None
    section_heading: str | None = None
    used_in_answer: bool = False
    char_start: int | None = None
    char_end: int | None = None
    source_span_ids: list[str] | None = None
    corpus_document_id: str | None = None


@dataclass(slots=True)
class RagAnswer:
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    model_name: str
    latency_ms: int
    no_context_found: bool = False
    ai_run_id: str | None = None
    retrieval_degraded: bool = False
    memory_degraded: bool = False
    degradation_reason: str | None = None
    injection_chunks_filtered: int = 0
    claims: list[ClaimCitation] = field(default_factory=list)
    retrieval_trace_id: str | None = None
    coverage: RetrievalCoverage | None = None
    citation_validation_failed: bool = False
    citation_validation_status: str = "valid"
    prompt_template_id: str | None = None
    prompt_version_id: str | None = None
    resolved_retrieval_query: str | None = None
    context_message_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RagQuery:
    id: str
    user_id: str
    query: str
    answer: str
    project_id: str | None
    retrieved_chunk_ids: list[str]
    model_name: str
    latency_ms: int
    created_at: datetime


@dataclass(slots=True)
class Document:
    id: str
    user_id: str
    filename: str
    original_filename: str
    content_type: str
    storage_path: str | None
    status: DocumentStatus
    source_type: SourceType
    project_id: str | None
    organization_id: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class IngestionJob:
    id: str
    document_id: str
    user_id: str
    project_id: str | None
    status: IngestionJobStatus
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
