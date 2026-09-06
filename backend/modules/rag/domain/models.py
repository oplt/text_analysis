from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.modules.rag.domain.enums import DocumentStatus, IngestionJobStatus, SourceType


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


@dataclass(slots=True)
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    degraded: bool = False
    degradation_reason: str | None = None
    no_matches: bool = False
    injection_chunks_filtered: int = 0


@dataclass(slots=True)
class Citation:
    document_id: str
    chunk_id: str
    filename: str
    score: float
    snippet: str
    page_number: int | None = None
    chunk_index: int | None = None


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
