from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.core.schemas import RequestModel
from pydantic import BaseModel, ConfigDict, Field


class RagDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    project_id: str | None
    organization_id: str | None
    filename: str
    original_filename: str
    content_type: str
    storage_path: str | None
    status: str
    source_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    current_revision_id: str | None = None


class RagDocumentUploadResponse(BaseModel):
    document: RagDocumentResponse
    ingestion_job: RagIngestionJobResponse


class RagChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision_id: str | None = None


class RagIngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    user_id: str
    project_id: str | None
    status: str
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class RagRetrieveRequest(RequestModel):
    query: str = Field(min_length=1, max_length=4000)
    project_id: str | None = None
    # None = unscoped (generic /rag); [] = empty allow-list (I1); [...] = filter
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    intent: str | None = None


class RagRetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    filename: str
    chunk_index: int
    page_number: int | None = None
    rank: int | None = None
    used_in_answer: bool = False
    index_revision_id: str | None = None
    source_span_ids: list[str] | None = None
    char_start: int | None = None
    char_end: int | None = None
    offset_coordinate_system: str | None = None
    offset_scope: str | None = None
    offset_scope_id: str | None = None
    source_spans: list[dict] | None = None


class RagCoverageResponse(BaseModel):
    documents_in_scope: int = 0
    documents_with_retrieved_evidence: int = 0
    retrieved_passage_count: int = 0
    coverage_ratio: float = 0.0


class RagRetrieveResponse(BaseModel):
    chunks: list[RagRetrievedChunkResponse]
    degraded: bool = False
    degradation_reason: str | None = None
    no_matches: bool = False
    injection_chunks_filtered: int = 0
    intent: str | None = None
    fusion_method: str | None = None
    coverage: RagCoverageResponse | None = None
    retrieval_trace_id: str | None = None
    index_revision_ids: list[str] = Field(default_factory=list)


class RagAskRequest(RequestModel):
    query: str = Field(min_length=1, max_length=4000)
    project_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    document_ids: list[str] | None = None
    intent: str | None = None


class RagCitationResponse(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    score: float
    snippet: str
    page_number: int | None = None
    chunk_index: int | None = None
    citation_number: int | None = None
    used_in_answer: bool = False
    section_heading: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_span_ids: list[str] | None = None
    parent_context_id: str | None = None
    offset_coordinate_system: str | None = None
    offset_scope: str | None = None
    offset_scope_id: str | None = None
    source_spans: list[dict] | None = None


class RagAskResponse(BaseModel):
    query: str
    answer: str
    citations: list[RagCitationResponse]
    retrieved_chunk_ids: list[str]
    model_name: str
    latency_ms: int
    no_context_found: bool
    ai_run_id: str | None = None
    retrieval_degraded: bool = False
    memory_degraded: bool = False
    degradation_reason: str | None = None
    injection_chunks_filtered: int = 0
    retrieval_trace_id: str | None = None
    evidence_revision_hash: str | None = None
    citation_validation_failed: bool = False
    coverage: RagCoverageResponse | None = None
    index_revision_ids: list[str] = Field(default_factory=list)


class RagQueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    query: str
    answer: str
    project_id: str | None
    model_name: str
    latency_ms: int
    created_at: datetime
