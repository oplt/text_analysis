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
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagRetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    filename: str
    chunk_index: int
    page_number: int | None = None


class RagRetrieveResponse(BaseModel):
    chunks: list[RagRetrievedChunkResponse]
    degraded: bool = False
    degradation_reason: str | None = None
    no_matches: bool = False
    injection_chunks_filtered: int = 0


class RagAskRequest(RequestModel):
    query: str = Field(min_length=1, max_length=4000)
    project_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class RagCitationResponse(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    score: float
    snippet: str
    page_number: int | None = None
    chunk_index: int | None = None


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


class RagQueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    query: str
    answer: str
    project_id: str | None
    model_name: str
    latency_ms: int
    created_at: datetime
