from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.core.schemas import RequestModel
from pydantic import BaseModel, ConfigDict, Field


class MemoryMetadataResponse(BaseModel):
    memory_level: str
    memory_type: str
    user_id: str
    agent_id: str
    run_id: str | None = None
    project_id: str | None = None
    source: str
    source_ref: str | None = None
    confidence: float = 0.0
    privacy: str = "normal"
    version: int = 1
    created_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_confirmed_at: datetime | None = None


class MemoryItemResponse(BaseModel):
    id: str
    content: str
    metadata: MemoryMetadataResponse
    score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryWriteRequest(RequestModel):
    content: str = Field(min_length=3, max_length=4000)
    memory_level: str
    memory_type: str | None = None
    agent_id: str = "default"
    run_id: str | None = None
    project_id: str | None = None
    source: str = "manual"
    source_ref: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    privacy: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteResponse(BaseModel):
    memory_id: str
    memory_level: str
    memory_type: str
    accepted: bool
    rejection_reason: str | None = None


class MemorySearchRequestBody(RequestModel):
    query: str = Field(min_length=1, max_length=2000)
    agent_id: str = "default"
    run_id: str | None = None
    project_id: str | None = None
    memory_levels: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)


class MemoryForgetRequest(RequestModel):
    reason: str = Field(min_length=3, max_length=500)


class MemoryAuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str | None
    agent_id: str | None
    run_id: str | None
    project_id: str | None
    operation: str
    external_memory_id: str | None
    memory_level: str | None
    memory_type: str | None
    source_ref: str | None
    created_at: datetime
