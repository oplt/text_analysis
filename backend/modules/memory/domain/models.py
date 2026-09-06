from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.modules.memory.domain.enums import (
    AgentScope,
    MemoryLevel,
    MemoryPrivacy,
    MemorySource,
    MemoryType,
)


@dataclass(slots=True)
class MemoryMetadata:
    memory_level: MemoryLevel
    memory_type: MemoryType
    user_id: str
    agent_id: str
    run_id: str | None = None
    project_id: str | None = None
    source: MemorySource = MemorySource.CHAT
    source_ref: str | None = None
    confidence: float = 0.0
    created_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_confirmed_at: datetime | None = None
    privacy: MemoryPrivacy = MemoryPrivacy.NORMAL
    version: int = 1
    scope: AgentScope | None = None
    event_type: str | None = None
    occurred_at: datetime | None = None
    expires_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "memory_level": self.memory_level.value,
            "memory_type": self.memory_type.value,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "source": self.source.value,
            "confidence": self.confidence,
            "privacy": self.privacy.value,
            "version": self.version,
        }
        if self.run_id:
            payload["run_id"] = self.run_id
        if self.project_id:
            payload["project_id"] = self.project_id
        if self.source_ref:
            payload["source_ref"] = self.source_ref
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.last_seen_at:
            payload["last_seen_at"] = self.last_seen_at.isoformat()
        if self.last_confirmed_at:
            payload["last_confirmed_at"] = self.last_confirmed_at.isoformat()
        if self.scope:
            payload["scope"] = self.scope.value
        if self.event_type:
            payload["event_type"] = self.event_type
        if self.occurred_at:
            payload["occurred_at"] = self.occurred_at.isoformat()
        if self.expires_at:
            payload["expires_at"] = self.expires_at.isoformat()
        if self.extra:
            payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryMetadata:
        return cls(
            memory_level=MemoryLevel(data.get("memory_level", MemoryLevel.USER.value)),
            memory_type=MemoryType(data.get("memory_type", MemoryType.FACT.value)),
            user_id=str(data.get("user_id", "")),
            agent_id=str(data.get("agent_id", "")),
            run_id=data.get("run_id"),
            project_id=data.get("project_id"),
            source=MemorySource(data.get("source", MemorySource.CHAT.value)),
            source_ref=data.get("source_ref"),
            confidence=float(data.get("confidence", 0.0)),
            privacy=MemoryPrivacy(data.get("privacy", MemoryPrivacy.NORMAL.value)),
            version=int(data.get("version", 1)),
            scope=AgentScope(data["scope"]) if data.get("scope") else None,
            event_type=data.get("event_type"),
        )


@dataclass(slots=True)
class MemoryItem:
    id: str
    content: str
    metadata: MemoryMetadata
    score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class MemoryWriteResult:
    memory_id: str
    memory_level: MemoryLevel
    memory_type: MemoryType
    accepted: bool
    rejection_reason: str | None = None


@dataclass(slots=True)
class MemorySearchRequest:
    user_id: str
    agent_id: str
    query: str
    run_id: str | None = None
    project_id: str | None = None
    memory_levels: list[MemoryLevel] | None = None
    limit: int = 10


@dataclass(slots=True)
class WorkingMemoryContext:
    """Ephemeral per-request context — never persisted."""

    user_message: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_inputs: list[str] = field(default_factory=list)
    retrieved_memories: list[MemoryItem] = field(default_factory=list)
    run_id: str | None = None
    project_id: str | None = None
