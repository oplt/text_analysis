"""Sanitized per-request RAG timing and branch diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4


@dataclass(slots=True)
class RagTraceContext:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    stage_timings_ms: dict[str, int] = field(default_factory=dict)
    branch_status: dict[str, str] = field(default_factory=dict)

    def measure(self, stage: str) -> float:
        return perf_counter()

    def complete(self, stage: str, started: float, *, status: str = "ok") -> None:
        self.stage_timings_ms[stage] = int((perf_counter() - started) * 1000)
        self.branch_status[stage] = status
