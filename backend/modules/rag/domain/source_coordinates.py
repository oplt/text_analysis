"""Explicit, parser-local source coordinates carried by RAG evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceCoordinate:
    scope_type: str
    scope_id: str
    coordinate_system: str
    start: int
    end: int
    source_span_id: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)
