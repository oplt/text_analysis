from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChunkMetadata:
    document_id: str
    user_id: str
    filename: str
    chunk_index: int
    organization_id: str | None = None
    project_id: str | None = None
    page_number: int | None = None
    source_type: str = "upload"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "source_type": self.source_type,
        }
        if self.organization_id:
            payload["organization_id"] = self.organization_id
        if self.project_id:
            payload["project_id"] = self.project_id
        if self.page_number is not None:
            payload["page_number"] = self.page_number
        return payload
