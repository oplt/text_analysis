from __future__ import annotations

from backend.modules.rag.domain.models import RetrievedChunk


def exclude_injection_flagged_chunks(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], int]:
    """Drop chunks flagged at ingestion time for suspected prompt injection."""
    kept: list[RetrievedChunk] = []
    removed = 0
    for chunk in chunks:
        if chunk.metadata.get("prompt_injection_suspected"):
            removed += 1
            continue
        kept.append(chunk)
    return kept, removed
