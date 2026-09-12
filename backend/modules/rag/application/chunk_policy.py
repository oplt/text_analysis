"""Small, inspectable document-type policies for structure-v1 chunking."""

from __future__ import annotations

from dataclasses import dataclass

from backend.modules.rag.domain.models import ParsedDocument


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    version: str
    target_tokens: int
    overlap_tokens: int
    profile: str = "prose"
    respect_page_boundaries: bool = False
    respect_section_boundaries: bool = True
    atomic_tables: bool = False


def resolve_chunk_policy(document: ParsedDocument, config) -> ChunkPolicy:
    """Keep prose defaults while applying PDF/page/table-aware profiles."""
    metadata = document.metadata or {}
    version = getattr(config, "chunk_policy_version", "structure-v1")
    base_size = int(getattr(config, "chunk_size", 1000))
    base_overlap = int(getattr(config, "chunk_overlap", 150))
    fmt = str(metadata.get("format") or "").lower()
    block_type = str(metadata.get("parsed_block_type") or "").lower()

    if block_type == "table" or fmt == "csv":
        return ChunkPolicy(
            version=version,
            target_tokens=base_size,
            overlap_tokens=0,
            profile="atomic_table",
            respect_page_boundaries=fmt == "pdf",
            respect_section_boundaries=True,
            atomic_tables=True,
        )
    if fmt == "pdf":
        # Page-aware: slightly smaller windows reduce cross-page parent bleed.
        return ChunkPolicy(
            version=version,
            target_tokens=min(base_size, 768),
            overlap_tokens=min(base_overlap, 96),
            profile="pdf_page",
            respect_page_boundaries=True,
            respect_section_boundaries=True,
            atomic_tables=False,
        )
    if fmt in {"markdown", "docx"}:
        return ChunkPolicy(
            version=version,
            target_tokens=base_size,
            overlap_tokens=base_overlap,
            profile="section_aware",
            respect_page_boundaries=False,
            respect_section_boundaries=True,
            atomic_tables=False,
        )
    return ChunkPolicy(
        version=version,
        target_tokens=base_size,
        overlap_tokens=base_overlap,
        profile="prose",
        respect_page_boundaries=False,
        respect_section_boundaries=True,
        atomic_tables=False,
    )
