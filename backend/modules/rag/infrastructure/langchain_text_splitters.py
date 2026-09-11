"""Structure-aware chunking: section/paragraph boundaries before token constraints."""

from __future__ import annotations

import hashlib
import re

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6}\s+\S|.{1,80}\n[=-]{3,}\s*$)", re.MULTILINE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack_units(
    units: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Greedy pack paragraph/section units into token-bounded chunks with overlap."""
    if not units:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append("\n\n".join(current))
        if chunk_overlap > 0 and current:
            # Keep trailing units for overlap while under overlap budget
            overlap_units: list[str] = []
            overlap_tokens = 0
            for unit in reversed(current):
                t = estimate_tokens(unit)
                if overlap_units and overlap_tokens + t > chunk_overlap:
                    break
                overlap_units.insert(0, unit)
                overlap_tokens += t
            current = overlap_units
            current_tokens = overlap_tokens
        else:
            current = []
            current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if unit_tokens > chunk_size:
            flush()
            # Fall back to recursive char split for oversized units
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=estimate_tokens,
            )
            chunks.extend(splitter.split_text(unit))
            continue
        if current and current_tokens + unit_tokens > chunk_size:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
    flush()
    return chunks


def split_documents(
    documents: list[ParsedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict]]:
    """Structure-aware split: preserve page/section metadata; pack by paragraphs.

    CPU-bound: call only via ``asyncio.to_thread`` (see ``ChunkingService.chunk``).
    """
    results: list[tuple[str, dict]] = []
    for doc in documents:
        base_meta = {**doc.metadata}
        if doc.page_number is not None:
            base_meta.setdefault("page_number", doc.page_number)
        section = base_meta.get("section_heading")
        paragraphs = _split_paragraphs(doc.content)
        pieces = _pack_units(paragraphs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for paragraph_index, piece in enumerate(pieces):
            meta = {
                **base_meta,
                "paragraph_index": paragraph_index,
                "content_hash": _content_hash(piece),
                "chunker_strategy": "structure-v1",
            }
            if section:
                meta["section_heading"] = section
            results.append((piece, meta))
    return results
