"""Structure-aware chunking: section/paragraph boundaries before token constraints."""

from __future__ import annotations

import hashlib
import re
from uuid import uuid4

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import ParsedDocument

_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+\S.+|(?:.|\n){1,80}\n[=-]{3,}\s*)$",
    re.MULTILINE,
)
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_SETEXT_HEADING = re.compile(r"^(.+)\n([=-]){3,}\s*$", re.MULTILINE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _extract_units_with_headings(text: str) -> list[tuple[str, str | None]]:
    """Split into paragraph units, tracking nearest preceding heading via _HEADING_RE."""
    if not text.strip():
        return []
    headings = list(_HEADING_RE.finditer(text))
    paragraphs = _split_paragraphs(text)
    if not headings:
        return [(p, None) for p in paragraphs]

    # Map char offsets of paragraphs in original text for heading association.
    units: list[tuple[str, str | None]] = []
    cursor = 0
    for para in paragraphs:
        idx = text.find(para, cursor)
        if idx < 0:
            units.append((para, None))
            continue
        current_heading: str | None = None
        for match in headings:
            if match.start() <= idx:
                raw = match.group(0).strip()
                atx = _ATX_HEADING.match(raw.split("\n", 1)[0])
                if atx:
                    current_heading = atx.group(2).strip()
                else:
                    setext = _SETEXT_HEADING.match(raw)
                    current_heading = (
                        setext.group(1).strip() if setext else raw.split("\n", 1)[0].strip()
                    )
            else:
                break
        units.append((para, current_heading))
        cursor = idx + len(para)
    return units


def _pack_units(
    units: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, list[int]]]:
    """Greedy pack paragraph units; return (chunk_text, source_unit_indexes)."""
    if not units:
        return []
    packed: list[tuple[str, list[int]]] = []
    current: list[str] = []
    current_idxs: list[int] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_idxs, current_tokens
        if not current:
            return
        packed.append(("\n\n".join(current), list(current_idxs)))
        if chunk_overlap > 0 and current:
            overlap_units: list[str] = []
            overlap_idxs: list[int] = []
            overlap_tokens = 0
            for unit, idx in zip(reversed(current), reversed(current_idxs), strict=True):
                t = estimate_tokens(unit)
                if overlap_units and overlap_tokens + t > chunk_overlap:
                    break
                overlap_units.insert(0, unit)
                overlap_idxs.insert(0, idx)
                overlap_tokens += t
            current = overlap_units
            current_idxs = overlap_idxs
            current_tokens = overlap_tokens
        else:
            current = []
            current_idxs = []
            current_tokens = 0

    for unit_index, unit in enumerate(units):
        unit_tokens = estimate_tokens(unit)
        if unit_tokens > chunk_size:
            flush()
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=estimate_tokens,
            )
            for piece in splitter.split_text(unit):
                packed.append((piece, [unit_index]))
            continue
        if current and current_tokens + unit_tokens > chunk_size:
            flush()
        current.append(unit)
        current_idxs.append(unit_index)
        current_tokens += unit_tokens
    flush()
    return packed


def split_documents(
    documents: list[ParsedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict]]:
    """Structure-aware split: preserve page/section metadata and source spans.

    CPU-bound: call only via ``asyncio.to_thread`` (see ``ChunkingService.chunk``).
    """
    results: list[tuple[str, dict]] = []
    for doc in documents:
        base_meta = {**doc.metadata}
        if doc.page_number is not None:
            base_meta.setdefault("page_number", doc.page_number)
        full_text = doc.content or ""
        unit_pairs = _extract_units_with_headings(full_text)
        units = [u for u, _ in unit_pairs]
        headings = [h for _, h in unit_pairs]
        # Absolute char offsets of each paragraph unit in the source document.
        unit_spans: list[tuple[int, int]] = []
        cursor = 0
        for unit in units:
            start = full_text.find(unit, cursor)
            if start < 0:
                start = cursor
            end = start + len(unit)
            unit_spans.append((start, end))
            cursor = end

        packed = _pack_units(units, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for pack_index, (piece, source_idxs) in enumerate(packed):
            section = base_meta.get("section_heading")
            for idx in source_idxs:
                if idx < len(headings) and headings[idx]:
                    section = headings[idx]
                    break
            span_ids = [str(uuid4()) for _ in source_idxs]
            char_start = unit_spans[source_idxs[0]][0] if source_idxs else None
            char_end = unit_spans[source_idxs[-1]][1] if source_idxs else None
            # Packed chunks reference constituent source paragraphs — do not call
            # the pack counter ``paragraph_index``.
            meta = {
                **base_meta,
                "source_paragraph_indexes": list(source_idxs),
                "pack_index": pack_index,
                "content_hash": _content_hash(piece),
                "chunker_strategy": "structure-v1",
                "char_start": char_start,
                "char_end": char_end,
                "source_span_ids": span_ids,
                "block_ids": span_ids,
            }
            if section:
                meta["section_heading"] = section
            # Legacy key only when a packed chunk is a single source paragraph.
            if len(source_idxs) == 1:
                meta["paragraph_index"] = source_idxs[0]
            results.append((piece, meta))
    return results
