"""Deterministic document segmentation for text research.

Research text units must be reproducible: re-running segmentation on the same
source text must always produce the same units, in the same order, at the
same stable positions and character offsets. This module contains no randomness
and no I/O.

Sentence segmentation is language-aware (via ``language_processing``) and skips
common false boundaries: abbreviations, initials, decimals, URLs, acronyms,
and numbered lists.

Page numbers and section headings are attached only when page provenance is
supplied (from a canonical extract). They are never invented.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from backend.modules.text_research.infrastructure.language_processing import (
    split_sentences,
)

#: Supported research unit types (mirrors backend.modules.text_research.domain.enums.UnitType).
UNIT_TYPES = ("document", "paragraph", "sentence")

# Blank line(s) (one or more newlines with only whitespace between them) separate paragraphs.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def hash_text(text: str) -> str:
    """Return a stable sha256 hex digest for ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_span(raw: str, absolute_start: int) -> tuple[int, int, str] | None:
    """Locate stripped content within ``raw`` and return absolute [start, end)."""
    stripped = raw.strip()
    if not stripped:
        return None
    rel = raw.find(stripped)
    if rel < 0:
        # Fallback should not happen for strip(); keep deterministic.
        return absolute_start, absolute_start + len(raw), raw
    start = absolute_start + rel
    end = start + len(stripped)
    return start, end, stripped


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    last = 0
    for match in _PARAGRAPH_SPLIT_RE.finditer(text):
        span = _content_span(text[last : match.start()], last)
        if span is not None:
            spans.append(span)
        last = match.end()
    span = _content_span(text[last:], last)
    if span is not None:
        spans.append(span)
    return spans


def _sentence_spans(text: str, language: str | None = None) -> list[tuple[int, int, str]]:
    """Language-aware sentence spans; unknown languages use generic Unicode heuristics."""
    return split_sentences(text, language)


def _unit_dict(
    *,
    position: int,
    text: str,
    char_start: int,
    char_end: int,
    paragraph_number: int | None,
    sentence_number: int | None,
    page_number: int | None = None,
    section_heading: str | None = None,
    source_text_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "position": position,
        "text": text,
        "text_hash": hash_text(text),
        "char_start": char_start,
        "char_end": char_end,
        "paragraph_number": paragraph_number,
        "sentence_number": sentence_number,
        "page_number": page_number,
        "section_heading": section_heading,
        "source_text_hash": source_text_hash,
    }


def lookup_page_provenance(
    char_start: int,
    page_provenance: list[dict[str, Any]] | None,
) -> tuple[int | None, str | None]:
    """Return (page_number, section_heading) for a unit start offset.

    Does not invent values: missing provenance → (None, None).
    """
    if not page_provenance:
        return None, None
    for page in page_provenance:
        start = page.get("char_start")
        end = page.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start <= char_start < end or (char_start == start == end):
            page_number = page.get("page_number")
            heading = page.get("section_heading")
            return (
                page_number if isinstance(page_number, int) else None,
                heading if isinstance(heading, str) and heading else None,
            )
    return None, None


def apply_page_provenance(
    units: list[dict[str, Any]],
    page_provenance: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Attach page/section from canonical provenance without inventing values."""
    if not page_provenance:
        return units
    enriched: list[dict[str, Any]] = []
    for unit in units:
        page_number, section_heading = lookup_page_provenance(unit["char_start"], page_provenance)
        enriched.append(
            {
                **unit,
                "page_number": page_number,
                "section_heading": section_heading,
            }
        )
    return enriched


def segment_document(
    text: str,
    unit_type: str,
    *,
    page_provenance: list[dict[str, Any]] | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministically segment ``text`` into research units with offsets.

    Returns dicts with: position, text, text_hash, char_start, char_end,
    paragraph_number, sentence_number, page_number, section_heading,
    source_text_hash (hash of the full input document text).

    ``language`` selects sentence-boundary heuristics when available; unknown
    languages degrade to generic Unicode segmentation without claiming
    language-specific accuracy.
    """
    normalized_unit_type = unit_type.strip().lower()
    source_text_hash = hash_text(text)

    if normalized_unit_type == "document":
        span = _content_span(text, 0)
        if span is None:
            units: list[dict[str, Any]] = []
        else:
            start, end, content = span
            units = [
                _unit_dict(
                    position=0,
                    text=content,
                    char_start=start,
                    char_end=end,
                    paragraph_number=None,
                    sentence_number=None,
                    source_text_hash=source_text_hash,
                )
            ]
        return apply_page_provenance(units, page_provenance)

    if normalized_unit_type == "paragraph":
        units = [
            _unit_dict(
                position=position,
                text=content,
                char_start=start,
                char_end=end,
                paragraph_number=position + 1,
                sentence_number=None,
                source_text_hash=source_text_hash,
            )
            for position, (start, end, content) in enumerate(_paragraph_spans(text))
        ]
        return apply_page_provenance(units, page_provenance)

    if normalized_unit_type == "sentence":
        units = [
            _unit_dict(
                position=position,
                text=content,
                char_start=start,
                char_end=end,
                paragraph_number=None,
                sentence_number=position + 1,
                source_text_hash=source_text_hash,
            )
            for position, (start, end, content) in enumerate(
                _sentence_spans(text, language=language)
            )
        ]
        return apply_page_provenance(units, page_provenance)

    raise ValueError(f"Unsupported unit_type '{unit_type}'; expected one of {UNIT_TYPES}")


def segment_text(
    text: str,
    unit_type: str,
    *,
    page_provenance: list[dict[str, Any]] | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Alias for :func:`segment_document` used by application services."""
    return segment_document(text, unit_type, page_provenance=page_provenance, language=language)
