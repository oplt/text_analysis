"""Deterministic document segmentation for Policy Text Lab.

Research text units must be reproducible: re-running segmentation on the same
source text must always produce the same units, in the same order, at the
same stable positions. This module contains no randomness and no I/O.
"""

from __future__ import annotations

import hashlib
import re

#: Supported research unit types (mirrors backend.modules.text_research.domain.enums.UnitType).
UNIT_TYPES = ("document", "paragraph", "sentence")

# Blank line(s) (one or more newlines with only whitespace between them) separate paragraphs.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# A sentence boundary is one-or-more terminal punctuation marks followed by
# whitespace or the end of the string. The punctuation is kept as part of the
# preceding sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)")


def hash_text(text: str) -> str:
    """Return a stable sha256 hex digest for ``text``.

    Used to verify that re-running segmentation on a document whose source
    text has not changed reproduces byte-identical text units.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into paragraphs on blank lines / double newlines.

    Empty paragraphs (blank strings after stripping) are skipped.
    """
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using a simple, deterministic regex.

    A sentence ends at a run of ``.``, ``!``, or ``?`` that is followed by
    whitespace or the end of the string. This is intentionally simple (no
    abbreviation handling) per the research-unit segmentation contract.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            sentences.append(segment)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def segment_document(text: str, unit_type: str) -> list[dict]:
    """Deterministically segment ``text`` into research units.

    Args:
        text: The full source document text.
        unit_type: One of ``"document"``, ``"paragraph"``, ``"sentence"``.

    Returns:
        A list of dicts, each with stable, zero-based ``position`` and the
        keys ``text``, ``paragraph_number``, ``sentence_number``, and
        ``page_number`` (``None`` when not applicable to the unit type).

    Rerunning this function with identical ``text``/``unit_type`` always
    returns an identical list (same order, same positions, same text).
    """
    normalized_unit_type = unit_type.strip().lower()

    if normalized_unit_type == "document":
        return [
            {
                "position": 0,
                "text": text.strip(),
                "paragraph_number": None,
                "sentence_number": None,
                "page_number": None,
            }
        ]

    if normalized_unit_type == "paragraph":
        return [
            {
                "position": position,
                "text": paragraph_text,
                "paragraph_number": position + 1,
                "sentence_number": None,
                "page_number": None,
            }
            for position, paragraph_text in enumerate(_split_paragraphs(text))
        ]

    if normalized_unit_type == "sentence":
        return [
            {
                "position": position,
                "text": sentence_text,
                "paragraph_number": None,
                "sentence_number": position + 1,
                "page_number": None,
            }
            for position, sentence_text in enumerate(_split_sentences(text))
        ]

    raise ValueError(f"Unsupported unit_type '{unit_type}'; expected one of {UNIT_TYPES}")


def segment_text(text: str, unit_type: str) -> list[dict]:
    """Alias for :func:`segment_document` used by application services."""
    return segment_document(text, unit_type)
