"""Canonical research source text — never built from overlapping RAG chunks.

Retrieval chunks may overlap for embedding search. Joining them duplicates
spans and corrupts frequencies, DFMs, classifiers, and hashes.

Research analysis must start from an immutable canonical extract built from
parser output (pages/sections) or an explicitly provided full text, then
persisted with checksums and provenance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

PARSER_NAME = "rag.DocumentParserService"
CANONICAL_JOIN = "\n\n"

# Minimum shared suffix/prefix length to treat adjacent chunks as overlapped.
_MIN_OVERLAP_CHARS = 32


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class CanonicalBuildResult:
    text: str
    text_checksum: str
    page_provenance: list[dict[str, Any]] = field(default_factory=list)
    transformation_metadata: dict[str, Any] = field(default_factory=dict)
    parser_name: str = PARSER_NAME
    parser_version: str = "unknown"
    language: str | None = None
    original_file_checksum: str | None = None
    source_file_reference: str | None = None
    extracted_at: datetime = field(default_factory=_utcnow)


def resolve_parser_version() -> str:
    versions: list[str] = []
    for package in ("pypdf", "python-docx"):
        try:
            from importlib.metadata import version

            versions.append(f"{package}={version(package)}")
        except Exception:  # noqa: BLE001
            continue
    return ";".join(versions) if versions else "unknown"


def build_canonical_from_pages(
    pages: list[dict[str, Any]],
    *,
    parser_name: str = PARSER_NAME,
    parser_version: str | None = None,
    language: str | None = None,
    original_file_checksum: str | None = None,
    source_file_reference: str | None = None,
    extra_transformation: dict[str, Any] | None = None,
) -> CanonicalBuildResult:
    """Stitch parser pages/sections into one canonical document.

    Pages are joined with blank lines. Character offsets are recorded for
    provenance. No sliding-window overlap is introduced.
    """
    parts: list[str] = []
    provenance: list[dict[str, Any]] = []
    cursor = 0
    join_len = len(CANONICAL_JOIN)

    for index, page in enumerate(pages):
        content = (page.get("content") or "").strip()
        if not content:
            continue
        if parts:
            cursor += join_len
        start = cursor
        end = start + len(content)
        provenance.append(
            {
                "index": index,
                "page_number": page.get("page_number"),
                "section_heading": page.get("section_heading"),
                "char_start": start,
                "char_end": end,
                "source_span_ids": list(page.get("source_span_ids") or []),
            }
        )
        parts.append(content)
        cursor = end

    text = CANONICAL_JOIN.join(parts)
    metadata = {
        "join": "double_newline",
        "page_count": len(parts),
        "builder": "build_canonical_from_pages",
        **(extra_transformation or {}),
    }
    return CanonicalBuildResult(
        text=text,
        text_checksum=sha256_text(text),
        page_provenance=provenance,
        transformation_metadata=metadata,
        parser_name=parser_name,
        parser_version=parser_version or resolve_parser_version(),
        language=language,
        original_file_checksum=original_file_checksum,
        source_file_reference=source_file_reference,
    )


def build_canonical_from_full_text(
    text: str,
    *,
    parser_name: str = "explicit.full_text",
    parser_version: str = "n/a",
    language: str | None = None,
    original_file_checksum: str | None = None,
    source_file_reference: str | None = None,
    extra_transformation: dict[str, Any] | None = None,
) -> CanonicalBuildResult:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return CanonicalBuildResult(
        text=normalized,
        text_checksum=sha256_text(normalized),
        page_provenance=(
            [
                {
                    "index": 0,
                    "page_number": None,
                    "section_heading": None,
                    "char_start": 0,
                    "char_end": len(normalized),
                    "source_span_ids": [],
                }
            ]
            if normalized
            else []
        ),
        transformation_metadata={
            "builder": "build_canonical_from_full_text",
            **(extra_transformation or {}),
        },
        parser_name=parser_name,
        parser_version=parser_version,
        language=language,
        original_file_checksum=original_file_checksum,
        source_file_reference=source_file_reference,
    )


def adjacent_chunk_overlap_chars(
    left: str, right: str, *, min_chars: int = _MIN_OVERLAP_CHARS
) -> int:
    """Return shared suffix/prefix length between adjacent chunk texts, else 0."""
    if not left or not right:
        return 0
    max_n = min(len(left), len(right))
    for n in range(max_n, min_chars - 1, -1):
        if left[-n:] == right[:n]:
            return n
    return 0


def chunks_have_retrieval_overlap(chunk_contents: list[str]) -> bool:
    """True when any adjacent pair shares a long overlapping span (RAG-style)."""
    for index in range(len(chunk_contents) - 1):
        if adjacent_chunk_overlap_chars(chunk_contents[index], chunk_contents[index + 1]) > 0:
            return True
    return False


def naive_join_chunks(chunk_contents: list[str], separator: str = "\n") -> str:
    """Unsafe reconstruction used only in tests to demonstrate duplication."""
    return separator.join(chunk_contents)


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_compare(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())
