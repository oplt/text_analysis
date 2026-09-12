"""Deterministic evidence de-duplication and ordering before context token packing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from backend.modules.rag.domain.models import RetrievedChunk


class ContextOrderingPolicy(StrEnum):
    """Explicit context packing order after overlap de-duplication."""

    RELEVANCE = "relevance"
    GROUPED_BY_DOCUMENT = "grouped_by_document"
    CHRONOLOGY_WHEN_METADATA = "chronology_when_metadata"


@dataclass(frozen=True, slots=True)
class ContextSelection:
    chunks: list[RetrievedChunk]
    removed_chunk_ids: list[str]
    selected_chunk_ids: list[str]
    ordering_policy: ContextOrderingPolicy = ContextOrderingPolicy.RELEVANCE
    budget_removed_chunk_ids: list[str] | None = None

    def provenance(self) -> dict:
        return {
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "removed_chunk_ids": list(self.removed_chunk_ids),
            "budget_removed_chunk_ids": list(self.budget_removed_chunk_ids or []),
            "ordering_policy": self.ordering_policy.value,
        }


def select_context_chunks(
    chunks: list[RetrievedChunk],
    *,
    overlap_threshold: float = 0.8,
    ordering_policy: ContextOrderingPolicy | str = ContextOrderingPolicy.RELEVANCE,
) -> ContextSelection:
    """Keep the best anchor for duplicate content or substantially overlapping spans."""
    policy = ContextOrderingPolicy(ordering_policy)
    selected: list[RetrievedChunk] = []
    removed: list[str] = []
    for chunk in sorted(chunks, key=_priority):
        duplicate_of = next(
            (kept for kept in selected if _duplicates(chunk, kept, overlap_threshold)),
            None,
        )
        if duplicate_of is None:
            selected.append(chunk)
            continue
        removed.append(chunk.chunk_id)
        metadata = dict(duplicate_of.metadata)
        metadata["context_selection_removed_chunk_ids"] = sorted(
            set(metadata.get("context_selection_removed_chunk_ids", [])) | {chunk.chunk_id}
        )
        selected[selected.index(duplicate_of)] = replace(duplicate_of, metadata=metadata)

    ordered = order_context_chunks(selected, policy=policy)
    return ContextSelection(
        chunks=ordered,
        removed_chunk_ids=removed,
        selected_chunk_ids=[chunk.chunk_id for chunk in ordered],
        ordering_policy=policy,
    )


def order_context_chunks(
    chunks: list[RetrievedChunk],
    *,
    policy: ContextOrderingPolicy | str = ContextOrderingPolicy.RELEVANCE,
) -> list[RetrievedChunk]:
    """Apply an explicit, deterministic ordering policy."""
    resolved = ContextOrderingPolicy(policy)
    if resolved == ContextOrderingPolicy.RELEVANCE:
        return sorted(chunks, key=_priority)
    if resolved == ContextOrderingPolicy.GROUPED_BY_DOCUMENT:
        by_document: dict[str, list[RetrievedChunk]] = {}
        for chunk in sorted(chunks, key=_priority):
            by_document.setdefault(chunk.document_id, []).append(chunk)
        document_order = sorted(
            by_document,
            key=lambda document_id: _priority(by_document[document_id][0]),
        )
        return [chunk for document_id in document_order for chunk in by_document[document_id]]
    # chronology_when_metadata: use page/date when present, otherwise relevance.
    if any(_has_chronology(chunk) for chunk in chunks):
        return sorted(chunks, key=lambda chunk: (*_chronology_sort_key(chunk), *_priority(chunk)))
    return sorted(chunks, key=_priority)


def _priority(chunk: RetrievedChunk) -> tuple[float, int, str]:
    return (-chunk.score, chunk.rank if chunk.rank is not None else 2**31 - 1, chunk.chunk_id)


def _has_chronology(chunk: RetrievedChunk) -> bool:
    return _chronology_sort_key(chunk)[0] == 0


def _chronology_sort_key(chunk: RetrievedChunk) -> tuple[int, str]:
    metadata = chunk.metadata or {}
    for key in ("published_at", "document_date", "date", "timestamp"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return (0, f"{value:020.6f}")
        if isinstance(value, str) and value.strip():
            return (0, value.strip())
    if chunk.page_number is not None:
        return (0, f"{chunk.page_number:020d}")
    page = metadata.get("page_number")
    if isinstance(page, (int, float)):
        return (0, f"{int(page):020d}")
    return (1, "")


def _duplicates(left: RetrievedChunk, right: RetrievedChunk, threshold: float) -> bool:
    if left.document_id != right.document_id or _revision(left) != _revision(right):
        return False
    left_hash = left.metadata.get("content_hash")
    right_hash = right.metadata.get("content_hash")
    if left_hash and left_hash == right_hash:
        return True
    left_spans, right_spans = _span_ids(left), _span_ids(right)
    if not left_spans or not right_spans:
        return False
    return len(left_spans & right_spans) / len(left_spans | right_spans) >= threshold


def _revision(chunk: RetrievedChunk) -> str | None:
    return chunk.index_revision_id or chunk.metadata.get("revision_id")


def _span_ids(chunk: RetrievedChunk) -> set[str]:
    return set(chunk.metadata.get("source_span_ids") or ())
