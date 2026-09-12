"""Expand child chunks to parent context when parent_chunk_id is set (I5 allow-list)."""

from __future__ import annotations

from dataclasses import replace

from backend.modules.rag.application.source_diversifier import filter_to_allow_list
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.repositories import RagRepository


async def expand_parent_chunks(
    chunks: list[RetrievedChunk],
    *,
    repo: RagRepository,
    document_ids: list[str] | None,
) -> list[RetrievedChunk]:
    """Expand child hits with parent text; dedupe when several children share a parent.

    Citations keep the child chunk_id that caused retrieval. Parents outside
    ``document_ids`` are ignored (I5).
    """
    if not chunks:
        return chunks

    child_rows = await repo.get_chunks_by_ids([c.chunk_id for c in chunks])
    child_by_id = {row.id: row for row in child_rows}
    parent_ids = [
        row.parent_chunk_id
        for chunk in chunks
        if (row := child_by_id.get(chunk.chunk_id)) is not None
        and getattr(row, "document_id", None) == chunk.document_id
        and getattr(row, "parent_chunk_id", None)
    ]
    if not parent_ids:
        return chunks

    parents = await repo.get_chunks_by_ids(list({pid for pid in parent_ids if pid}))
    parent_by_id = {row.id: row for row in parents}

    parent_as_retrieved = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            content=row.content,
            score=0.0,
            filename="",
            chunk_index=row.chunk_index,
        )
        for row in parents
    ]
    allowed_parents = {c.chunk_id for c in filter_to_allow_list(parent_as_retrieved, document_ids)}

    child_to_parent = {
        chunk.chunk_id: row.parent_chunk_id
        for chunk in chunks
        if (row := child_by_id.get(chunk.chunk_id)) is not None
        and getattr(row, "document_id", None) == chunk.document_id
        and row.parent_chunk_id in allowed_parents
        and getattr(parent_by_id.get(row.parent_chunk_id), "document_id", None) == chunk.document_id
    }
    if not child_to_parent:
        return chunks

    expanded: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    for chunk in chunks:
        parent_id = child_to_parent.get(chunk.chunk_id)
        if not parent_id:
            expanded.append(chunk)
            continue
        citation_content = chunk.citation_content or chunk.content
        provenance = {
            "citation_chunk_id": chunk.citation_chunk_id or chunk.chunk_id,
            "parent_context_id": parent_id,
        }
        if parent_id in seen_parents:
            # Keep child citation provenance without duplicating parent context.
            meta = dict(chunk.metadata)
            meta.update(provenance)
            meta["parent_expanded"] = True
            meta["parent_deduped"] = True
            expanded.append(
                replace(
                    chunk,
                    metadata=meta,
                    citation_content=citation_content,
                    citation_chunk_id=provenance["citation_chunk_id"],
                    parent_context_id=parent_id,
                )
            )
            continue
        parent = parent_by_id.get(parent_id)
        if parent is None:
            expanded.append(chunk)
            continue
        seen_parents.add(parent_id)
        meta = dict(chunk.metadata)
        meta.update(provenance)
        meta["parent_expanded"] = True
        meta["retrieved_child_chunk_id"] = chunk.chunk_id
        expanded.append(
            replace(
                chunk,
                metadata=meta,
                citation_content=citation_content,
                citation_chunk_id=provenance["citation_chunk_id"],
                context_content=parent.content,
                parent_context_id=parent_id,
                retrieval_sources=tuple(dict.fromkeys([*chunk.retrieval_sources, "parent_expand"])),
            )
        )
    return expanded
