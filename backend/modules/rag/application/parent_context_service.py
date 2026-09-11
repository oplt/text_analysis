"""Expand child chunks to parent context when parent_chunk_id is set (I5 allow-list)."""

from __future__ import annotations

from backend.modules.rag.application.source_diversifier import filter_to_allow_list
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.repositories import RagRepository


async def expand_parent_chunks(
    chunks: list[RetrievedChunk],
    *,
    repo: RagRepository,
    document_ids: list[str] | None,
) -> list[RetrievedChunk]:
    """Replace child passage text with parent chunk text when parent is in allow-list.

    Parents outside ``document_ids`` are ignored (I5). Empty allow-list yields no expansion.
    """
    if not chunks:
        return chunks

    child_rows = await repo.get_chunks_by_ids([c.chunk_id for c in chunks])
    parent_ids = [
        row.parent_chunk_id for row in child_rows if getattr(row, "parent_chunk_id", None)
    ]
    if not parent_ids:
        return chunks

    parents = await repo.get_chunks_by_ids(list({pid for pid in parent_ids if pid}))
    parent_by_id = {row.id: row for row in parents}

    # Build temporary RetrievedChunks for allow-list filtering
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
    allowed_parents = {
        c.chunk_id for c in filter_to_allow_list(parent_as_retrieved, document_ids)
    }

    child_to_parent = {
        row.id: row.parent_chunk_id
        for row in child_rows
        if row.parent_chunk_id and row.parent_chunk_id in allowed_parents
    }
    if not child_to_parent:
        return chunks

    expanded: list[RetrievedChunk] = []
    for chunk in chunks:
        parent_id = child_to_parent.get(chunk.chunk_id)
        if not parent_id:
            expanded.append(chunk)
            continue
        parent = parent_by_id.get(parent_id)
        if parent is None:
            expanded.append(chunk)
            continue
        meta = dict(chunk.metadata)
        meta["parent_chunk_id"] = parent_id
        meta["parent_expanded"] = True
        expanded.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=parent.content,
                score=chunk.score,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                metadata=meta,
                rank=chunk.rank,
                used_in_answer=chunk.used_in_answer,
                retrieval_sources=tuple(
                    dict.fromkeys([*chunk.retrieval_sources, "parent_expand"])
                ),
            )
        )
    return expanded
