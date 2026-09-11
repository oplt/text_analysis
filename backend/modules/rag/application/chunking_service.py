from __future__ import annotations

import asyncio
from uuid import uuid4

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import DocumentChunk, ParsedDocument
from backend.modules.rag.infrastructure.langchain_text_splitters import split_documents
from backend.modules.rag.infrastructure.rag_config import RagConfig


def _split_documents_with_token_counts(
    documents: list[ParsedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
    document_revision: str | None = None,
) -> list[tuple[str, int, dict]]:
    return [
        (content, estimate_tokens(content), meta)
        for content, meta in split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_revision=document_revision,
        )
    ]


def _attach_parent_windows(
    leaves: list[DocumentChunk],
    *,
    window: int = 3,
) -> list[DocumentChunk]:
    """Create parent chunks spanning consecutive leaves; link children via parent_chunk_id."""
    if not leaves or window < 2:
        return leaves

    out: list[DocumentChunk] = []
    parents: list[DocumentChunk] = []
    for start in range(0, len(leaves), window):
        group = leaves[start : start + window]
        parent_id = str(uuid4())
        parent_content = _merge_non_overlapping_chunks(group)
        first = group[0]
        source_indexes = []
        source_span_ids = []
        for child in group:
            source_indexes.extend(child.metadata.get("source_paragraph_indexes") or [])
            source_span_ids.extend(child.metadata.get("source_span_ids") or [])
        parent = DocumentChunk(
            id=parent_id,
            document_id=first.document_id,
            user_id=first.user_id,
            chunk_index=-(start // window + 1),  # negative index marks parent rows
            content=parent_content,
            token_count=estimate_tokens(parent_content),
            organization_id=first.organization_id,
            project_id=first.project_id,
            content_hash=None,
            metadata={
                **{k: v for k, v in first.metadata.items() if k != "paragraph_index"},
                "chunk_role": "parent",
                "child_count": len(group),
                "source_paragraph_indexes": list(dict.fromkeys(source_indexes)),
                "source_span_ids": list(dict.fromkeys(source_span_ids)),
                "source_unit_ids": list(
                    dict.fromkeys(
                        source_unit_id
                        for child in group
                        for source_unit_id in child.metadata.get("source_unit_ids") or []
                    )
                ),
                "block_ids": list(dict.fromkeys(source_span_ids)),
                "char_start": first.metadata.get("char_start"),
                "char_end": group[-1].metadata.get("char_end"),
                "page_numbers": list(
                    dict.fromkeys(
                        child.metadata.get("page_number")
                        for child in group
                        if child.metadata.get("page_number") is not None
                    )
                ),
            },
        )
        parents.append(parent)
        for child in group:
            child.id = child.id or str(uuid4())
            child.parent_chunk_id = parent_id
            child.metadata = {
                **child.metadata,
                "chunk_role": "child",
                "parent_chunk_id": parent_id,
            }
            out.append(child)

    # Persist parents after children indexes for stable leaf ordering in UI; parents still stored
    # Parents first so FK-like references resolve when loading by id.
    return parents + out


def _merge_non_overlapping_chunks(group: list[DocumentChunk]) -> str:
    """Build a parent from canonical text, removing repeated child overlap."""
    merged = ""
    for child in group:
        content = child.content
        if not merged:
            merged = content
            continue
        max_overlap = min(len(merged), len(content))
        overlap = next(
            (
                size
                for size in range(max_overlap, 7, -1)
                if merged[-size:] == content[:size]
            ),
            0,
        )
        suffix = content[overlap:].lstrip("\n")
        if suffix:
            merged = f"{merged}\n\n{suffix}"
    return merged


class ChunkingService:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig.from_settings()

    async def chunk(
        self,
        documents: list[ParsedDocument],
        *,
        document_id: str,
        user_id: str,
        filename: str,
        project_id: str | None = None,
        organization_id: str | None = None,
        document_revision: str | None = None,
    ) -> list[DocumentChunk]:
        pieces = await asyncio.to_thread(
            _split_documents_with_token_counts,
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            document_revision=document_revision,
        )
        chunks: list[DocumentChunk] = []
        for index, (content, token_count, meta) in enumerate(pieces):
            content_hash = meta.get("content_hash")
            chunks.append(
                DocumentChunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    user_id=user_id,
                    chunk_index=index,
                    content=content,
                    token_count=token_count,
                    organization_id=organization_id,
                    project_id=project_id,
                    content_hash=content_hash if isinstance(content_hash, str) else None,
                    metadata={
                        "document_id": document_id,
                        "user_id": user_id,
                        "organization_id": organization_id,
                        "project_id": project_id,
                        "filename": filename,
                        "chunk_index": index,
                        "page_number": meta.get("page_number"),
                        "section_heading": meta.get("section_heading"),
                        "paragraph_index": meta.get("paragraph_index"),
                        "source_paragraph_indexes": meta.get("source_paragraph_indexes"),
                        "pack_index": meta.get("pack_index"),
                        "char_start": meta.get("char_start"),
                        "char_end": meta.get("char_end"),
                        "source_span_ids": meta.get("source_span_ids"),
                        "source_unit_ids": meta.get("source_unit_ids"),
                        "block_ids": meta.get("block_ids"),
                        "content_hash": content_hash,
                        "offset_coordinate_system": meta.get("offset_coordinate_system"),
                        "parser_version": getattr(self.config, "parser_version", None),
                        "chunker_version": getattr(self.config, "chunker_version", None),
                        "embedding_provider": getattr(self.config, "embedding_provider", None),
                        "embedding_model": getattr(self.config, "embedding_model", None),
                        "embedding_dimensions": getattr(
                            self.config, "embedding_dimensions", None
                        ),
                        "source_type": "upload",
                        **{
                            k: v
                            for k, v in meta.items()
                            if k
                            not in {
                                "content_hash",
                                "paragraph_index",
                                "source_paragraph_indexes",
                                "pack_index",
                                "char_start",
                                "char_end",
                                "source_span_ids",
                                "source_unit_ids",
                                "block_ids",
                                "offset_coordinate_system",
                            }
                        },
                    },
                )
            )
        if getattr(self.config, "parent_context_enabled", False):
            chunks = _attach_parent_windows(chunks, window=3)
            # Re-number leaf chunk_index ascending for retrieval display; keep parents negative.
            leaf_index = 0
            for chunk in chunks:
                if chunk.parent_chunk_id:
                    chunk.chunk_index = leaf_index
                    chunk.metadata["chunk_index"] = leaf_index
                    leaf_index += 1
        return chunks
