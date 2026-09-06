from __future__ import annotations

import asyncio

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import DocumentChunk, ParsedDocument
from backend.modules.rag.infrastructure.langchain_text_splitters import split_documents
from backend.modules.rag.infrastructure.rag_config import RagConfig


def _split_documents_with_token_counts(
    documents: list[ParsedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, dict]]:
    return [
        (content, estimate_tokens(content), meta)
        for content, meta in split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


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
    ) -> list[DocumentChunk]:
        pieces = await asyncio.to_thread(
            _split_documents_with_token_counts,
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks: list[DocumentChunk] = []
        for index, (content, token_count, meta) in enumerate(pieces):
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    user_id=user_id,
                    chunk_index=index,
                    content=content,
                    token_count=token_count,
                    organization_id=organization_id,
                    project_id=project_id,
                    metadata={
                        "document_id": document_id,
                        "user_id": user_id,
                        "organization_id": organization_id,
                        "project_id": project_id,
                        "filename": filename,
                        "chunk_index": index,
                        "page_number": meta.get("page_number"),
                        "source_type": "upload",
                        **meta,
                    },
                )
            )
        return chunks
