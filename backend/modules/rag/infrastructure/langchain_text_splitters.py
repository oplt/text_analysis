from __future__ import annotations

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.domain.models import ParsedDocument


def split_documents(
    documents: list[ParsedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict]]:
    """Split parsed documents using LangChain RecursiveCharacterTextSplitter.

    CPU-bound: call only via ``asyncio.to_thread`` (see ``ChunkingService.chunk``).

    ``chunk_size`` / ``chunk_overlap`` are interpreted as token estimates (len/4),
    aligned with ``RagContextBuilder.trim_chunks_to_token_budget``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=estimate_tokens,
    )
    results: list[tuple[str, dict]] = []
    for doc in documents:
        pieces = splitter.split_text(doc.content)
        for piece in pieces:
            metadata = {**doc.metadata}
            if doc.page_number is not None:
                metadata.setdefault("page_number", doc.page_number)
            results.append((piece, metadata))
    return results
