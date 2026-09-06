from __future__ import annotations

import logging

from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.application.retrieval_service import RetrievalService
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def rag_search_tool(
    db: AsyncSession,
    *,
    query: str,
    user_id: str,
    project_id: str | None = None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """
    Agent-callable RAG retrieval tool.

    Returns retrieved chunks and a formatted context block for prompt injection.
    """
    config = RagConfig.from_settings()
    if not config.enabled:
        return [], ""

    retrieval = RetrievalService(db, config)
    filters = {"document_ids": document_ids} if document_ids else None
    try:
        outcome = await retrieval.retrieve(
            query,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k,
            filters=filters,
        )
        chunks = outcome.chunks
    except Exception:
        logger.exception("rag_search_tool failed for user=%s", user_id)
        return [], ""

    builder = RagContextBuilder()
    bounded = builder.trim_chunks_to_token_budget(
        chunks,
        max_tokens=config.max_context_tokens,
    )
    context = builder.build_document_context_block(bounded)
    return bounded, context
