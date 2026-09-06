from __future__ import annotations

import logging

from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VectorStoreAdapter:
    async def similarity_search(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        top_k: int,
        filters: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]: ...

    async def delete_document(self, document_id: str, user_id: str) -> None: ...


class PgVectorAdapter:
    """
    Postgres pgvector-backed vector store.

    Uses HNSW indexed cosine search when pgvector is available and embedding
    dimensions match RAG_EMBEDDING_DIMENSIONS. Falls back to a bounded JSON scan
    for legacy rows that were never indexed into pgvector.
    """

    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.repo = RagRepository(db)
        self.config = config or RagConfig.from_settings()

    async def similarity_search(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None,
        top_k: int,
        filters: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        if query_embedding is None:
            return []

        document_ids = (filters or {}).get("document_ids")
        indexed = await self.repo.similarity_search_indexed(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=self.config.score_threshold,
        )
        if indexed is None:
            from backend.modules.rag.infrastructure import metrics

            logger.debug(
                "Using JSON embedding fallback for query "
                "(pgvector unavailable or dimension mismatch)"
            )
            metrics.rag_json_fallback_total.inc()
            return await self.repo.similarity_search_json_fallback(
                user_id=user_id,
                project_id=project_id,
                document_ids=document_ids,
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=self.config.score_threshold,
            )

        if indexed:
            return indexed

        relaxed_threshold = max(0.05, self.config.score_threshold * 0.5)
        if relaxed_threshold < self.config.score_threshold:
            relaxed = await self.repo.similarity_search_indexed(
                user_id=user_id,
                project_id=project_id,
                document_ids=document_ids,
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=relaxed_threshold,
            )
            if relaxed:
                logger.debug(
                    "Indexed search recovered %s chunk(s) with relaxed threshold %.2f",
                    len(relaxed),
                    relaxed_threshold,
                )
                return relaxed

        from backend.modules.rag.infrastructure import metrics

        logger.debug("Indexed search returned no matches; trying JSON embedding fallback")
        metrics.rag_json_fallback_total.inc()
        return await self.repo.similarity_search_json_fallback(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=self.config.score_threshold,
        )

    async def delete_document(self, document_id: str, user_id: str) -> None:
        del user_id
        await self.repo.delete_chunks_for_document(document_id)


def build_vector_store(db: AsyncSession, config: RagConfig | None = None) -> PgVectorAdapter:
    return PgVectorAdapter(db, config)
