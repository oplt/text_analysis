from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from backend.core.pagination import DEFAULT_PAGE_LIMIT, paginate_scalars
from backend.lib.vector_search import (
    embedding_is_indexable,
    json_fallback_max_candidates,
    parse_embedding_json,
    rank_embedding_matches,
    store_chunk_embeddings_batch,
)
from backend.lib.vector_search import (
    pgvector_is_available as check_pgvector_is_available,
)
from backend.lib.vectors import vector_literal
from backend.modules.rag.domain.enums import DocumentStatus, IngestionJobStatus
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.rag.infrastructure.models import (
    RagChunk,
    RagDocument,
    RagIngestionJob,
    RagQueryRecord,
)
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class RagRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_document(
        self,
        *,
        user_id: str,
        filename: str,
        original_filename: str,
        content_type: str,
        storage_path: str | None,
        project_id: str | None,
        organization_id: str | None,
        source_type: str,
        metadata: dict | None,
    ) -> RagDocument:
        row = RagDocument(
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            content_type=content_type,
            storage_path=storage_path,
            project_id=project_id,
            organization_id=organization_id,
            source_type=source_type,
            status=DocumentStatus.UPLOADED.value,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=True),
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_document(self, document_id: str) -> RagDocument | None:
        result = await self.db.execute(
            select(RagDocument).where(
                RagDocument.id == document_id,
                RagDocument.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def list_documents_for_user(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagDocument], int]:
        stmt = select(RagDocument).where(
            RagDocument.user_id == user_id,
            RagDocument.deleted_at.is_(None),
        )
        if project_id:
            stmt = stmt.where(RagDocument.project_id == project_id)
        stmt = stmt.order_by(RagDocument.created_at.desc())
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def list_document_ids_for_user(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
    ) -> list[str]:
        stmt = select(RagDocument.id).where(
            RagDocument.user_id == user_id,
            RagDocument.deleted_at.is_(None),
        )
        if project_id:
            stmt = stmt.where(RagDocument.project_id == project_id)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def filter_document_ids_for_user(
        self,
        user_id: str,
        document_ids: list[str],
        *,
        project_id: str | None = None,
    ) -> list[str]:
        if not document_ids:
            return []
        stmt = select(RagDocument.id).where(
            RagDocument.user_id == user_id,
            RagDocument.deleted_at.is_(None),
            RagDocument.id.in_(document_ids),
        )
        if project_id:
            stmt = stmt.where(RagDocument.project_id == project_id)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def list_indexed_documents(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RagDocument]:
        stmt = select(RagDocument).where(
            RagDocument.user_id == user_id,
            RagDocument.status == DocumentStatus.INDEXED.value,
            RagDocument.deleted_at.is_(None),
        )
        if project_id:
            stmt = stmt.where(RagDocument.project_id == project_id)
        if document_ids:
            stmt = stmt.where(RagDocument.id.in_(document_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_document_status(
        self, document: RagDocument, status: DocumentStatus
    ) -> RagDocument:
        document.status = status.value
        document.updated_at = datetime.now(UTC)
        await self.db.flush()
        return document

    async def soft_delete_document(self, document: RagDocument) -> RagDocument:
        document.status = DocumentStatus.DELETED.value
        document.deleted_at = datetime.now(UTC)
        document.updated_at = datetime.now(UTC)
        await self.db.flush()
        return document

    async def replace_chunks(
        self,
        document: RagDocument,
        chunks: list[dict],
    ) -> list[RagChunk]:
        await self.db.execute(delete(RagChunk).where(RagChunk.document_id == document.id))
        rows: list[RagChunk] = []
        for item in chunks:
            row = RagChunk(
                document_id=document.id,
                user_id=document.user_id,
                organization_id=document.organization_id,
                project_id=document.project_id,
                chunk_index=item["chunk_index"],
                content=item["content"],
                token_count=item["token_count"],
                metadata_json=json.dumps(item.get("metadata", {}), ensure_ascii=True),
                embedding_json=json.dumps(item.get("embedding", []), ensure_ascii=True),
                vector_external_id=item.get("vector_external_id"),
            )
            self.db.add(row)
            rows.append(row)
        await self.db.flush()
        await store_chunk_embeddings_batch(
            self.db,
            table="rag_chunks",
            items=[
                (row.id, item.get("embedding") or [])
                for row, item in zip(rows, chunks, strict=True)
            ],
        )
        await self.db.flush()
        return rows

    async def delete_chunks_for_document(self, document_id: str) -> None:
        await self.db.execute(delete(RagChunk).where(RagChunk.document_id == document_id))

    async def list_chunks_for_documents(self, document_ids: list[str]) -> list[RagChunk]:
        if not document_ids:
            return []
        result = await self.db.execute(
            select(RagChunk)
            .where(RagChunk.document_id.in_(document_ids))
            .order_by(RagChunk.document_id, RagChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def list_chunks_for_document(
        self,
        document_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagChunk], int]:
        stmt = (
            select(RagChunk)
            .where(RagChunk.document_id == document_id)
            .order_by(RagChunk.chunk_index)
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def pgvector_is_available(self) -> bool:
        return await check_pgvector_is_available(self.db)

    async def similarity_search_indexed(
        self,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk] | None:
        if not await check_pgvector_is_available(self.db):
            return None
        if not embedding_is_indexable(query_embedding):
            return None

        filters = [
            "c.user_id = :user_id",
            "c.embedding IS NOT NULL",
            "d.status = 'indexed'",
            "d.deleted_at IS NULL",
            "(1 - (c.embedding <=> CAST(:query_vec AS vector))) >= :score_threshold",
        ]
        params: dict = {
            "user_id": user_id,
            "query_vec": vector_literal(query_embedding),
            "score_threshold": score_threshold,
            "top_k": top_k,
        }
        if project_id:
            filters.append("c.project_id = :project_id")
            params["project_id"] = project_id
        if document_ids:
            filters.append("c.document_id = ANY(:document_ids)")
            params["document_ids"] = document_ids

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.chunk_index,
                c.metadata_json,
                d.original_filename,
                (1 - (c.embedding <=> CAST(:query_vec AS vector))) AS score
            FROM rag_chunks c
            INNER JOIN rag_documents d ON d.id = c.document_id
            WHERE {" AND ".join(filters)}
            ORDER BY c.embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
        """
        try:
            result = await self.db.execute(text(sql), params)
        except Exception:
            logger.exception("Indexed pgvector search failed")
            return None

        rows = result.mappings().all()
        retrieved: list[RetrievedChunk] = []
        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=round(float(row["score"]), 4),
                    filename=row["original_filename"],
                    chunk_index=row["chunk_index"],
                    page_number=meta.get("page_number"),
                    metadata=meta,
                )
            )
        return retrieved

    async def similarity_search_json_fallback(
        self,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        filters = [
            "c.user_id = :user_id",
            "c.embedding_json IS NOT NULL",
            "d.status = 'indexed'",
            "d.deleted_at IS NULL",
        ]
        params: dict = {
            "user_id": user_id,
            "max_candidates": json_fallback_max_candidates(top_k),
        }
        if project_id:
            filters.append("c.project_id = :project_id")
            params["project_id"] = project_id
        if document_ids:
            filters.append("c.document_id = ANY(:document_ids)")
            params["document_ids"] = document_ids

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.chunk_index,
                c.metadata_json,
                c.embedding_json,
                d.original_filename
            FROM rag_chunks c
            INNER JOIN rag_documents d ON d.id = c.document_id
            WHERE {" AND ".join(filters)}
            ORDER BY c.updated_at DESC
            LIMIT :max_candidates
        """
        result = await self.db.execute(text(sql), params)
        rows = [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata_json": row["metadata_json"],
                "original_filename": row["original_filename"],
                "embedding": parse_embedding_json(row["embedding_json"]),
            }
            for row in result.mappings().all()
        ]
        if len(rows) >= params["max_candidates"]:
            logger.warning(
                "JSON embedding fallback hit candidate cap (%s) for user=%s",
                params["max_candidates"],
                user_id,
            )

        def build_match(row: dict, score: float) -> RetrievedChunk:
            meta = json.loads(row["metadata_json"] or "{}")
            return RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                content=row["content"],
                score=score,
                filename=row["original_filename"],
                chunk_index=row["chunk_index"],
                page_number=meta.get("page_number"),
                metadata=meta,
            )

        return rank_embedding_matches(
            query_embedding,
            rows,
            top_k=top_k,
            score_threshold=score_threshold,
            build_match=build_match,
        )

    async def create_ingestion_job(
        self, *, document_id: str, user_id: str, project_id: str | None
    ) -> RagIngestionJob:
        job = RagIngestionJob(
            document_id=document_id,
            user_id=user_id,
            project_id=project_id,
            status=IngestionJobStatus.PENDING.value,
        )
        self.db.add(job)
        await self.db.flush()
        return job

    async def get_ingestion_job(self, job_id: str) -> RagIngestionJob | None:
        result = await self.db.execute(select(RagIngestionJob).where(RagIngestionJob.id == job_id))
        return result.scalar_one_or_none()

    async def update_ingestion_job(
        self,
        job: RagIngestionJob,
        *,
        status: IngestionJobStatus,
        error_message: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> RagIngestionJob:
        job.status = status.value
        if error_message is not None:
            job.error_message = error_message
        if started:
            job.started_at = datetime.now(UTC)
        if finished:
            job.finished_at = datetime.now(UTC)
        await self.db.flush()
        return job

    async def create_query_record(
        self,
        *,
        user_id: str,
        project_id: str | None,
        organization_id: str | None,
        query: str,
        answer: str,
        retrieved_chunk_ids: list[str],
        model_name: str,
        latency_ms: int,
    ) -> RagQueryRecord:
        row = RagQueryRecord(
            user_id=user_id,
            project_id=project_id,
            organization_id=organization_id,
            query=query,
            answer=answer,
            retrieved_chunk_ids_json=json.dumps(retrieved_chunk_ids, ensure_ascii=True),
            model_name=model_name,
            latency_ms=latency_ms,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_queries_for_user(
        self,
        user_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagQueryRecord], int]:
        stmt = (
            select(RagQueryRecord)
            .where(RagQueryRecord.user_id == user_id)
            .order_by(RagQueryRecord.created_at.desc())
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)
