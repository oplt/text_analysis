from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from backend.core.pagination import DEFAULT_PAGE_LIMIT, paginate_scalars
from backend.lib.vector_search import (
    DenseFallbackScopeTooLarge,
    embedding_is_indexable,
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
from backend.modules.rag.infrastructure.lexical_languages import POSTGRES_TEXT_SEARCH_CONFIGS
from backend.modules.rag.infrastructure.models import (
    RagChunk,
    RagConversation,
    RagDocument,
    RagDocumentRevision,
    RagIngestionJob,
    RagMessage,
    RagQueryRecord,
    RagRetrievalTrace,
)
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_IN_CLAUSE_BATCH = 500
_LANGUAGE_METADATA_SQL = """
COALESCE(
    c.metadata_json::json->>'language',
    c.metadata_json::json->>'language_code',
    d.metadata_json::json->>'language',
    d.metadata_json::json->>'language_code',
    ''
)
"""
_LANGUAGE_CONFIG_CASE_SQL = "\n".join(
    [
        "CASE lower(split_part(" + _LANGUAGE_METADATA_SQL + ", '-', 1))",
        *[
            f"WHEN '{language}' THEN '{config}'"
            for language, config in sorted(POSTGRES_TEXT_SEARCH_CONFIGS.items())
        ],
        "ELSE 'simple' END",
    ]
)


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

    async def get_documents_by_ids(
        self,
        document_ids: list[str],
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[RagDocument]:
        """Fetch documents in bounded batches; optionally scope to ``user_id``."""
        if not document_ids:
            return []
        rows: list[RagDocument] = []
        for start in range(0, len(document_ids), _IN_CLAUSE_BATCH):
            stmt = select(RagDocument).where(
                RagDocument.id.in_(document_ids[start : start + _IN_CLAUSE_BATCH])
            )
            if not include_deleted:
                stmt = stmt.where(RagDocument.deleted_at.is_(None))
            if user_id is not None:
                stmt = stmt.where(RagDocument.user_id == user_id)
            result = await self.db.execute(stmt)
            rows.extend(result.scalars().all())
        return rows

    async def find_document_by_checksum(
        self,
        *,
        user_id: str,
        checksum_sha256: str,
        project_id: str | None = None,
    ) -> RagDocument | None:
        """Return the newest non-deleted document with matching content checksum.

        Checksums are hex digests stored in ``metadata_json``; matching is scoped
        to the owning user and optional project so provenance stays isolated.
        """
        digest = checksum_sha256.strip().lower()
        if not digest or any(c not in "0123456789abcdef" for c in digest):
            return None
        needle = f'"checksum_sha256": "{digest}"'
        stmt = (
            select(RagDocument)
            .where(
                RagDocument.user_id == user_id,
                RagDocument.deleted_at.is_(None),
                RagDocument.metadata_json.contains(needle),
            )
            .order_by(RagDocument.created_at.desc())
            .limit(1)
        )
        if project_id is not None:
            stmt = stmt.where(RagDocument.project_id == project_id)
        else:
            stmt = stmt.where(RagDocument.project_id.is_(None))
        result = await self.db.execute(stmt)
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

    async def create_document_revision(
        self,
        document: RagDocument,
        *,
        source_content_hash: str | None,
        parser_version: str | None,
        chunker_version: str | None,
        index_version: str | None,
        embedding_provider: str | None,
        embedding_model: str | None,
        embedding_model_version: str | None,
        embedding_dimensions: int | None,
        embedding_preprocessing_version: str | None,
    ) -> RagDocumentRevision:
        revision = RagDocumentRevision(
            document_id=document.id,
            source_content_hash=source_content_hash,
            parser_version=parser_version,
            chunker_version=chunker_version,
            index_version=index_version,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_model_version=embedding_model_version,
            embedding_dimensions=embedding_dimensions,
            embedding_preprocessing_version=embedding_preprocessing_version,
        )
        self.db.add(revision)
        await self.db.flush()
        return revision

    async def activate_document_revision(
        self, document: RagDocument, revision: RagDocumentRevision
    ) -> None:
        document.current_revision_id = revision.id
        await self.db.flush()

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
        *,
        revision_id: str | None = None,
        expected_embedding_dimensions: int | None = None,
    ) -> list[RagChunk]:
        # Revisioned indexing appends immutable rows. Legacy callers without a
        # revision retain the original replace semantics.
        if revision_id is None:
            await self.db.execute(delete(RagChunk).where(RagChunk.document_id == document.id))
        if expected_embedding_dimensions is not None:
            invalid = [
                item.get("id") or item.get("chunk_index")
                for item in chunks
                if item.get("embedding") is not None
                and len(item["embedding"]) != expected_embedding_dimensions
            ]
            if invalid:
                raise ValueError(
                    "chunk embeddings do not match expected dimension "
                    f"{expected_embedding_dimensions}: {invalid}"
                )
        rows: list[RagChunk] = []
        for item in chunks:
            meta = item.get("metadata") or {}
            embedding = item.get("embedding")
            row = RagChunk(
                document_id=document.id,
                user_id=document.user_id,
                organization_id=document.organization_id,
                project_id=document.project_id,
                chunk_index=item["chunk_index"],
                content=item["content"],
                token_count=item["token_count"],
                metadata_json=json.dumps(meta, ensure_ascii=True),
                embedding_json=(
                    json.dumps(embedding, ensure_ascii=True) if embedding is not None else None
                ),
                vector_external_id=item.get("vector_external_id"),
                content_hash=item.get("content_hash"),
                parent_chunk_id=item.get("parent_chunk_id"),
                parser_version=item.get("parser_version")
                or (meta.get("parser_version") if isinstance(meta, dict) else None),
                chunker_version=item.get("chunker_version")
                or (meta.get("chunker_version") if isinstance(meta, dict) else None),
                revision_id=revision_id,
            )
            if item.get("id"):
                row.id = item["id"]
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
            .join(RagDocument, RagDocument.id == RagChunk.document_id)
            .where(RagChunk.document_id.in_(document_ids))
            .where(
                (RagChunk.revision_id == RagDocument.current_revision_id)
                | (RagDocument.current_revision_id.is_(None) & RagChunk.revision_id.is_(None))
            )
            .order_by(RagChunk.document_id, RagChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def list_evidence_revision_chunks(
        self,
        document_ids: list[str],
        *,
        revision_ids: list[str] | None = None,
    ) -> list[RagChunk]:
        """Return indexed chunk identity fields for a reproducibility hash."""
        if not document_ids:
            return []
        rows: list[RagChunk] = []
        for start in range(0, len(document_ids), _IN_CLAUSE_BATCH):
            statement = (
                select(RagChunk)
                .join(RagDocument, RagDocument.id == RagChunk.document_id)
                .where(RagChunk.document_id.in_(document_ids[start : start + _IN_CLAUSE_BATCH]))
            )
            if revision_ids is None:
                statement = statement.where(
                    (RagChunk.revision_id == RagDocument.current_revision_id)
                    | (RagDocument.current_revision_id.is_(None) & RagChunk.revision_id.is_(None))
                )
            else:
                statement = statement.where(RagChunk.revision_id.in_(revision_ids))
            result = await self.db.execute(
                statement.order_by(RagChunk.document_id, RagChunk.chunk_index, RagChunk.id)
            )
            rows.extend(result.scalars().all())
        return rows

    async def list_current_revision_ids(self, document_ids: list[str] | None) -> list[str]:
        if not document_ids:
            return []
        result = await self.db.execute(
            select(RagDocument.current_revision_id).where(
                RagDocument.id.in_(document_ids),
                RagDocument.current_revision_id.is_not(None),
            )
        )
        return list(dict.fromkeys(result.scalars().all()))

    async def list_chunks_for_document(
        self,
        document_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagChunk], int]:
        stmt = (
            select(RagChunk)
            .join(RagDocument, RagDocument.id == RagChunk.document_id)
            .where(
                RagChunk.document_id == document_id,
                (RagChunk.revision_id == RagDocument.current_revision_id)
                | (RagDocument.current_revision_id.is_(None) & RagChunk.revision_id.is_(None)),
            )
            .order_by(RagChunk.chunk_index)
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def pgvector_is_available(self) -> bool:
        return await check_pgvector_is_available(self.db)

    def _retrieval_scope_filters(
        self,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        owner_scoped: bool,
        params: dict,
        exclude_parents: bool = False,
        index_revision_ids: list[str] | None = None,
    ) -> list[str] | None:
        """Build shared WHERE clauses. Returns None when empty allow-list (I1)."""
        from backend.modules.rag.application.document_scope import (
            document_ids_is_empty_allow_list,
            should_apply_document_id_filter,
        )

        if document_ids_is_empty_allow_list(document_ids):
            return None

        filters = [
            "d.status = 'indexed'",
            "d.deleted_at IS NULL",
        ]
        if index_revision_ids is None:
            filters.append(
                "(d.current_revision_id = c.revision_id OR "
                "(d.current_revision_id IS NULL AND c.revision_id IS NULL))"
            )
        elif not index_revision_ids:
            return None
        else:
            filters.append("c.revision_id = ANY(:index_revision_ids)")
            params["index_revision_ids"] = index_revision_ids
        if owner_scoped:
            filters.append("c.user_id = :user_id")
            params["user_id"] = user_id
        if project_id:
            filters.append("c.project_id = :project_id")
            params["project_id"] = project_id
        if should_apply_document_id_filter(document_ids):
            filters.append("c.document_id = ANY(:document_ids)")
            params["document_ids"] = list(document_ids or [])
        if exclude_parents:
            # Parents use negative chunk_index and/or chunk_role=parent metadata.
            filters.append("c.chunk_index >= 0")
            filters.append(
                "(c.metadata_json IS NULL OR "
                "COALESCE(c.metadata_json::json->>'chunk_role', 'child') <> 'parent')"
            )
        return filters

    async def similarity_search_indexed(
        self,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
        owner_scoped: bool = True,
        exclude_parents: bool = False,
        index_revision_ids: list[str] | None = None,
    ) -> list[RetrievedChunk] | None:
        if not await check_pgvector_is_available(self.db):
            return None
        if not embedding_is_indexable(query_embedding):
            return None

        params: dict = {
            "query_vec": vector_literal(query_embedding),
            "score_threshold": score_threshold,
            "top_k": top_k,
        }
        filters = self._retrieval_scope_filters(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            owner_scoped=owner_scoped,
            params=params,
            exclude_parents=exclude_parents,
            index_revision_ids=index_revision_ids,
        )
        if filters is None:
            return []

        filters.append("c.embedding IS NOT NULL")
        filters.append("(1 - (c.embedding <=> CAST(:query_vec AS vector))) >= :score_threshold")

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.chunk_index,
                c.metadata_json,
                c.revision_id,
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
                    retrieval_sources=("dense",),
                    index_revision_id=row["revision_id"],
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
        owner_scoped: bool = True,
        exclude_parents: bool = False,
        index_revision_ids: list[str] | None = None,
        exact_max_rows: int | None = None,
    ) -> list[RetrievedChunk]:
        params: dict = {}
        filters = self._retrieval_scope_filters(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            owner_scoped=owner_scoped,
            params=params,
            exclude_parents=exclude_parents,
            index_revision_ids=index_revision_ids,
        )
        if filters is None:
            return []

        filters.append("c.embedding_json IS NOT NULL")

        count_result = await self.db.execute(
            text(
                "SELECT COUNT(*) FROM rag_chunks c "
                "INNER JOIN rag_documents d ON d.id = c.document_id "
                f"WHERE {' AND '.join(filters)}"
            ),
            params,
        )
        eligible_count = int(count_result.scalar() or 0)
        max_rows = exact_max_rows if exact_max_rows is not None else 5000
        if eligible_count > max_rows:
            raise DenseFallbackScopeTooLarge(
                f"pgvector_required_for_scope_size:{eligible_count}>{max_rows}"
            )

        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.chunk_index,
                c.metadata_json,
                c.revision_id,
                c.embedding_json,
                d.original_filename
            FROM rag_chunks c
            INNER JOIN rag_documents d ON d.id = c.document_id
            WHERE {" AND ".join(filters)}
            ORDER BY c.document_id, c.chunk_index, c.id
        """
        result = await self.db.execute(text(sql), params)
        rows = [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata_json": row["metadata_json"],
                "revision_id": row["revision_id"],
                "original_filename": row["original_filename"],
                "embedding": parse_embedding_json(row["embedding_json"]),
            }
            for row in result.mappings().all()
        ]

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
                retrieval_sources=("dense",),
                index_revision_id=row.get("revision_id"),
            )

        return rank_embedding_matches(
            query_embedding,
            rows,
            top_k=top_k,
            score_threshold=score_threshold,
            build_match=build_match,
        )

    async def lexical_search(
        self,
        *,
        user_id: str,
        project_id: str | None,
        document_ids: list[str] | None,
        query: str,
        top_k: int,
        owner_scoped: bool = True,
        exclude_parents: bool = False,
        index_revision_ids: list[str] | None = None,
        phrase_boost: bool = False,
    ) -> list[RetrievedChunk]:
        """Independent PostgreSQL full-text lexical ranking path."""
        params: dict = {
            "query": query,
            "top_k": top_k,
            "phrase_boost": phrase_boost,
        }
        filters = self._retrieval_scope_filters(
            user_id=user_id,
            project_id=project_id,
            document_ids=document_ids,
            owner_scoped=owner_scoped,
            params=params,
            exclude_parents=exclude_parents,
            index_revision_ids=index_revision_ids,
        )
        if filters is None:
            return []

        # A document's declared language uses its PostgreSQL stemmer where
        # available; every document is additionally searched with ``simple``.
        # This keeps Turkish and unknown languages safe and handles mixed corpora.
        sql = f"""
            WITH scoped_chunks AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.metadata_json,
                    c.revision_id,
                    d.original_filename,
                    ({_LANGUAGE_CONFIG_CASE_SQL})::regconfig AS language_config,
                    COALESCE(
                        c.content_tsv,
                        to_tsvector('simple', coalesce(c.content, ''))
                    ) AS simple_vector
                FROM rag_chunks c
                INNER JOIN rag_documents d ON d.id = c.document_id
                WHERE {" AND ".join(filters)}
            )
            SELECT *, GREATEST(
                ts_rank_cd(
                    to_tsvector(language_config, coalesce(content, '')),
                    plainto_tsquery(language_config, :query)
                ),
                ts_rank_cd(simple_vector, plainto_tsquery('simple', :query))
            ) + CASE WHEN :phrase_boost AND (
                to_tsvector(language_config, coalesce(content, ''))
                    @@ phraseto_tsquery(language_config, :query)
                OR simple_vector @@ phraseto_tsquery('simple', :query)
            ) THEN 1 ELSE 0 END AS score
            FROM scoped_chunks
            WHERE to_tsvector(language_config, coalesce(content, ''))
                @@ plainto_tsquery(language_config, :query)
               OR simple_vector @@ plainto_tsquery('simple', :query)
            ORDER BY score DESC
            LIMIT :top_k
        """
        try:
            result = await self.db.execute(text(sql), params)
        except Exception:
            logger.exception("Lexical full-text search failed")
            raise

        retrieved: list[RetrievedChunk] = []
        for rank, row in enumerate(result.mappings().all(), start=1):
            meta = json.loads(row["metadata_json"] or "{}")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=round(float(row["score"] or 0.0), 4),
                    filename=row["original_filename"],
                    chunk_index=row["chunk_index"],
                    page_number=meta.get("page_number"),
                    metadata=meta,
                    rank=rank,
                    retrieval_sources=("lexical",),
                    index_revision_id=row["revision_id"],
                )
            )
        return retrieved

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

    async def get_active_ingestion_job(self, document_id: str) -> RagIngestionJob | None:
        result = await self.db.execute(
            select(RagIngestionJob)
            .where(
                RagIngestionJob.document_id == document_id,
                RagIngestionJob.status.in_(
                    [IngestionJobStatus.PENDING.value, IngestionJobStatus.RUNNING.value]
                ),
            )
            .order_by(RagIngestionJob.created_at.desc())
        )
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

    async def create_conversation(
        self,
        *,
        user_id: str,
        project_id: str | None,
        organization_id: str | None = None,
        title: str = "Untitled",
        scope_snapshot: dict | None = None,
    ) -> RagConversation:
        row = RagConversation(
            user_id=user_id,
            project_id=project_id,
            organization_id=organization_id,
            title=title,
            scope_snapshot_json=json.dumps(scope_snapshot or {}, ensure_ascii=True),
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_conversation(self, conversation_id: str) -> RagConversation | None:
        result = await self.db.execute(
            select(RagConversation).where(RagConversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def list_conversations_for_user(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagConversation], int]:
        stmt = select(RagConversation).where(
            RagConversation.user_id == user_id,
            RagConversation.archived_at.is_(None),
        )
        if project_id is not None:
            stmt = stmt.where(RagConversation.project_id == project_id)
        stmt = stmt.order_by(RagConversation.updated_at.desc())
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def create_retrieval_trace(
        self,
        *,
        user_id: str,
        project_id: str | None,
        conversation_id: str | None,
        query: str,
        intent: str | None,
        scope_hash: str | None,
        evidence_revision_hash: str | None = None,
        index_revision_ids: list[str] | None = None,
        document_ids: list[str] | None,
        retrieved_chunks: list[dict],
        coverage: dict | None,
        config: dict | None,
        degraded: bool,
        degradation_reason: str | None,
        no_matches: bool,
        injection_chunks_filtered: int,
        latency_ms: int,
        request_id: str | None = None,
        stage_timings: dict | None = None,
        branch_status: dict | None = None,
    ) -> RagRetrievalTrace:
        row = RagRetrievalTrace(
            user_id=user_id,
            project_id=project_id,
            conversation_id=conversation_id,
            query=query,
            request_id=request_id,
            intent=intent,
            scope_hash=scope_hash,
            evidence_revision_hash=evidence_revision_hash,
            index_revision_ids_json=json.dumps(index_revision_ids or [], ensure_ascii=True),
            document_ids_json=json.dumps(document_ids, ensure_ascii=True)
            if document_ids is not None
            else None,
            retrieved_chunks_json=json.dumps(retrieved_chunks, ensure_ascii=True),
            coverage_json=json.dumps(coverage or {}, ensure_ascii=True),
            config_json=json.dumps(config or {}, ensure_ascii=True),
            degraded=degraded,
            degradation_reason=degradation_reason,
            no_matches=no_matches,
            injection_chunks_filtered=injection_chunks_filtered,
            latency_ms=latency_ms,
            stage_timings_json=json.dumps(stage_timings or {}, ensure_ascii=True),
            branch_status_json=json.dumps(branch_status or {}, ensure_ascii=True),
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_retrieval_trace(self, trace_id: str) -> RagRetrievalTrace | None:
        result = await self.db.execute(
            select(RagRetrievalTrace).where(RagRetrievalTrace.id == trace_id)
        )
        return result.scalar_one_or_none()

    async def create_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        model_name: str | None = None,
        prompt_template_id: str | None = None,
        prompt_version_id: str | None = None,
        retrieval_trace_id: str | None = None,
        citations: list[dict] | None = None,
        claims: list[dict] | None = None,
        metadata: dict | None = None,
        citation_validation_status: str | None = None,
        resolved_retrieval_query: str | None = None,
        context_message_ids: list[str] | None = None,
        ai_run_id: str | None = None,
        evidence_revision_hash: str | None = None,
    ) -> RagMessage:
        meta = metadata or {}
        row = RagMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model_name=model_name,
            prompt_template_id=prompt_template_id,
            prompt_version_id=prompt_version_id,
            retrieval_trace_id=retrieval_trace_id,
            citations_json=json.dumps(citations or [], ensure_ascii=True),
            claims_json=json.dumps(claims or [], ensure_ascii=True),
            metadata_json=json.dumps(meta, ensure_ascii=True),
            citation_validation_status=citation_validation_status
            or meta.get("citation_validation_status"),
            resolved_retrieval_query=resolved_retrieval_query
            or meta.get("resolved_retrieval_query"),
            context_message_ids_json=json.dumps(
                context_message_ids
                if context_message_ids is not None
                else meta.get("context_message_ids") or [],
                ensure_ascii=True,
            ),
            ai_run_id=ai_run_id or meta.get("ai_run_id"),
            evidence_revision_hash=evidence_revision_hash or meta.get("evidence_revision_hash"),
        )
        self.db.add(row)
        await self.db.flush()
        conversation = await self.get_conversation(conversation_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(UTC)
            await self.db.flush()
        return row

    async def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[RagMessage], int]:
        stmt = (
            select(RagMessage)
            .where(RagMessage.conversation_id == conversation_id)
            .order_by(RagMessage.created_at.asc())
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[RagChunk]:
        if not chunk_ids:
            return []
        result = await self.db.execute(select(RagChunk).where(RagChunk.id.in_(chunk_ids)))
        return list(result.scalars().all())

    async def list_available_revision_ids(
        self, *, document_ids: list[str], revision_ids: list[str]
    ) -> list[str]:
        """Return frozen revisions that still have retrievable chunk evidence."""
        if not document_ids or not revision_ids:
            return []
        result = await self.db.execute(
            select(RagChunk.revision_id)
            .join(RagDocument, RagDocument.id == RagChunk.document_id)
            .where(
                RagChunk.document_id.in_(document_ids),
                RagChunk.revision_id.in_(revision_ids),
                RagDocument.deleted_at.is_(None),
                RagDocument.status == DocumentStatus.INDEXED.value,
            )
            .distinct()
        )
        return [revision_id for revision_id in result.scalars().all() if revision_id]
