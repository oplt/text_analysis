from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

from backend.core.storage import ObjectStorageError, StorageNotConfiguredError
from backend.lib.project_access import ProjectAccessPort, SqlAlchemyProjectAccessPort
from backend.lib.retrieval_cache import invalidate_retrieval_cache_for_document
from backend.modules.rag.application.chunking_service import ChunkingService
from backend.modules.rag.application.document_parser_service import DocumentParserService
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.rag_policy_service import RagPolicyService
from backend.modules.rag.domain.enums import DocumentStatus, IngestionJobStatus
from backend.modules.rag.infrastructure import metrics
from backend.modules.rag.infrastructure.file_storage_adapter import (
    EmptyUploadError,
    FileStorageAdapter,
    FileTooLargeError,
    StoredDocument,
)
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.rag.infrastructure.vector_store_adapter import build_vector_store
from backend.modules.rag.workers import queue_document_cleanup, queue_document_indexing
from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self, db: AsyncSession, config: RagConfig | None = None):
        self.db = db
        self.config = config or RagConfig.from_settings()
        self.repo = RagRepository(db)
        self.project_access: ProjectAccessPort = SqlAlchemyProjectAccessPort(db)
        self.parser = DocumentParserService()
        self.chunker = ChunkingService(self.config)
        self.embeddings = EmbeddingService(self.config)
        self.policy = RagPolicyService()
        self.storage = FileStorageAdapter()
        self.vector_store = build_vector_store(db, self.config)

    async def upload_document(
        self,
        *,
        user_id: str,
        filename: str,
        content: bytes | None = None,
        content_type: str,
        project_id: str | None = None,
        organization_id: str | None = None,
        metadata: dict | None = None,
        upload: UploadFile | None = None,
    ):
        """Upload a document from bytes or a streamed ``UploadFile``.

        Prefer ``upload=`` for HTTP multipart so the whole body is not buffered
        before size enforcement. ``content=`` remains for small in-memory text.
        """
        if not self.config.enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")

        if content is None and upload is None:
            raise HTTPException(status_code=400, detail="No file content provided")
        if content is not None and upload is not None:
            raise HTTPException(
                status_code=400,
                detail="Provide either content or upload, not both",
            )

        if not self.policy.is_allowed_file_type(filename, self.config.allowed_file_types):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        if project_id:
            await self._ensure_project_access(user_id, project_id)

        async def _dedupe_lookup(checksum: str) -> StoredDocument | None:
            existing = await self.repo.find_document_by_checksum(
                user_id=user_id,
                checksum_sha256=checksum,
                project_id=project_id,
            )
            if existing is None or not existing.storage_path:
                return None
            meta = json.loads(existing.metadata_json or "{}")
            return StoredDocument(
                storage_path=existing.storage_path,
                size_bytes=int(meta.get("size_bytes") or 0),
                checksum_sha256=checksum,
                content_type=existing.content_type,
                reused_existing=True,
            )

        try:
            if upload is not None:
                stored = await self._store_upload_stream(
                    user_id=user_id,
                    filename=filename,
                    content_type=content_type,
                    upload=upload,
                    dedupe_lookup=_dedupe_lookup,
                )
            else:
                assert content is not None
                if len(content) > self.config.max_file_bytes:
                    raise HTTPException(status_code=413, detail="File too large")
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded document file is empty")
                stored = await self.storage.store_document(
                    user_id=user_id,
                    filename=filename,
                    content=content,
                    content_type=content_type,
                    max_bytes=self.config.max_file_bytes,
                    dedupe_lookup=_dedupe_lookup,
                )
        except FileTooLargeError as exc:
            raise HTTPException(status_code=413, detail="File too large") from exc
        except EmptyUploadError as exc:
            raise HTTPException(status_code=400, detail="Uploaded document file is empty") from exc
        except StorageNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ObjectStorageError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Object storage unavailable. Start MinIO "
                    "(`docker compose -f infra/docker-compose.yml up -d minio`) "
                    "and confirm STORAGE_* credentials match MINIO_ROOT_*."
                ),
            ) from exc

        if stored.reused_existing:
            document = await self.repo.find_document_by_checksum(
                user_id=user_id,
                checksum_sha256=stored.checksum_sha256,
                project_id=project_id,
            )
            if document is None:
                raise HTTPException(status_code=500, detail="Duplicate document lookup failed")
            active_job = await self.repo.get_active_ingestion_job(document.id)
            if active_job:
                return document, active_job, None
            job = await self.repo.create_ingestion_job(
                document_id=document.id,
                user_id=user_id,
                project_id=project_id,
            )
            await self.db.commit()
            await self.db.refresh(document)
            await self.db.refresh(job)
            return document, job, None

        document_metadata = {
            "size_bytes": stored.size_bytes,
            "checksum_sha256": stored.checksum_sha256,
            "content_type": stored.content_type,
            "storage_key": stored.storage_path,
            **(metadata or {}),
        }
        # Prefer measured size/checksum over caller overrides.
        document_metadata["size_bytes"] = stored.size_bytes
        document_metadata["checksum_sha256"] = stored.checksum_sha256
        document_metadata["storage_key"] = stored.storage_path

        document = await self.repo.create_document(
            user_id=user_id,
            filename=filename,
            original_filename=filename,
            content_type=content_type,
            storage_path=stored.storage_path,
            project_id=project_id,
            organization_id=organization_id,
            source_type="upload",
            metadata=document_metadata,
        )
        job = await self.repo.create_ingestion_job(
            document_id=document.id,
            user_id=user_id,
            project_id=project_id,
        )
        await self.db.commit()
        await self.db.refresh(document)
        await self.db.refresh(job)
        metrics.rag_document_upload_total.inc()

        # Content is not kept in memory after streaming; callers download from storage.
        return document, job, None

    async def _store_upload_stream(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        upload: UploadFile,
        dedupe_lookup: Callable[[str], Awaitable[StoredDocument | None]] | None = None,
    ) -> StoredDocument:
        from backend.core.config import settings

        chunk_size = max(8 * 1024, int(getattr(settings, "RAG_UPLOAD_CHUNK_SIZE", 256 * 1024)))

        async def _chunks() -> AsyncIterator[bytes]:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        return await self.storage.store_document_stream(
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            chunks=_chunks(),
            max_bytes=self.config.max_file_bytes,
            chunk_size=chunk_size,
            dedupe_lookup=dedupe_lookup,
        )

    async def enqueue_document_indexing(
        self,
        *,
        document_id: str,
        user_id: str,
        file_content: bytes | None = None,
        is_admin: bool = False,
    ):
        document = await self._get_document_for_indexing(
            document_id=document_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        active_job = await self.repo.get_active_ingestion_job(document.id)
        if active_job:
            return active_job
        try:
            async with self.db.begin_nested():
                job = await self.repo.create_ingestion_job(
                    document_id=document.id,
                    user_id=user_id,
                    project_id=document.project_id,
                )
        except IntegrityError:
            active_job = await self.repo.get_active_ingestion_job(document.id)
            if active_job:
                return active_job
            raise
        await self.db.commit()
        await self.db.refresh(job)
        queue_document_indexing(
            document_id=document.id,
            user_id=user_id,
            job_id=job.id,
        )
        return job

    async def index_document(
        self,
        *,
        document_id: str,
        user_id: str,
        file_content: bytes | None = None,
        is_admin: bool = False,
        job_id: str | None = None,
    ):
        document = await self._get_document_for_indexing(
            document_id=document_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        if job_id:
            job = await self.repo.get_ingestion_job(job_id)
            if not job or job.document_id != document.id:
                raise HTTPException(status_code=404, detail="Ingestion job not found")
        else:
            job = await self.repo.create_ingestion_job(
                document_id=document.id,
                user_id=user_id,
                project_id=document.project_id,
            )
        await self.repo.update_ingestion_job(job, status=IngestionJobStatus.RUNNING, started=True)
        await self.repo.update_document_status(document, DocumentStatus.PARSING)
        await self.db.commit()

        try:
            content = file_content
            if content is None and document.storage_path:
                content = await self.storage.download_document(document.storage_path)
            if content is None:
                raise HTTPException(
                    status_code=422,
                    detail="No file content available for indexing",
                )

            parsed = await self.parser.parse_bytes(
                content=content,
                filename=document.original_filename,
                content_type=document.content_type,
                metadata={"document_id": document.id},
            )
            if not parsed:
                raise ValueError("No text extracted from document")
            metrics.rag_parse_success_total.inc()

            await self.repo.update_document_status(document, DocumentStatus.CHUNKING)
            await self.db.commit()

            chunks = await self.chunker.chunk(
                parsed,
                document_id=document.id,
                user_id=document.user_id,
                filename=document.original_filename,
                project_id=document.project_id,
                organization_id=document.organization_id,
            )
            injection_flags = 0
            for chunk in chunks:
                if self.policy.contains_prompt_injection(chunk.content):
                    chunk.metadata = {
                        **chunk.metadata,
                        "prompt_injection_suspected": True,
                    }
                    injection_flags += 1
            if injection_flags:
                logger.warning(
                    "Suspected prompt injection in %s chunk(s) for document=%s user=%s",
                    injection_flags,
                    document.id,
                    document.user_id,
                )
            metrics.rag_chunk_count.observe(len(chunks))

            await self.repo.update_document_status(document, DocumentStatus.EMBEDDING)
            await self.db.commit()

            texts = [chunk.content for chunk in chunks]
            vectors = await self.embeddings.embed_texts(texts)
            for chunk, vector in zip(chunks, vectors, strict=True):
                # Parents are expansion context only — do not index as retrieval candidates.
                if (
                    chunk.chunk_index < 0
                    or (chunk.metadata or {}).get("chunk_role") == "parent"
                ):
                    chunk.embedding = []
                else:
                    chunk.embedding = vector

            chunk_rows = await self.repo.replace_chunks(
                document,
                [
                    {
                        "id": c.id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "token_count": c.token_count,
                        "metadata": c.metadata,
                        "embedding": c.embedding or [],
                        "vector_external_id": c.id,
                        "content_hash": c.content_hash,
                        "parent_chunk_id": c.parent_chunk_id,
                        "parser_version": (c.metadata or {}).get("parser_version"),
                        "chunker_version": (c.metadata or {}).get("chunker_version"),
                    }
                    for c in chunks
                ],
            )

            await self.repo.update_document_status(document, DocumentStatus.INDEXED)
            metadata = json.loads(document.metadata_json or "{}")
            metadata["chunk_count"] = len(chunk_rows)
            document.metadata_json = json.dumps(metadata, ensure_ascii=True)
            await self.repo.update_ingestion_job(
                job, status=IngestionJobStatus.COMPLETED, finished=True
            )
            document.updated_at = datetime.now(UTC)
            await self.db.commit()
            await self.db.refresh(document)
            await invalidate_retrieval_cache_for_document(
                user_id=document.user_id,
                project_id=document.project_id,
            )
            return document, chunk_rows, job
        except HTTPException as exc:
            await self.repo.update_document_status(document, DocumentStatus.FAILED)
            await self.repo.update_ingestion_job(
                job,
                status=IngestionJobStatus.FAILED,
                error_message=(
                    exc.detail[:500] if isinstance(exc.detail, str) else "Document indexing failed"
                ),
                finished=True,
            )
            await self.db.commit()
            raise
        except Exception as exc:
            logger.exception("Document indexing failed for %s", document_id)
            metrics.rag_parse_failure_total.inc()
            metrics.rag_vector_upsert_failure_total.inc()
            await self.repo.update_document_status(document, DocumentStatus.FAILED)
            await self.repo.update_ingestion_job(
                job,
                status=IngestionJobStatus.FAILED,
                error_message=str(exc)[:500],
                finished=True,
            )
            await self.db.commit()
            raise HTTPException(status_code=502, detail="Document indexing failed") from exc

    async def delete_document(self, *, document_id: str, user_id: str, is_admin: bool = False):
        document = await self.repo.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != user_id and not is_admin:
            metrics.rag_permission_denied_total.inc()
            raise HTTPException(status_code=403, detail="You cannot delete this document")

        await self.repo.soft_delete_document(document)
        await self.db.commit()
        await invalidate_retrieval_cache_for_document(
            user_id=document.user_id,
            project_id=document.project_id,
        )
        queue_document_cleanup(
            document_id=document_id,
            user_id=document.user_id,
            storage_path=document.storage_path,
        )

    async def cleanup_deleted_document(
        self,
        *,
        document_id: str,
        user_id: str,
        storage_path: str | None,
    ) -> None:
        await self.vector_store.delete_document(document_id, user_id)
        await self.storage.delete_document(storage_path)
        await self.db.commit()
        logger.info("RAG document cleanup completed document=%s user=%s", document_id, user_id)

    async def _get_document_for_indexing(
        self,
        *,
        document_id: str,
        user_id: str,
        is_admin: bool,
    ):
        document = await self.repo.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.user_id != user_id and not is_admin:
            raise HTTPException(status_code=403, detail="You cannot index this document")
        if document.project_id:
            await self._ensure_project_access(user_id, document.project_id)
        return document

    async def _ensure_project_access(self, user_id: str, project_id: str) -> None:
        await self.project_access.ensure_project_access(user_id, project_id)
