from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile
from uuid import uuid4

from backend.core.config import settings
from backend.core.storage import (
    PRIVATE_UPLOAD_CACHE_CONTROL,
    ObjectStorageError,
    StorageNotConfiguredError,
    object_storage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoredDocument:
    storage_path: str
    size_bytes: int
    checksum_sha256: str
    content_type: str
    reused_existing: bool = False


def build_document_object_key(user_id: str, filename: str) -> str:
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return f"rag/{user_id}/{uuid4().hex}_{safe_name}"


class FileTooLargeError(ValueError):
    """Raised when a streamed upload exceeds the configured max size."""


class EmptyUploadError(ValueError):
    """Raised when a streamed upload contains no bytes."""


class FileStorageAdapter:
    async def store_document(
        self,
        *,
        user_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        max_bytes: int | None = None,
        dedupe_lookup: Callable[[str], Awaitable[StoredDocument | None]] | None = None,
    ) -> StoredDocument:
        """Store an in-memory payload (small text uploads). Prefer streaming for files."""

        async def _chunks() -> AsyncIterator[bytes]:
            yield content

        return await self.store_document_stream(
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            chunks=_chunks(),
            max_bytes=max_bytes if max_bytes is not None else len(content),
            dedupe_lookup=dedupe_lookup,
        )

    async def store_document_stream(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        chunks: AsyncIterator[bytes],
        max_bytes: int,
        chunk_size: int | None = None,
        dedupe_lookup: Callable[[str], Awaitable[StoredDocument | None]] | None = None,
    ) -> StoredDocument:
        """Stream chunks to spooled temp storage, hash, enforce size, then upload.

        Size limits are enforced incrementally before the full body is buffered.
        When ``dedupe_lookup`` returns an existing stored document for the
        computed checksum, object upload is skipped.
        """
        if not object_storage.is_configured:
            raise StorageNotConfiguredError(
                "Object storage is required to retain documents for asynchronous indexing"
            )

        resolved_chunk_hint = chunk_size or getattr(settings, "RAG_UPLOAD_CHUNK_SIZE", 256 * 1024)
        # Keep at most one upload chunk in RAM before spilling to disk.
        spool_max = max(64 * 1024, min(int(resolved_chunk_hint), 1_048_576))
        hasher = hashlib.sha256()
        total = 0
        object_key = build_document_object_key(user_id, filename)

        with SpooledTemporaryFile(max_size=spool_max) as spool:
            async for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise FileTooLargeError(f"Upload exceeds maximum size of {max_bytes} bytes")
                hasher.update(chunk)
                spool.write(chunk)

            if total == 0:
                raise EmptyUploadError("Uploaded document file is empty")

            checksum = hasher.hexdigest()
            if dedupe_lookup is not None:
                existing = await dedupe_lookup(checksum)
                if existing is not None:
                    return StoredDocument(
                        storage_path=existing.storage_path,
                        size_bytes=existing.size_bytes or total,
                        checksum_sha256=checksum,
                        content_type=existing.content_type or content_type,
                        reused_existing=True,
                    )

            spool.seek(0)
            try:
                await object_storage.ensure_bucket()
                await object_storage.upload_fileobj(
                    object_key=object_key,
                    fileobj=spool,
                    content_type=content_type,
                    cache_control=PRIVATE_UPLOAD_CACHE_CONTROL,
                )
            except StorageNotConfiguredError:
                raise
            except ObjectStorageError:
                raise
            except Exception as exc:
                raise ObjectStorageError("Failed to upload document to object storage") from exc

        return StoredDocument(
            storage_path=object_key,
            size_bytes=total,
            checksum_sha256=checksum,
            content_type=content_type,
            reused_existing=False,
        )

    async def delete_document(self, storage_path: str | None) -> None:
        if storage_path:
            await object_storage.delete_object(storage_path)

    async def download_document(self, storage_path: str) -> bytes:
        if not object_storage.is_configured:
            raise StorageNotConfiguredError("Object storage is not configured")

        def _get() -> bytes:
            response = object_storage._client.get_object(  # noqa: SLF001
                Bucket=settings.STORAGE_BUCKET,
                Key=storage_path,
            )
            return response["Body"].read()

        import asyncio

        return await asyncio.to_thread(_get)
