"""Phase 21: streaming upload, size enforcement, failed ingestion paths."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService
from backend.modules.rag.domain.enums import DocumentStatus, IngestionJobStatus
from backend.modules.rag.infrastructure.file_storage_adapter import (
    EmptyUploadError,
    FileStorageAdapter,
    FileTooLargeError,
)


async def _aiter_bytes(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


class StreamUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_document_stream_enforces_max_bytes_incrementally(self) -> None:
        adapter = FileStorageAdapter()
        with (
            patch(
                "backend.modules.rag.infrastructure.file_storage_adapter.object_storage"
            ) as storage,
        ):
            storage.is_configured = True
            with self.assertRaises(FileTooLargeError):
                await adapter.store_document_stream(
                    user_id="user-a",
                    filename="big.txt",
                    content_type="text/plain",
                    chunks=_aiter_bytes([b"abcd", b"efgh"]),
                    max_bytes=6,
                )
            self.assertFalse(storage.upload_fileobj.called)

    async def test_store_document_stream_rejects_empty(self) -> None:
        adapter = FileStorageAdapter()
        with patch(
            "backend.modules.rag.infrastructure.file_storage_adapter.object_storage"
        ) as storage:
            storage.is_configured = True
            with self.assertRaises(EmptyUploadError):
                await adapter.store_document_stream(
                    user_id="user-a",
                    filename="empty.txt",
                    content_type="text/plain",
                    chunks=_aiter_bytes([b"", b""]),
                    max_bytes=100,
                )

    async def test_upload_document_uses_stream_path_for_upload_file(self) -> None:
        db = AsyncMock()
        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(
            enabled=True,
            max_file_bytes=1_000_000,
            allowed_file_types=("txt",),
        )
        service.policy = MagicMock()
        service.policy.is_allowed_file_type.return_value = True
        service.repo = MagicMock()
        service.repo.create_document = AsyncMock(
            return_value=SimpleNamespace(id="doc-1", status=DocumentStatus.UPLOADED.value)
        )
        service.repo.create_ingestion_job = AsyncMock(return_value=SimpleNamespace(id="job-1"))
        service.repo.find_document_by_checksum = AsyncMock(return_value=None)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        upload = MagicMock()
        upload.filename = "notes.txt"
        upload.content_type = "text/plain"

        with patch.object(
            service,
            "_store_upload_stream",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    storage_path="rag/key",
                    size_bytes=5,
                    checksum_sha256="abc",
                    content_type="text/plain",
                    reused_existing=False,
                )
            ),
        ) as store_stream:
            document, job, content = await service.upload_document(
                user_id="user-a",
                filename="notes.txt",
                content_type="text/plain",
                upload=upload,
            )

        store_stream.assert_awaited_once()
        self.assertEqual(document.id, "doc-1")
        self.assertEqual(job.id, "job-1")
        self.assertIsNone(content)


class FailedIngestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_document_marks_failed_on_unexpected_error(self) -> None:
        db = AsyncMock()
        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(enabled=True)
        document = SimpleNamespace(
            id="doc-1",
            user_id="user-a",
            project_id=None,
            organization_id=None,
            original_filename="notes.txt",
            content_type="text/plain",
            storage_path="rag/key",
            status=DocumentStatus.UPLOADED.value,
            metadata_json="{}",
        )
        job = SimpleNamespace(
            id="job-1",
            document_id="doc-1",
            status=IngestionJobStatus.PENDING.value,
        )
        service.repo = MagicMock()
        service.repo.get_ingestion_job = AsyncMock(return_value=job)
        service.repo.update_document_status = AsyncMock()
        service.repo.update_ingestion_job = AsyncMock()
        service.repo.create_ingestion_job = AsyncMock(return_value=job)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        with (
            patch.object(
                service,
                "_get_document_for_indexing",
                new=AsyncMock(return_value=document),
            ),
            patch.object(
                service.storage,
                "download_document",
                new=AsyncMock(return_value=b"hello"),
            ),
            patch.object(
                service.parser,
                "parse_bytes",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            self.assertRaises(HTTPException) as ctx,
        ):
            await service.index_document(
                document_id="doc-1",
                user_id="user-a",
                job_id="job-1",
            )

        self.assertEqual(ctx.exception.status_code, 502)
        failed_calls = [
            call
            for call in service.repo.update_document_status.await_args_list
            if call.args[1] == DocumentStatus.FAILED
        ]
        self.assertTrue(failed_calls)
        self.assertTrue(
            any(
                call.kwargs.get("status") == IngestionJobStatus.FAILED
                for call in service.repo.update_ingestion_job.await_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
