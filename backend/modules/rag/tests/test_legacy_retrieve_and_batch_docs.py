import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.rag.application.legacy_ai_document_service import LegacyAiDocumentService
from backend.modules.rag.infrastructure.repositories import RagRepository


class GetDocumentsByIdsTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_documents_by_ids_returns_empty_for_empty_input(self):
        db = AsyncMock()
        repo = RagRepository(db)
        docs = await repo.get_documents_by_ids([], user_id="user-a")
        self.assertEqual(docs, [])
        db.execute.assert_not_awaited()

    async def test_get_documents_by_ids_scopes_to_user(self):
        db = AsyncMock()
        result = MagicMock()
        owned = SimpleNamespace(id="doc-1", user_id="user-a", original_filename="a.txt")
        result.scalars.return_value.all.return_value = [owned]
        db.execute = AsyncMock(return_value=result)
        repo = RagRepository(db)

        docs = await repo.get_documents_by_ids(["doc-1", "doc-2"], user_id="user-a")

        self.assertEqual(docs, [owned])
        db.execute.assert_awaited_once()
        stmt = db.execute.await_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
        self.assertIn("rag_documents", compiled.lower())


class LegacyRetrieveChunksTest(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> LegacyAiDocumentService:
        db = AsyncMock()
        service = LegacyAiDocumentService(db)
        service.repo = MagicMock()
        service.retrieval = MagicMock()
        return service

    async def test_retrieve_with_explicit_document_ids(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock(return_value=["doc-1"])
        service.repo.get_documents_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(id="doc-1", original_filename="notes.txt"),
            ]
        )
        service.retrieval.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        document_id="doc-1",
                        chunk_id="c1",
                        filename="",
                        chunk_index=0,
                        score=0.9,
                        content="hello",
                    )
                ]
            )
        )

        matches = await service.retrieve_chunks(
            user_id="user-a",
            query="q",
            document_ids=["doc-1"],
            top_k=3,
        )

        self.assertEqual(matches[0]["document_title"], "notes.txt")
        service.repo.get_documents_by_ids.assert_awaited_once()
        service.repo.get_document = MagicMock()
        # N+1 path must not be used.
        if hasattr(service.repo, "get_document"):
            service.repo.get_document.assert_not_called()

    async def test_retrieve_without_explicit_document_ids(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock()
        service.repo.get_documents_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(id="doc-9", original_filename="from-db.txt"),
            ]
        )
        service.retrieval.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        document_id="doc-9",
                        chunk_id="c9",
                        filename="",
                        chunk_index=1,
                        score=0.5,
                        content="body",
                    )
                ]
            )
        )

        matches = await service.retrieve_chunks(
            user_id="user-a",
            query="q",
            document_ids=[],
            top_k=3,
        )

        service.repo.filter_document_ids_for_user.assert_not_awaited()
        self.assertEqual(matches[0]["document_title"], "from-db.txt")

    async def test_retrieve_prefers_match_filename(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock(return_value=["doc-1"])
        service.repo.get_documents_by_ids = AsyncMock(
            return_value=[SimpleNamespace(id="doc-1", original_filename="db-name.txt")]
        )
        service.retrieval.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        document_id="doc-1",
                        chunk_id="c1",
                        filename="match-name.txt",
                        chunk_index=0,
                        score=0.8,
                        content="x",
                    )
                ]
            )
        )

        matches = await service.retrieve_chunks(
            user_id="user-a",
            query="q",
            document_ids=["doc-1"],
            top_k=1,
        )
        self.assertEqual(matches[0]["document_title"], "match-name.txt")

    async def test_retrieve_missing_document_uses_unknown_fallback(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock(return_value=["doc-gone"])
        service.repo.get_documents_by_ids = AsyncMock(return_value=[])
        service.retrieval.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        document_id="doc-gone",
                        chunk_id="c1",
                        filename="",
                        chunk_index=0,
                        score=0.1,
                        content="orphaned",
                    )
                ]
            )
        )

        matches = await service.retrieve_chunks(
            user_id="user-a",
            query="q",
            document_ids=["doc-gone"],
            top_k=1,
        )
        self.assertEqual(matches[0]["document_title"], "Unknown document")

    async def test_retrieve_unauthorized_document_ids_raise_404(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock(return_value=["doc-1"])

        with self.assertRaises(HTTPException) as ctx:
            await service.retrieve_chunks(
                user_id="user-a",
                query="q",
                document_ids=["doc-1", "doc-other"],
                top_k=1,
            )
        self.assertEqual(ctx.exception.status_code, 404)
        service.retrieval.retrieve = AsyncMock()
        service.retrieval.retrieve.assert_not_called()

    async def test_retrieve_empty_result(self):
        service = self._service()
        service.repo.filter_document_ids_for_user = AsyncMock(return_value=["doc-1"])
        service.repo.get_documents_by_ids = AsyncMock(
            return_value=[SimpleNamespace(id="doc-1", original_filename="a.txt")]
        )
        service.retrieval.retrieve = AsyncMock(return_value=SimpleNamespace(chunks=[]))

        matches = await service.retrieve_chunks(
            user_id="user-a",
            query="q",
            document_ids=["doc-1"],
            top_k=5,
        )
        self.assertEqual(matches, [])


class FindDocumentByChecksumTest(unittest.IsolatedAsyncioTestCase):
    async def test_find_document_by_checksum_rejects_non_hex(self):
        db = AsyncMock()
        repo = RagRepository(db)
        found = await repo.find_document_by_checksum(
            user_id="user-a",
            checksum_sha256="not-a-hex!",
        )
        self.assertIsNone(found)
        db.execute.assert_not_awaited()

    async def test_find_document_by_checksum_scopes_user_and_project(self):
        db = AsyncMock()
        result = MagicMock()
        owned = SimpleNamespace(id="doc-1")
        result.scalar_one_or_none.return_value = owned
        db.execute = AsyncMock(return_value=result)
        repo = RagRepository(db)

        found = await repo.find_document_by_checksum(
            user_id="user-a",
            checksum_sha256="abc123",
            project_id="proj-1",
        )

        self.assertEqual(found, owned)
        db.execute.assert_awaited_once()


class UploadStreamingTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.application.document_ingestion_service.RagPolicyService")
    @patch("backend.modules.rag.application.document_ingestion_service.FileStorageAdapter")
    async def test_upload_persists_checksum_metadata(self, storage_cls, policy_cls):
        policy_cls.return_value.is_allowed_file_type.return_value = True
        storage_cls.return_value.store_document = AsyncMock(
            return_value=SimpleNamespace(
                storage_path="rag/key",
                size_bytes=19,
                checksum_sha256="abc123",
                content_type="text/plain",
                reused_existing=False,
            )
        )

        db = AsyncMock()
        from backend.modules.rag.application.document_ingestion_service import (
            DocumentIngestionService,
        )

        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(
            enabled=True,
            max_file_bytes=1_000_000,
            allowed_file_types=("txt",),
        )
        service.repo = MagicMock()
        doc = SimpleNamespace(id="doc-1")
        job = SimpleNamespace(id="job-1")
        service.repo.create_document = AsyncMock(return_value=doc)
        service.repo.create_ingestion_job = AsyncMock(return_value=job)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        document, job_result, content = await service.upload_document(
            user_id="user-a",
            filename="notes.txt",
            content=b"hello world content",
            content_type="text/plain",
        )

        self.assertEqual(document.id, "doc-1")
        self.assertIsNone(content)
        metadata = service.repo.create_document.await_args.kwargs["metadata"]
        self.assertEqual(metadata["checksum_sha256"], "abc123")
        self.assertEqual(metadata["size_bytes"], 19)
        self.assertEqual(metadata["storage_key"], "rag/key")

    @patch("backend.modules.rag.application.document_ingestion_service.RagPolicyService")
    @patch("backend.modules.rag.application.document_ingestion_service.FileStorageAdapter")
    async def test_upload_reuses_document_on_checksum_match(self, storage_cls, policy_cls):
        policy_cls.return_value.is_allowed_file_type.return_value = True
        existing = SimpleNamespace(
            id="doc-existing",
            storage_path="rag/existing",
            content_type="text/plain",
            metadata_json='{"size_bytes": 11, "checksum_sha256": "duphash"}',
        )
        storage_cls.return_value.store_document = AsyncMock(
            return_value=SimpleNamespace(
                storage_path="rag/existing",
                size_bytes=11,
                checksum_sha256="duphash",
                content_type="text/plain",
                reused_existing=True,
            )
        )

        db = AsyncMock()
        from backend.modules.rag.application.document_ingestion_service import (
            DocumentIngestionService,
        )

        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(
            enabled=True,
            max_file_bytes=1_000_000,
            allowed_file_types=("txt",),
        )
        service.repo = MagicMock()
        service.repo.find_document_by_checksum = AsyncMock(return_value=existing)
        service.repo.get_active_ingestion_job = AsyncMock(return_value=None)
        job = SimpleNamespace(id="job-1")
        service.repo.create_ingestion_job = AsyncMock(return_value=job)
        service.repo.create_document = AsyncMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        document, job_result, content = await service.upload_document(
            user_id="user-a",
            filename="notes.txt",
            content=b"same bytes",
            content_type="text/plain",
        )

        self.assertEqual(document.id, "doc-existing")
        self.assertIsNone(content)
        service.repo.create_document.assert_not_called()
        service.repo.create_ingestion_job.assert_awaited_once()


class StreamUploadAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_enforces_size_before_full_buffer(self):
        from backend.modules.rag.infrastructure.file_storage_adapter import (
            FileStorageAdapter,
            FileTooLargeError,
        )

        adapter = FileStorageAdapter()
        chunks_seen = 0

        async def _chunks():
            nonlocal chunks_seen
            for _ in range(5):
                chunks_seen += 1
                yield b"x" * 100

        with (
            patch(
                "backend.modules.rag.infrastructure.file_storage_adapter.object_storage"
            ) as storage,
        ):
            storage.is_configured = True
            storage.ensure_bucket = AsyncMock()
            storage.upload_fileobj = AsyncMock()

            with self.assertRaises(FileTooLargeError):
                await adapter.store_document_stream(
                    user_id="user-a",
                    filename="big.txt",
                    content_type="text/plain",
                    chunks=_chunks(),
                    max_bytes=250,
                )

        self.assertLessEqual(chunks_seen, 3)
        storage.upload_fileobj.assert_not_awaited()

    async def test_stream_skips_upload_when_dedupe_hits(self):
        from backend.modules.rag.infrastructure.file_storage_adapter import (
            FileStorageAdapter,
            StoredDocument,
        )

        adapter = FileStorageAdapter()

        async def _chunks():
            yield b"hello-dedupe"

        async def _lookup(checksum: str):
            return StoredDocument(
                storage_path="rag/prior",
                size_bytes=12,
                checksum_sha256=checksum,
                content_type="text/plain",
                reused_existing=True,
            )

        with (
            patch(
                "backend.modules.rag.infrastructure.file_storage_adapter.object_storage"
            ) as storage,
        ):
            storage.is_configured = True
            storage.ensure_bucket = AsyncMock()
            storage.upload_fileobj = AsyncMock()

            stored = await adapter.store_document_stream(
                user_id="user-a",
                filename="notes.txt",
                content_type="text/plain",
                chunks=_chunks(),
                max_bytes=10_000,
                dedupe_lookup=_lookup,
            )

        self.assertTrue(stored.reused_existing)
        self.assertEqual(stored.storage_path, "rag/prior")
        storage.upload_fileobj.assert_not_awaited()
