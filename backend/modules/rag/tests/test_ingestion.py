import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.rag.application.chunking_service import ChunkingService
from backend.modules.rag.application.document_parser_service import DocumentParserService
from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.rag.domain.models import ParsedDocument


class IngestionParserTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.infrastructure.langchain_document_loaders.asyncio.to_thread")
    async def test_parse_offloads_to_thread(self, to_thread):
        to_thread.return_value = []
        parser = DocumentParserService()
        cases = [
            ("notes.txt", "text/plain", b"User prefers PostgreSQL for project X."),
            ("report.pdf", "application/pdf", b"%PDF-1.4\n"),
            (
                "brief.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                b"PK\x03\x04",
            ),
            ("data.csv", "text/csv", b"col1,col2\na,b"),
        ]
        for filename, content_type, content in cases:
            with self.subTest(filename=filename):
                to_thread.reset_mock()
                await parser.parse_bytes(
                    content=content,
                    filename=filename,
                    content_type=content_type,
                )
                to_thread.assert_awaited_once()
                called_fn = to_thread.await_args.args[0]
                self.assertEqual(called_fn.__name__, "_parse_bytes_sync")

    @patch("backend.modules.rag.infrastructure.langchain_document_loaders.asyncio.to_thread")
    async def test_parse_txt_offloads_to_thread(self, to_thread):
        to_thread.side_effect = lambda fn, *args, **kwargs: fn(*args, **kwargs)
        parser = DocumentParserService()
        docs = await parser.parse_bytes(
            content=b"User prefers PostgreSQL for project X.",
            filename="notes.txt",
            content_type="text/plain",
        )
        to_thread.assert_awaited_once()
        self.assertEqual(len(docs), 1)
        self.assertIn("PostgreSQL", docs[0].content)

    @patch("backend.modules.rag.infrastructure.langchain_document_loaders.asyncio.to_thread")
    async def test_parse_txt_creates_content(self, to_thread):
        to_thread.side_effect = lambda fn, *args, **kwargs: fn(*args, **kwargs)
        parser = DocumentParserService()
        docs = await parser.parse_bytes(
            content=b"User prefers PostgreSQL for project X.",
            filename="notes.txt",
            content_type="text/plain",
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("PostgreSQL", docs[0].content)

    def test_chunking_creates_indexed_chunks(self):
        config = SimpleNamespace(chunk_size=50, chunk_overlap=10)
        chunker = ChunkingService(config)
        docs = [ParsedDocument(content="A" * 120, metadata={})]

        async def run():
            with patch(
                "backend.modules.rag.application.chunking_service.asyncio.to_thread",
                side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
            ) as to_thread:
                chunks = await chunker.chunk(
                    docs,
                    document_id="doc-1",
                    user_id="user-a",
                    filename="test.txt",
                )
                to_thread.assert_called_once()
                called_fn = to_thread.call_args.args[0]
                self.assertEqual(called_fn.__name__, "_split_documents_with_token_counts")
                return chunks

        import asyncio

        chunks = asyncio.run(run())
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["document_id"], "doc-1")
        self.assertEqual(chunks[0].metadata["user_id"], "user-a")


class UploadCreatesDocumentTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.application.document_ingestion_service.RagPolicyService")
    @patch("backend.modules.rag.application.document_ingestion_service.FileStorageAdapter")
    async def test_upload_creates_document_row(self, storage_cls, policy_cls):
        policy_cls.return_value.is_allowed_file_type.return_value = True
        storage_cls.return_value.store_document = AsyncMock(
            return_value=SimpleNamespace(
                storage_path="rag/key",
                size_bytes=19,
                checksum_sha256="deadbeef",
                content_type="text/plain",
                reused_existing=False,
            )
        )

        db = AsyncMock()
        service = __import__(
            "backend.modules.rag.application.document_ingestion_service",
            fromlist=["DocumentIngestionService"],
        ).DocumentIngestionService(db)
        service.config = SimpleNamespace(
            enabled=True,
            max_file_bytes=1_000_000,
            allowed_file_types=("txt",),
        )
        service.repo = MagicMock()
        doc = SimpleNamespace(id="doc-1", status=DocumentStatus.UPLOADED.value)
        job = SimpleNamespace(id="job-1")
        service.repo.create_document = AsyncMock(return_value=doc)
        service.repo.create_ingestion_job = AsyncMock(return_value=job)
        service.projects_repo = MagicMock()
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
        service.repo.create_document.assert_awaited_once()
        metadata = service.repo.create_document.await_args.kwargs["metadata"]
        self.assertEqual(metadata["checksum_sha256"], "deadbeef")


class EmbeddingAdapterTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.infrastructure.langchain_embeddings.AiProviderRegistry")
    async def test_embeddings_created_through_adapter(self, registry_cls):
        registry_cls.return_value.get.return_value.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )
        from backend.modules.rag.application.embedding_service import EmbeddingService

        service = EmbeddingService(
            SimpleNamespace(embedding_provider="local", embedding_model="test")
        )
        vectors = await service.embed_texts(["a", "b"])
        self.assertEqual(len(vectors), 2)


class EnqueueDocumentIndexingTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.application.document_ingestion_service.queue_document_indexing")
    async def test_enqueue_creates_job_and_queues_worker(self, queue_fn):
        db = AsyncMock()
        service = __import__(
            "backend.modules.rag.application.document_ingestion_service",
            fromlist=["DocumentIngestionService"],
        ).DocumentIngestionService(db)
        service.config = SimpleNamespace(enabled=True)
        service.repo = MagicMock()
        document = SimpleNamespace(id="doc-1", user_id="user-a", project_id=None)
        job = SimpleNamespace(id="job-1")
        service._get_document_for_indexing = AsyncMock(return_value=document)
        service.repo.get_active_ingestion_job = AsyncMock(return_value=None)
        service.repo.create_ingestion_job = AsyncMock(return_value=job)
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        nested_transaction = MagicMock()
        nested_transaction.__aenter__ = AsyncMock(return_value=None)
        nested_transaction.__aexit__ = AsyncMock(return_value=None)
        db.begin_nested = MagicMock(return_value=nested_transaction)

        result = await service.enqueue_document_indexing(
            document_id="doc-1",
            user_id="user-a",
        )

        self.assertEqual(result.id, "job-1")
        queue_fn.assert_called_once_with(
            document_id="doc-1",
            user_id="user-a",
            job_id="job-1",
        )
        service.repo.create_ingestion_job.assert_awaited_once()

    @patch("backend.modules.rag.application.document_ingestion_service.queue_document_indexing")
    async def test_enqueue_reuses_active_job(self, queue_fn):
        db = AsyncMock()
        service = __import__(
            "backend.modules.rag.application.document_ingestion_service",
            fromlist=["DocumentIngestionService"],
        ).DocumentIngestionService(db)
        service.repo = MagicMock()
        document = SimpleNamespace(id="doc-1", user_id="user-a", project_id=None)
        active_job = SimpleNamespace(id="job-active")
        service._get_document_for_indexing = AsyncMock(return_value=document)
        service.repo.get_active_ingestion_job = AsyncMock(return_value=active_job)

        result = await service.enqueue_document_indexing(
            document_id="doc-1",
            user_id="user-a",
        )

        self.assertEqual(result.id, "job-active")
        service.repo.create_ingestion_job.assert_not_called()
        queue_fn.assert_not_called()
