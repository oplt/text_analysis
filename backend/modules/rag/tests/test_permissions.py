import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService


class PermissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_unauthorized_project_member_cannot_upload(self):
        db = AsyncMock()
        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(
            enabled=True,
            max_file_bytes=1_000_000,
            allowed_file_types=("txt",),
        )
        service.policy = MagicMock()
        service.policy.is_allowed_file_type.return_value = True
        service.project_access = MagicMock()
        service.project_access.ensure_project_access = AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Forbidden")
        )
        service.storage = MagicMock()

        with self.assertRaises(HTTPException) as ctx:
            await service.upload_document(
                user_id="user-a",
                filename="x.txt",
                content=b"data",
                content_type="text/plain",
                project_id="proj-forbidden",
            )
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_user_cannot_delete_other_users_document(self):
        db = AsyncMock()
        service = DocumentIngestionService(db)
        service.config = SimpleNamespace(enabled=True)
        service.repo = MagicMock()
        service.repo.get_document = AsyncMock(
            return_value=SimpleNamespace(
                id="doc-b",
                user_id="user-b",
                storage_path=None,
            )
        )
        service.vector_store = MagicMock()
        service.storage = MagicMock()

        with self.assertRaises(HTTPException) as ctx:
            await service.delete_document(
                document_id="doc-b",
                user_id="user-a",
                is_admin=False,
            )
        self.assertEqual(ctx.exception.status_code, 403)


class RagRepositoryDocumentAccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_filter_document_ids_for_user_returns_only_owned_ids(self):
        from backend.modules.rag.infrastructure.repositories import RagRepository

        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = ["doc-1"]
        db.execute = AsyncMock(return_value=result)
        repo = RagRepository(db)

        allowed = await repo.filter_document_ids_for_user(
            "user-a",
            ["doc-1", "doc-2"],
        )

        self.assertEqual(allowed, ["doc-1"])
        db.execute.assert_awaited_once()
