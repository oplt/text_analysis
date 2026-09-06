from __future__ import annotations

from typing import Any

from fastapi import HTTPException, UploadFile

from backend.core.config import settings
from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.ai.base_service import AiBaseService
from backend.modules.identity_access.models import User
from backend.modules.rag.application.legacy_ai_document_service import LegacyAiDocumentService


class AiDocumentService(AiBaseService):
    """Legacy /ai/documents contract backed exclusively by RAG storage and ingestion."""

    async def list_documents(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        self._require_rag_documents()
        return await LegacyAiDocumentService(self.db).list_documents(
            user.id, limit=limit, offset=offset
        )

    async def get_document(self, user: User, document_id: str):
        self._require_rag_documents()
        return await LegacyAiDocumentService(self.db).get_document(user.id, document_id)

    async def create_document_from_text(
        self,
        user: User,
        *,
        title: str,
        description: str | None,
        content: str,
        content_type: str,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not content.strip():
            raise HTTPException(status_code=422, detail="Document content must not be empty")
        max_bytes = settings.RAG_MAX_FILE_BYTES
        if len(content.encode("utf-8")) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Document exceeds the maximum size of {max_bytes} bytes",
            )
        self._require_rag_documents()
        return await LegacyAiDocumentService(self.db).create_from_text(
            user_id=user.id,
            title=title,
            description=description,
            content=content,
            content_type=content_type,
            filename=filename,
            metadata=metadata,
        )

    async def create_document_from_upload(
        self, user: User, file: UploadFile, description: str | None
    ):
        self._require_rag_documents()
        return await LegacyAiDocumentService(self.db).create_from_upload(
            user_id=user.id,
            file=file,
            description=description,
        )

    async def retrieve_chunks(
        self,
        user: User,
        *,
        query: str,
        document_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not settings.RAG_ENABLED:
            return []
        return await LegacyAiDocumentService(self.db).retrieve_chunks(
            user_id=user.id,
            query=query,
            document_ids=document_ids,
            top_k=top_k,
        )

    @staticmethod
    def _require_rag_documents() -> None:
        if not settings.RAG_ENABLED:
            raise HTTPException(status_code=503, detail="RAG is disabled")
