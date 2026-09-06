from __future__ import annotations

import logging

from backend.modules.rag.domain.models import ParsedDocument
from backend.modules.rag.infrastructure.langchain_document_loaders import load_documents_from_bytes

logger = logging.getLogger(__name__)


class DocumentParserService:
    async def parse_bytes(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        metadata: dict | None = None,
    ) -> list[ParsedDocument]:
        docs = await load_documents_from_bytes(
            content=content,
            filename=filename,
            content_type=content_type,
        )
        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)
        return docs
