from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.cache import embedding_cache_key
from backend.modules.rag.application.embedding_service import EmbeddingService
from backend.modules.rag.application.evidence_revision import build_evidence_revision_hash
from backend.modules.rag.infrastructure.rag_config import RagConfig
from backend.modules.rag.infrastructure.repositories import RagRepository


def test_embedding_cache_key_includes_embedding_identity_and_normalized_content():
    base = embedding_cache_key(
        "openai",
        "text-embedding-3-small",
        "cafe\u0301",
        dimensions=1536,
        model_version="2026-01",
        preprocessing_version="unicode-nfc-v1",
    )
    assert base == embedding_cache_key(
        "openai",
        "text-embedding-3-small",
        "caf\u00e9",
        dimensions=1536,
        model_version="2026-01",
        preprocessing_version="unicode-nfc-v1",
    )
    assert base != embedding_cache_key(
        "openai",
        "text-embedding-3-small",
        "caf\u00e9",
        dimensions=768,
        model_version="2026-01",
        preprocessing_version="unicode-nfc-v1",
    )
    assert base != embedding_cache_key(
        "openai",
        "text-embedding-3-small",
        "caf\u00e9",
        dimensions=1536,
        model_version="2026-02",
        preprocessing_version="unicode-nfc-v1",
    )


class EmbeddingRevisionIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_service_rejects_wrong_dimensions_before_persistence(self):
        config = replace(RagConfig.from_settings(), embedding_dimensions=3)
        service = EmbeddingService(config)
        with patch(
            "backend.modules.rag.application.embedding_service.embed_texts_with_cache",
            AsyncMock(return_value=[[0.1, 0.2]]),
        ), self.assertRaisesRegex(ValueError, "unexpected dimension"):
            await service.embed_texts(["evidence"])

    async def test_document_revision_persists_complete_embedding_identity(self):
        db = MagicMock()
        db.flush = AsyncMock()
        repo = RagRepository(db)
        revision = await repo.create_document_revision(
            SimpleNamespace(id="document-1"),
            source_content_hash="source",
            parser_version="parser-v1",
            chunker_version="chunker-v1",
            index_version="index-v1",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_model_version="2026-01",
            embedding_dimensions=1536,
            embedding_preprocessing_version="unicode-nfc-v1",
        )

        self.assertEqual(revision.embedding_provider, "openai")
        self.assertEqual(revision.embedding_model, "text-embedding-3-small")
        self.assertEqual(revision.embedding_model_version, "2026-01")
        self.assertEqual(revision.embedding_dimensions, 1536)
        self.assertEqual(revision.embedding_preprocessing_version, "unicode-nfc-v1")


def test_evidence_hash_changes_when_embedding_identity_changes():
    config = RagConfig.from_settings()
    document_revisions = [{"rag_document_id": "document-1", "chunks": []}]
    baseline = build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=document_revisions,
        config=config,
    )
    changed = build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=document_revisions,
        config=replace(config, embedding_preprocessing_version="unicode-nfkc-v1"),
    )
    assert baseline != changed
