"""Tests for deterministic indexed-evidence revision identity."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.application.evidence_revision import (
    build_evidence_revision_hash,
    chunk_revision_identity,
)
from backend.modules.rag.domain.enums import DocumentStatus
from backend.modules.text_research.application.corpus_scope_service import CorpusScopeService


def _config(**overrides):
    values = {
        "parser_version": "parser-v1",
        "chunker_version": "chunker-v1",
        "embedding_provider": "local",
        "embedding_model": "model-v1",
        "embedding_dimensions": 1536,
        "retrieval_algorithm_version": "retrieval-v1",
        "index_version": "index-v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _revision(content_hash: str = "content-v1") -> list[dict]:
    return [
        {
            "rag_document_id": "doc-1",
            "content_fingerprint": "file-v1",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "content_hash": content_hash,
                }
            ],
        }
    ]


def test_evidence_revision_hash_is_order_independent():
    first = _revision() + [{"rag_document_id": "doc-2", "content_fingerprint": "file-v2"}]
    second = list(reversed(first))

    assert build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=first,
        config=_config(),
    ) == build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=second,
        config=_config(),
    )


def test_evidence_revision_hash_changes_for_content_and_index_inputs():
    baseline = build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=_revision(),
        config=_config(),
    )
    content_changed = build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=_revision("content-v2"),
        config=_config(),
    )
    index_changed = build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=_revision(),
        config=_config(index_version="index-v2"),
    )

    assert baseline != content_changed
    assert baseline != index_changed


def test_evidence_revision_hash_changes_when_content_hash_is_stale():
    first = SimpleNamespace(
        id="chunk-1", chunk_index=0, content="first", content_hash="same"
    )
    second = SimpleNamespace(
        id="chunk-1", chunk_index=0, content="second", content_hash="same"
    )
    first_revision = {
        "rag_document_id": "doc-1",
        "chunks": [chunk_revision_identity(first)],
    }
    second_revision = {
        "rag_document_id": "doc-1",
        "chunks": [chunk_revision_identity(second)],
    }

    assert build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=[first_revision],
        config=_config(),
    ) != build_evidence_revision_hash(
        corpus_id="corpus-1",
        project_id="project-1",
        document_revisions=[second_revision],
        config=_config(),
    )


class CorpusScopeRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_revision_changes_when_indexed_chunk_changes(self):
        service = CorpusScopeService(MagicMock())
        service.rag_config = _config()
        service.get_corpus_or_404 = AsyncMock(
            return_value=SimpleNamespace(id="corpus-1", project_id="project-1", name="Corpus")
        )
        service.repo.list_documents = AsyncMock(
            return_value=[SimpleNamespace(id="corpus-doc-1", rag_document_id="doc-1")]
        )
        service.rag_repo.get_documents_by_ids = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="doc-1",
                    deleted_at=None,
                    status=DocumentStatus.INDEXED.value,
                    metadata_json='{"checksum_sha256":"file-v1"}',
                )
            ]
        )
        service.rag_repo.list_evidence_revision_chunks = AsyncMock(
            side_effect=[
                [
                    SimpleNamespace(
                        id="chunk-1",
                        document_id="doc-1",
                        chunk_index=0,
                        content="first version",
                        content_hash="chunk-v1",
                    )
                ],
                [
                    SimpleNamespace(
                        id="chunk-1",
                        document_id="doc-1",
                        chunk_index=0,
                        content="second version",
                        content_hash="chunk-v2",
                    )
                ],
            ]
        )

        first = await service.resolve(corpus_id="corpus-1", user_id="user-1")
        second = await service.resolve(corpus_id="corpus-1", user_id="user-1")

        self.assertNotEqual(first.evidence_revision_hash, second.evidence_revision_hash)
