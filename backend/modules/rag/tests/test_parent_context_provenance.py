"""Parent context must not replace child citation provenance."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.rag.application.chunking_service import _attach_parent_windows
from backend.modules.rag.application.citation_validation_service import (
    CitationValidationService,
)
from backend.modules.rag.application.parent_context_service import expand_parent_chunks
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.domain.models import DocumentChunk, RetrievedChunk


class ParentContextProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_parent_is_deduplicated_and_child_citations_are_exact(self):
        children = [
            RetrievedChunk(
                chunk_id="child-a",
                document_id="doc-1",
                content="exact child A",
                score=0.9,
                filename="report.pdf",
                chunk_index=3,
                page_number=1,
                metadata={"char_start": 10, "char_end": 23},
            ),
            RetrievedChunk(
                chunk_id="child-b",
                document_id="doc-1",
                content="exact child B",
                score=0.8,
                filename="report.pdf",
                chunk_index=4,
                page_number=2,
                metadata={"char_start": 100, "char_end": 113},
            ),
        ]
        repo = MagicMock()
        repo.get_chunks_by_ids = AsyncMock(
            side_effect=[
                [
                    SimpleNamespace(
                        id="child-a", parent_chunk_id="parent-1", document_id="doc-1"
                    ),
                    SimpleNamespace(
                        id="child-b", parent_chunk_id="parent-1", document_id="doc-1"
                    ),
                ],
                [
                    SimpleNamespace(
                        id="parent-1",
                        document_id="doc-1",
                        content="parent page one\n\nparent page two",
                        chunk_index=-1,
                    )
                ],
            ]
        )

        expanded = await expand_parent_chunks(children, repo=repo, document_ids=["doc-1"])
        context = RagContextBuilder().build_document_context_block(expanded)
        validated = CitationValidationService().validate(
            raw_output='{"answer":"exact child A","claims":[{"text":"exact child A",'
            '"chunk_ids":["child-a"]}]}',
            retrieved_chunks=expanded,
            allowed_document_ids=["doc-1"],
        )

        self.assertEqual([item.context_content for item in expanded], [
            "parent page one\n\nparent page two",
            None,
        ])
        self.assertIn("parent page one", context)
        self.assertNotIn("\ncontent: exact child A\n", context)
        self.assertIn("citation_content: exact child A", context)
        self.assertEqual(validated.citations[0].snippet, "exact child A")
        self.assertEqual(validated.citations[0].chunk_id, "child-a")
        self.assertEqual(validated.citations[0].parent_context_id, "parent-1")
        self.assertEqual(validated.citations[0].char_start, 10)

    async def test_parent_outside_allow_list_cannot_expand_child(self):
        child = RetrievedChunk(
            chunk_id="child-a",
            document_id="doc-allowed",
            content="child evidence",
            score=0.9,
            filename="report.pdf",
            chunk_index=1,
        )
        repo = MagicMock()
        repo.get_chunks_by_ids = AsyncMock(
            side_effect=[
                [
                    SimpleNamespace(
                        id="child-a", parent_chunk_id="parent-1", document_id="doc-allowed"
                    )
                ],
                [
                    SimpleNamespace(
                        id="parent-1",
                        document_id="doc-forbidden",
                        content="forbidden parent",
                        chunk_index=-1,
                    )
                ],
            ]
        )

        expanded = await expand_parent_chunks(
            [child], repo=repo, document_ids=["doc-allowed"]
        )

        self.assertEqual(expanded[0].content, "child evidence")
        self.assertIsNone(expanded[0].context_content)

    def test_parent_windows_are_non_overlapping_and_keep_cross_page_offsets(self):
        leaves = [
            DocumentChunk(
                id="child-1",
                document_id="doc-1",
                user_id="user-a",
                chunk_index=0,
                content="paragraph one\n\nparagraph two",
                token_count=5,
                metadata={
                    "source_paragraph_indexes": [0, 1],
                    "source_span_ids": ["span-1", "span-2"],
                    "char_start": 0,
                    "char_end": 29,
                    "page_number": 1,
                },
            ),
            DocumentChunk(
                id="child-2",
                document_id="doc-1",
                user_id="user-a",
                chunk_index=1,
                content="paragraph two\n\nparagraph three",
                token_count=5,
                metadata={
                    "source_paragraph_indexes": [1, 2],
                    "source_span_ids": ["span-2", "span-3"],
                    "char_start": 17,
                    "char_end": 47,
                    "page_number": 2,
                },
            ),
        ]

        chunks = _attach_parent_windows(leaves, window=2)

        self.assertEqual(chunks[0].content, "paragraph one\n\nparagraph two\n\nparagraph three")
        self.assertEqual(chunks[0].metadata["char_start"], 0)
        self.assertEqual(chunks[0].metadata["char_end"], 47)
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])
