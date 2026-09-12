from __future__ import annotations

import unittest

from backend.modules.rag.application.chunking_service import _attach_parent_windows
from backend.modules.rag.domain.models import DocumentChunk, ParsedDocument
from backend.modules.rag.infrastructure.langchain_text_splitters import split_documents
from backend.modules.rag.infrastructure.pdf_quality import assess_pdf_text_quality


class SourceCoordinateScopeTests(unittest.TestCase):
    def test_page_chunks_carry_explicit_scope_and_resolvable_spans(self):
        pieces = split_documents(
            [
                ParsedDocument(
                    content="Page evidence.",
                    metadata={"page_number": 2},
                    page_number=2,
                )
            ],
            chunk_size=50,
            chunk_overlap=0,
            document_revision="revision-1",
        )

        _content, metadata = pieces[0]
        self.assertEqual(metadata["offset_scope"], "page")
        self.assertEqual(metadata["offset_scope_id"], "page:revision-1:2")
        self.assertEqual(metadata["source_spans"][0]["scope_id"], "page:revision-1:2")
        self.assertEqual(metadata["source_spans"][0]["start"], 0)
        self.assertEqual(metadata["source_spans"][0]["end"], len("Page evidence."))

    def test_parent_windows_do_not_cross_coordinate_scopes(self):
        def leaf(index: int, scope_id: str) -> DocumentChunk:
            return DocumentChunk(
                id=f"child-{index}",
                document_id="doc-1",
                user_id="user-1",
                chunk_index=index,
                content=f"page {index}",
                token_count=2,
                metadata={
                    "offset_scope": "page",
                    "offset_scope_id": scope_id,
                    "source_spans": [],
                },
            )

        chunks = _attach_parent_windows([leaf(0, "page-1"), leaf(1, "page-2")])
        parents = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") == "parent"]
        self.assertEqual(len(parents), 2)
        self.assertEqual(
            {parent.metadata["offset_scope_id"] for parent in parents}, {"page-1", "page-2"}
        )

    def test_pdf_quality_marks_empty_and_replacement_text_as_low_quality(self):
        self.assertTrue(assess_pdf_text_quality("").is_low_quality)
        self.assertTrue(assess_pdf_text_quality("\ufffd" * 100).is_low_quality)
        self.assertFalse(assess_pdf_text_quality("Evidence " * 30).is_low_quality)
