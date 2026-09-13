from __future__ import annotations

import unittest
from dataclasses import replace

from backend.lib.vectors import estimate_tokens
from backend.modules.rag.application.context_selection import (
    ContextOrderingPolicy,
    order_context_chunks,
    select_context_chunks,
)
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.domain.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    document_id: str,
    score: float,
    spans: list[str],
    content_hash: str,
    *,
    page_number: int | None = None,
    metadata: dict | None = None,
):
    meta = {"source_span_ids": spans, "content_hash": content_hash}
    if metadata:
        meta.update(metadata)
    return RetrievedChunk(
        chunk_id,
        document_id,
        "content",
        score,
        "doc.txt",
        0,
        metadata=meta,
        index_revision_id="revision",
        page_number=page_number,
    )


class ContextDeduplicationTests(unittest.TestCase):
    def test_overlapping_chunks_keep_highest_ranked_anchor(self):
        first = _chunk("first", "doc", 0.9, ["a", "b"], "one")
        second = _chunk("second", "doc", 0.8, ["a", "b"], "two")
        result = select_context_chunks([second, first])
        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["first"])
        self.assertEqual(result.removed_chunk_ids, ["second"])
        self.assertEqual(result.selected_chunk_ids, ["first"])
        self.assertEqual(
            result.chunks[0].metadata["context_selection_removed_chunk_ids"], ["second"]
        )
        self.assertEqual(result.provenance()["ordering_policy"], "relevance")

    def test_cross_document_identical_content_is_not_merged(self):
        chunks = [
            _chunk("one", "doc-1", 0.9, ["a"], "same"),
            _chunk("two", "doc-2", 0.8, ["a"], "same"),
        ]
        self.assertEqual(len(select_context_chunks(chunks).chunks), 2)

    def test_token_budget_is_preserved_after_selection(self):
        chunks = [
            _chunk("one", "doc", 0.9, ["a"], "same"),
            _chunk("two", "doc", 0.8, ["a"], "same"),
        ]
        selected = RagContextBuilder.trim_chunks_to_token_budget(
            chunks, max_tokens=900, reserved_tokens=512
        )
        self.assertEqual(len(selected), 1)

    def test_grouped_by_document_ordering(self):
        chunks = [
            _chunk("a1", "doc-a", 0.95, ["a"], "a1"),
            _chunk("b1", "doc-b", 0.9, ["b"], "b1"),
            _chunk("a2", "doc-a", 0.85, ["c"], "a2"),
        ]
        ordered = order_context_chunks(chunks, policy=ContextOrderingPolicy.GROUPED_BY_DOCUMENT)
        self.assertEqual([chunk.chunk_id for chunk in ordered], ["a1", "a2", "b1"])

    def test_chronology_when_metadata_uses_page_order(self):
        chunks = [
            _chunk("later", "doc", 0.9, ["a"], "later", page_number=3),
            _chunk("earlier", "doc", 0.8, ["b"], "earlier", page_number=1),
        ]
        ordered = order_context_chunks(
            chunks, policy=ContextOrderingPolicy.CHRONOLOGY_WHEN_METADATA
        )
        self.assertEqual([chunk.chunk_id for chunk in ordered], ["earlier", "later"])

    def test_select_context_for_generation_records_budget_removals(self):
        chunks = [
            _chunk("one", "doc", 0.9, ["a"], "one"),
            _chunk("two", "doc", 0.8, ["b"], "two"),
        ]
        selection = RagContextBuilder.select_context_for_generation(
            chunks,
            max_tokens=estimate_tokens(
                RagContextBuilder().build_document_context_block(chunks[:1])
            ),
            reserved_tokens=0,
            ordering_policy=ContextOrderingPolicy.RELEVANCE,
        )
        self.assertEqual(selection.selected_chunk_ids, ["one"])
        self.assertIn("two", selection.budget_removed_chunk_ids or [])

    def test_rendered_context_budget_and_citation_authority(self):
        original = replace(
            _chunk("one", "doc", 0.9, ["a"], "one"),
            content="original citation " * 500,
            citation_content="original citation " * 500,
        )
        parent = replace(
            original, context_content="expanded parent " * 1000, parent_context_id="parent"
        )
        builder = RagContextBuilder()
        for chunk in (original, parent):
            with self.subTest(parent=chunk.parent_context_id):
                selection = builder.select_context_for_generation(
                    [chunk], max_tokens=300, reserved_tokens=0
                )
                self.assertEqual(len(selection.chunks), 1)
                included = selection.chunks[0]
                self.assertLessEqual(
                    estimate_tokens(builder.build_document_context_block(selection.chunks)), 300
                )
                self.assertEqual(included.content, original.content)
                self.assertEqual(included.citation_content, original.citation_content)
                self.assertTrue(included.metadata["context_truncated"])
                self.assertEqual(selection.provenance()["context_token_count"], 300)
                self.assertNotIn(
                    "citation_content:", builder.build_document_context_block(selection.chunks)
                )

    def test_zero_and_exact_fit_budgets(self):
        chunk = _chunk("one", "doc", 0.9, ["a"], "one")
        builder = RagContextBuilder()
        exact = estimate_tokens(builder.build_document_context_block([chunk]))
        for budget in (0, 1, exact):
            selection = builder.select_context_for_generation(
                [chunk], max_tokens=budget, reserved_tokens=0
            )
            self.assertLessEqual(selection.context_token_count, budget)
            self.assertEqual(selection.selected_chunk_ids, ["one"] if budget == exact else [])
            if budget != exact:
                self.assertEqual(selection.budget_removed_chunk_ids, ["one"])
