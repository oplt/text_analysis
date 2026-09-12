from __future__ import annotations

import unittest

from backend.modules.rag.application.chunking_service import _attach_parent_windows
from backend.modules.rag.domain.models import DocumentChunk, ParsedDocument
from backend.modules.rag.eval.chunking_experiments import measure_structure_v1


class ParentBoundaryTests(unittest.TestCase):
    def test_structure_v1_experiment_reports_overlap_duplication(self):
        result = measure_structure_v1(
            [ParsedDocument(content="one two three four five\n\none two three four five")],
            chunk_size=4,
            chunk_overlap=2,
        )
        self.assertGreater(result.chunk_count, 1)
        self.assertGreaterEqual(result.duplicate_token_ratio, 0.0)

    def test_parent_windows_do_not_cross_structural_parent(self):
        leaves = [
            DocumentChunk(
                id=f"child-{index}",
                document_id="doc-1",
                user_id="user-1",
                chunk_index=index,
                content=f"evidence {index}",
                token_count=2,
                metadata={"structural_parent_id": boundary},
            )
            for index, boundary in enumerate(["page-1|A|prose", "page-2|B|prose"])
        ]

        parents = [
            chunk
            for chunk in _attach_parent_windows(leaves, window=3)
            if chunk.metadata.get("chunk_role") == "parent"
        ]
        self.assertEqual(len(parents), 2)
        self.assertEqual(
            {parent.metadata["structural_parent_id"] for parent in parents},
            {"page-1|A|prose", "page-2|B|prose"},
        )
