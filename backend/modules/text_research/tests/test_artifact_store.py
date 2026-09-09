"""Typed ArtifactStore tests."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from backend.modules.text_research.infrastructure.artifact_store import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_stores_payload_with_content_addressed_lineage(self):
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
        ):
            store = ArtifactStore()
            descriptor = store.put(
                "manifest",
                {"corpus_checksum": "abc"},
                parent_artifact_ids=["prepared:abc"],
                producing_run_id="run-1",
                implementation_version="engine-2",
                payload_format="json",
            )
            self.assertEqual(store.load(descriptor.artifact_id), {"corpus_checksum": "abc"})

        self.assertEqual(descriptor.parent_artifact_ids, ("prepared:abc",))
        self.assertEqual(descriptor.producing_run_id, "run-1")
        self.assertEqual(store.get(descriptor.artifact_id), descriptor)
