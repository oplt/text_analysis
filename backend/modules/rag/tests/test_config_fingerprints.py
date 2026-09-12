from __future__ import annotations

import unittest
from dataclasses import replace

from backend.modules.rag.infrastructure.rag_config import RagConfig


class ConfigFingerprintTests(unittest.TestCase):
    def test_component_fingerprints_are_stable(self):
        config = RagConfig.from_settings()
        self.assertEqual(config.embedding.fingerprint(), config.embedding.fingerprint())
        self.assertEqual(config.retrieval.fingerprint(), config.retrieval.fingerprint())
        self.assertEqual(config.parsing.fingerprint(), config.parsing.fingerprint())
        self.assertEqual(config.reranking.fingerprint(), config.reranking.fingerprint())
        self.assertEqual(config.generation.fingerprint(), config.generation.fingerprint())
        self.assertEqual(config.synthesis.fingerprint(), config.synthesis.fingerprint())
        self.assertEqual(config.evaluation.fingerprint(), config.evaluation.fingerprint())

    def test_output_affecting_change_changes_only_its_component_identity(self):
        config = RagConfig.from_settings()
        changed = replace(config, chunk_size=config.chunk_size + 1)
        self.assertNotEqual(config.chunking.fingerprint(), changed.chunking.fingerprint())
        self.assertEqual(config.embedding.fingerprint(), changed.embedding.fingerprint())

    def test_parsing_and_reranking_knobs_change_fingerprints(self):
        config = RagConfig.from_settings()
        parsing_changed = replace(config, pdf_ocr_enabled=not config.pdf_ocr_enabled)
        self.assertNotEqual(config.parsing.fingerprint(), parsing_changed.parsing.fingerprint())
        rerank_changed = replace(config, rerank_max_depth=config.rerank_max_depth + 1)
        self.assertNotEqual(config.reranking.fingerprint(), rerank_changed.reranking.fingerprint())
        generation_changed = replace(config, context_ordering_policy="grouped_by_document")
        self.assertNotEqual(
            config.generation.fingerprint(), generation_changed.generation.fingerprint()
        )
        synthesis_changed = replace(
            config, synthesis_batch_size=config.synthesis_batch_size + 1
        )
        self.assertNotEqual(
            config.synthesis.fingerprint(), synthesis_changed.synthesis.fingerprint()
        )

    def test_operational_cache_version_does_not_change_evidence_identity(self):
        config = RagConfig.from_settings()
        changed = replace(config, retrieval_cache_artifact_version="v3")
        self.assertEqual(config.embedding.fingerprint(), changed.embedding.fingerprint())
        self.assertEqual(config.chunking.fingerprint(), changed.chunking.fingerprint())
