"""Tests for prepared corpus domain types, builder, and artifact registry."""

from __future__ import annotations

import unittest

from backend.modules.text_research.domain.prepared_corpus import (
    compute_corpus_checksum,
    compute_pipeline_checksum,
)
from backend.modules.text_research.infrastructure import artifact_registry
from backend.modules.text_research.infrastructure.artifact_registry import ArtifactRegistry
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
    tokenize,
)


class CorpusChecksumTests(unittest.TestCase):
    def test_stable_for_same_input(self):
        unit_ids = ["u2", "u1", "u3"]
        texts = ["beta text", "alpha text", "gamma text"]
        first = compute_corpus_checksum(unit_ids, texts)
        second = compute_corpus_checksum(list(reversed(unit_ids)), list(reversed(texts)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_changes_when_text_changes(self):
        unit_ids = ["u1", "u2"]
        original = compute_corpus_checksum(unit_ids, ["hello", "world"])
        changed = compute_corpus_checksum(unit_ids, ["hello", "world!"])
        self.assertNotEqual(original, changed)


class PipelineChecksumTests(unittest.TestCase):
    def test_different_preprocessing_yields_different_checksum(self):
        base_cfg = PreprocessingConfig().to_dict()
        lower_off = PreprocessingConfig(lowercase=False).to_dict()
        impl = describe_implementation(base_cfg)

        base = compute_pipeline_checksum(base_cfg, impl)
        changed = compute_pipeline_checksum(lower_off, describe_implementation(lower_off))
        self.assertNotEqual(base, changed)


class PrepareTextsTests(unittest.TestCase):
    def test_tokens_match_tokenize(self):
        texts = ["The Policy, IS Universal!", "Education matters."]
        config = {"remove_stopwords": False, "lowercase": True}
        artifact = prepare_texts(texts, config)

        expected = [tokenize(text, config) for text in texts]
        self.assertEqual([list(seq) for seq in artifact.token_sequences], expected)
        self.assertEqual(list(artifact.texts_joined), [" ".join(seq) for seq in expected])

    def test_original_text_never_mutated(self):
        texts = ["The Policy, IS Universal!", "Education matters."]
        originals = list(texts)
        prepare_texts(texts, PreprocessingConfig())
        self.assertEqual(texts, originals)


class ArtifactRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ArtifactRegistry()

    def test_idempotent_on_same_checksum(self):
        checksum = "abc123"
        first_id = self.registry.register(
            "prepared_corpus",
            "artifact-a",
            {"unit_count": 2},
            checksum,
        )
        second_id = self.registry.register(
            "prepared_corpus",
            "artifact-b",
            {"unit_count": 99},
            checksum,
        )
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(self.registry.list_artifacts("prepared_corpus")), 1)

    def test_find_by_checksum(self):
        checksum = compute_corpus_checksum(["u1"], ["hello"])
        artifact_id = self.registry.register(
            "prepared_corpus",
            "pc-1",
            {"unit_count": 1},
            checksum,
        )
        record = self.registry.find_by_checksum("prepared_corpus", checksum)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.artifact_id, artifact_id)
        self.assertEqual(self.registry.get(artifact_id), record)


class ModuleLevelRegistryTests(unittest.TestCase):
    def test_module_register_is_idempotent(self):
        checksum = "module-level-checksum"
        first = artifact_registry.register("dfm", "dfm-1", {"mode": "count"}, checksum)
        second = artifact_registry.register("dfm", "dfm-2", {"mode": "tfidf"}, checksum)
        self.assertEqual(first, second)
