"""Tests for shared PreparedCorpusArtifact wiring in topic/classifier paths."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.application.topic_model_service import build_metadata_breakdowns
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure import topic_models


class PreparedTopicModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = [
            "market economy trade liberty freedom",
            "market economy trade liberty freedom",
            "equality solidarity cohesion community welfare",
            "equality solidarity cohesion community welfare",
        ]

    def test_prepare_texts_then_train_topic_model_with_prepared(self) -> None:
        config = PreprocessingConfig().to_dict()
        prepared = prepare_texts(self.texts, config)
        result = topic_models.train_topic_model(
            self.texts,
            algorithm="nmf",
            n_topics=2,
            random_seed=42,
            prepared=prepared,
            top_n_terms=5,
        )
        self.assertEqual(result["n_topics"], 2)
        self.assertEqual(len(result["dominant_topics"]), len(self.texts))
        self.assertIn("coherence_npmi", result["diagnostics"])

    def test_transform_topic_model_with_prepared(self) -> None:
        config = PreprocessingConfig().to_dict()
        prepared = prepare_texts(self.texts, config)
        trained = topic_models.train_topic_model(
            self.texts,
            algorithm="nmf",
            n_topics=2,
            random_seed=42,
            prepared=prepared,
            top_n_terms=5,
        )
        holdout = prepare_texts(["market economy trade"], config)
        inferred = topic_models.transform_topic_model(
            trained["vectorizer"],
            trained["model"],
            holdout,
            algorithm="nmf",
        )
        self.assertEqual(len(inferred["dominant_topics"]), 1)
        self.assertEqual(len(inferred["doc_topic_distribution"]), 1)

    def test_lda_holdout_perplexity(self) -> None:
        config = PreprocessingConfig().to_dict()
        prepared = prepare_texts(self.texts, config)
        holdout = list(
            prepare_texts(["market economy trade liberty"], config).texts_joined
        )
        result = topic_models.train_topic_model(
            self.texts,
            algorithm="lda",
            n_topics=2,
            random_seed=42,
            prepared=prepared,
            holdout_texts=holdout,
            top_n_terms=5,
        )
        self.assertIn("holdout_perplexity", result["diagnostics"])


class MetadataBreakdownTests(unittest.TestCase):
    def test_only_requested_group_by_fields(self) -> None:
        units = [
            SimpleNamespace(corpus_document_id="doc-1"),
            SimpleNamespace(corpus_document_id="doc-2"),
            SimpleNamespace(corpus_document_id="doc-1"),
        ]
        documents = {
            "doc-1": SimpleNamespace(
                get_field_value=lambda field: {"organization": "WHO", "region": "EU"}.get(field)
            ),
            "doc-2": SimpleNamespace(
                get_field_value=lambda field: {"organization": "UN", "region": "APAC"}.get(field)
            ),
        }
        dominant = [0, 1, 0]

        breakdowns = build_metadata_breakdowns(
            dominant,
            units,
            documents,
            group_by=["organization"],
        )
        self.assertIn("organization", breakdowns)
        self.assertNotIn("region", breakdowns)
        self.assertEqual(breakdowns["organization"]["WHO"]["0"], 2)
        self.assertEqual(breakdowns["organization"]["UN"]["1"], 1)

    def test_empty_when_group_by_not_provided(self) -> None:
        breakdowns = build_metadata_breakdowns([], [], {}, group_by=None)
        self.assertEqual(breakdowns, {})


class ClassificationChecksumTests(unittest.TestCase):
    def test_prepare_texts_records_pipeline_checksum(self) -> None:
        texts = ["Alpha beta gamma.", "Delta epsilon zeta."]
        prepared = prepare_texts(texts, PreprocessingConfig())
        self.assertTrue(prepared.pipeline_checksum)
        self.assertEqual(len(prepared.pipeline_checksum), 64)
        self.assertEqual(prepared.corpus_checksum, prepared.corpus_checksum)

    def test_attach_run_identity_adds_analysis_spec_hash(self) -> None:
        spec = build_spec_from_request(
            "classification",
            "corpus-1",
            snapshot_id="snap-1",
            model={"family": "logistic_regression", "task_type": "binary"},
            validation={"strategy": "grouped_holdout", "test_size": 0.2},
        )
        params = attach_run_identity({"snapshot_id": "snap-1"}, spec)
        self.assertEqual(params["analysis_spec_hash"], spec.spec_hash())
        self.assertIn("engine_version", params)


if __name__ == "__main__":
    unittest.main()
