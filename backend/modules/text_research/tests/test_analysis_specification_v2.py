"""Tests for AnalysisSpecification v2 and pipeline compilation."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_quantitative_spec,
    build_spec_from_request,
)
from backend.modules.text_research.domain.analysis_specification import (
    AnalysisSpecification,
    normalize_corpus_filters,
    preprocessing_config_fingerprint,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ENGINE_VERSION,
    compile_plan,
    computation_identity,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class AnalysisSpecificationV2Tests(unittest.TestCase):
    def _classification_spec(self, *, random_seed: int = 42) -> AnalysisSpecification:
        return AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            unit_type="paragraph",
            filters={"language": "en"},
            preprocessing_profile_id="prep-1",
            feature={"type": "tfidf", "ngram_range": (1, 2)},
            model={"family": "logistic_regression"},
            validation={"strategy": "grouped_holdout", "group_field": "source_document"},
            random_seed=random_seed,
        )

    def test_same_logical_spec_same_hash(self) -> None:
        left = self._classification_spec()
        right = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            unit_type="paragraph",
            filters={"language": "en"},
            preprocessing_profile_id="prep-1",
            feature={"type": "tfidf", "ngram_range": (1, 2)},
            model={"family": "logistic_regression"},
            validation={"strategy": "grouped_holdout", "group_field": "source_document"},
            random_seed=42,
        )
        self.assertEqual(left.normalize().spec_hash(), right.normalize().spec_hash())

    def test_filter_key_order_irrelevant_after_normalize(self) -> None:
        left = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="dfm",
            filters={"language": "en", "year": 2020},
        )
        right = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="dfm",
            filters={"year": 2020, "language": "en"},
        )
        self.assertEqual(left.normalize().spec_hash(), right.normalize().spec_hash())

    def test_different_seed_different_hash(self) -> None:
        left = self._classification_spec(random_seed=42)
        right = self._classification_spec(random_seed=43)
        self.assertNotEqual(left.spec_hash(), right.spec_hash())

    def test_from_flat_produces_v2_and_roundtrip(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="classification",
            unit_type="sentence",
            filters={"language": "en"},
            feature={"type": "tfidf", "ngram_range": (1, 2)},
            model={"family": "logistic_regression"},
            validation={"strategy": "grouped_cv", "group_field": "organization"},
            random_seed=7,
        )
        self.assertEqual(spec.spec_version, "2.0")
        params = spec.to_run_parameters()
        self.assertEqual(params["corpus"]["corpus_id"], "c1")
        self.assertEqual(params["unit_type"], "sentence")
        self.assertEqual(params["filters"]["language"], "en")
        self.assertEqual(params["random_seed"], 7)
        self.assertEqual(params["analysis"]["type"], "classification")

    def test_compile_plan_includes_prepare_corpus_for_classification(self) -> None:
        plan = compile_plan(self._classification_spec())
        self.assertIn("prepare_corpus", plan.stages)
        self.assertIn("classification", plan.stages)

    def test_compile_plan_includes_prepare_corpus_for_topic_model(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="topic_model",
            analysis_parameters={"algorithm": "lda", "n_topics": 5},
        )
        plan = compile_plan(spec)
        self.assertIn("prepare_corpus", plan.stages)
        self.assertIn("topic_model", plan.stages)

    def test_build_spec_and_attach_run_identity(self) -> None:
        spec = build_spec_from_request(
            "kwic",
            "corpus-1",
            filters={"language": "en"},
        )
        params = attach_run_identity(spec.to_run_parameters(), spec)
        self.assertEqual(params["analysis_spec_hash"], spec.spec_hash())
        self.assertEqual(params["engine_version"], ENGINE_VERSION)

    def test_computation_identity_is_stable(self) -> None:
        left = computation_identity("spec-hash", "snapshot-hash")
        right = computation_identity("spec-hash", "snapshot-hash")
        self.assertEqual(left, right)
        self.assertNotEqual(left, computation_identity("other-spec", "snapshot-hash"))

    def test_unit_type_changes_quantitative_hash(self) -> None:
        paragraph = build_quantitative_spec(
            "frequencies",
            "corpus-1",
            unit_type="paragraph",
            filters={"language": "en"},
            preprocessing_config=PreprocessingConfig().to_dict(),
            analysis_parameters={"top_n": 50},
        )
        sentence = build_quantitative_spec(
            "frequencies",
            "corpus-1",
            unit_type="sentence",
            filters={"language": "en"},
            preprocessing_config=PreprocessingConfig().to_dict(),
            analysis_parameters={"top_n": 50},
        )
        self.assertNotEqual(paragraph.spec_hash(), sentence.spec_hash())

    def test_organization_language_year_filters_change_hash(self) -> None:
        base = build_quantitative_spec(
            "dfm",
            "corpus-1",
            unit_type="paragraph",
            preprocessing_config=PreprocessingConfig().to_dict(),
        )
        by_org = build_quantitative_spec(
            "dfm",
            "corpus-1",
            unit_type="paragraph",
            filters={"organization": "WHO"},
            preprocessing_config=PreprocessingConfig().to_dict(),
        )
        by_lang = build_quantitative_spec(
            "dfm",
            "corpus-1",
            unit_type="paragraph",
            filters={"language": "en"},
            preprocessing_config=PreprocessingConfig().to_dict(),
        )
        by_year = build_quantitative_spec(
            "dfm",
            "corpus-1",
            unit_type="paragraph",
            filters={"publication_year_min": 2018, "publication_year_max": 2022},
            preprocessing_config=PreprocessingConfig().to_dict(),
        )
        hashes = {
            base.spec_hash(),
            by_org.spec_hash(),
            by_lang.spec_hash(),
            by_year.spec_hash(),
        }
        self.assertEqual(len(hashes), 4)
        # language is part of identity when present (TASK-002 coordination)
        self.assertIn("language", by_lang.normalize().corpus.filters)

    def test_preprocessing_profile_and_config_change_hash(self) -> None:
        default_cfg = PreprocessingConfig().to_dict()
        stemmed_cfg = PreprocessingConfig(stemming=True, remove_stopwords=True).to_dict()
        profile_a = build_quantitative_spec(
            "frequencies",
            "corpus-1",
            preprocessing_profile_id="prep-a",
            preprocessing_config=default_cfg,
        )
        profile_b = build_quantitative_spec(
            "frequencies",
            "corpus-1",
            preprocessing_profile_id="prep-b",
            preprocessing_config=default_cfg,
        )
        config_b = build_quantitative_spec(
            "frequencies",
            "corpus-1",
            preprocessing_profile_id="prep-a",
            preprocessing_config=stemmed_cfg,
        )
        self.assertNotEqual(profile_a.spec_hash(), profile_b.spec_hash())
        self.assertNotEqual(profile_a.spec_hash(), config_b.spec_hash())
        self.assertEqual(
            profile_a.preprocessing.preprocessing_config_hash,
            preprocessing_config_fingerprint(default_cfg),
        )
        self.assertNotEqual(
            profile_a.preprocessing.preprocessing_config_hash,
            config_b.preprocessing.preprocessing_config_hash,
        )

    def test_identical_normalized_quantitative_requests_same_hash(self) -> None:
        left = build_quantitative_spec(
            "ngrams",
            "corpus-1",
            unit_type="paragraph",
            filters={"language": "en", "year_min": "2020", "organization": "UN"},
            preprocessing_profile_id="prep-1",
            preprocessing_config={"lowercase": True, "remove_punctuation": True},
            analysis_parameters={"n": 2, "top_n": 50},
            random_seed=7,
        )
        right = build_quantitative_spec(
            "ngrams",
            "corpus-1",
            unit_type="paragraph",
            filters={"organization": "UN", "publication_year_min": 2020, "language": "en"},
            preprocessing_profile_id="prep-1",
            preprocessing_config={"remove_punctuation": True, "lowercase": True},
            analysis_parameters={"top_n": 50, "n": 2},
            random_seed=7,
        )
        self.assertEqual(left.normalize().spec_hash(), right.normalize().spec_hash())
        self.assertEqual(
            normalize_corpus_filters({"year_min": "2020"}),
            {"publication_year_min": 2020},
        )

    def test_attach_run_identity_persists_normalized_spec_json(self) -> None:
        spec = build_quantitative_spec(
            "cooccurrence",
            "corpus-1",
            unit_type="sentence",
            filters={"language": "de", "organization": "OECD"},
            preprocessing_config=PreprocessingConfig(lowercase=False).to_dict(),
            analysis_parameters={"window_size": 5, "top_n": 20},
        )
        params = attach_run_identity(
            {"corpus_checksum": "corp", "pipeline_checksum": "pipe"},
            spec,
            preprocessing_config=PreprocessingConfig(lowercase=False).to_dict(),
        )
        self.assertEqual(params["analysis_spec_hash"], spec.spec_hash())
        persisted = params["analysis_specification"]
        self.assertEqual(persisted["corpus"]["unit_type"], "sentence")
        self.assertEqual(persisted["corpus"]["filters"]["language"], "de")
        self.assertEqual(persisted["corpus"]["filters"]["organization"], "OECD")
        self.assertIsNotNone(persisted["preprocessing"]["preprocessing_config_hash"])
        self.assertEqual(
            list(persisted["corpus"]["filters"]),
            sorted(persisted["corpus"]["filters"]),
        )


if __name__ == "__main__":
    unittest.main()
