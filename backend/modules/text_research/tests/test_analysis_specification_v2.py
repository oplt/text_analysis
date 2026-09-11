"""Tests for AnalysisSpecification v2 and pipeline compilation."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.domain.analysis_specification import (
    AnalysisSpecification,
    EngineSpec,
)
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ENGINE_VERSION,
    compile_plan,
    computation_identity,
)


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
        self.assertEqual(spec.spec_version, "2.1")
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
        left = computation_identity("spec-hash", "snapshot-hash", engine_version="engine/1")
        right = computation_identity("spec-hash", "snapshot-hash", engine_version="engine/1")
        self.assertEqual(left, right)
        self.assertNotEqual(
            left,
            computation_identity("other-spec", "snapshot-hash", engine_version="engine/1"),
        )
        self.assertNotEqual(
            left,
            computation_identity(
                "spec-hash",
                "snapshot-hash",
                engine_version="engine/1",
                engine_name="r",
            ),
        )

    def test_spec_hash_version_aware_for_legacy_2_0(self) -> None:
        modern = AnalysisSpecification.from_flat(corpus_id="c1", analysis_type="dfm")
        legacy = modern.model_copy(update={"spec_version": "2.0"})
        # 2.0 omits engine from the hash; 2.1 includes it — hashes must differ.
        self.assertNotEqual(modern.normalize().spec_hash(), legacy.normalize().spec_hash())
        left_legacy = legacy.normalize().spec_hash()
        right_legacy = (
            legacy.model_copy(update={"engine": EngineSpec(runtime="r")}).normalize().spec_hash()
        )
        self.assertEqual(left_legacy, right_legacy)

    def test_absent_and_canonical_implementation_hash_identically(self) -> None:
        absent = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="frequencies",
            engine={"runtime": "r"},
        )
        canonical = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="frequencies",
            engine={"runtime": "r", "implementation": "quanteda"},
        )
        self.assertEqual(absent.engine.implementation, "quanteda")
        self.assertEqual(canonical.engine.implementation, "quanteda")
        self.assertEqual(absent.spec_hash(), canonical.spec_hash())

    def test_arbitrary_implementation_rejected(self) -> None:
        with self.assertRaises(Exception) as ctx:
            AnalysisSpecification.from_flat(
                corpus_id="c1",
                analysis_type="frequencies",
                engine={"runtime": "r", "implementation": "foo"},
            )
        self.assertIn("incompatible", str(ctx.exception).lower())

    def test_runtime_change_changes_spec_identity(self) -> None:
        python = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="frequencies",
            engine={"runtime": "python"},
        )
        r = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="frequencies",
            engine={"runtime": "r"},
        )
        self.assertNotEqual(python.spec_hash(), r.spec_hash())
        self.assertEqual(python.engine.implementation, "python")
        self.assertEqual(r.engine.implementation, "quanteda")
        plan = compile_plan(r)
        self.assertEqual(plan.engine_name, "r")
        self.assertEqual(plan.engine_version, RAnalysisEngine.implementation_version)


if __name__ == "__main__":
    unittest.main()
