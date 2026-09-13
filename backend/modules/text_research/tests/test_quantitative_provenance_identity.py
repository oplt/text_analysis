"""Regression tests for quantitative scientific-identity completeness."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.text_research.application.analysis_executor import build_quantitative_spec
from backend.modules.text_research.application.analysis_identity import (
    QUANTITATIVE_PARAMETER_SCHEMA_VERSION,
    normalize_quantitative_run_parameters,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.enums import AnalysisRunStatus
from backend.modules.text_research.domain.models import dumps
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class QuantitativeIdentityMutationTests(unittest.TestCase):
    _config = PreprocessingConfig().to_dict()

    def _hash(self, operation: str, parameters: dict[str, object]) -> str:
        return build_quantitative_spec(
            operation,
            "corpus-1",
            unit_type="paragraph",
            preprocessing_config=self._config,
            analysis_parameters=parameters,
        ).spec_hash()

    def test_analysis_types_are_distinct(self) -> None:
        self.assertNotEqual(self._hash("corpus_stats", {}), self._hash("frequencies", {}))
        self.assertNotEqual(self._hash("ngrams", {"n": 2}), self._hash("frequencies", {"n": 2}))

    def test_output_changing_parameter_mutations_change_identity(self) -> None:
        cases = (
            ("corpus_stats", {"group_by": None}, {"group_by": "organization"}),
            ("frequencies", {"top_n": 10, "group_by": None}, {"top_n": 10, "group_by": "year"}),
            ("ngrams", {"n": 2, "skip": 0}, {"n": 2, "skip": 1}),
            ("dfm", {"weighting": "bm25", "k1": 1.2}, {"weighting": "bm25", "k1": 1.8}),
            ("dfm", {"weighting": "bm25", "b": 0.5}, {"weighting": "bm25", "b": 0.8}),
            (
                "dfm",
                {"weighting": "tfidf", "smooth_idf": True},
                {"weighting": "tfidf", "smooth_idf": False},
            ),
            (
                "dfm",
                {"weighting": "count", "force_sparse_only": False},
                {"weighting": "count", "force_sparse_only": True},
            ),
            (
                "kwic",
                {"keyword": "health", "window_size": 5},
                {"keyword": "health", "window_size": 6},
            ),
            (
                "kwic",
                {"keyword": "health", "case_sensitive": False},
                {"keyword": "health", "case_sensitive": True},
            ),
            (
                "kwic",
                {"keyword": "health", "query_mode": "word"},
                {"keyword": "health", "query_mode": "lemma"},
            ),
            (
                "kwic",
                {"keyword": "health", "query_language": "en"},
                {"keyword": "health", "query_language": "tr"},
            ),
            (
                "kwic",
                {"keyword": "health", "token_attribute": "surface"},
                {"keyword": "health", "token_attribute": "lemma"},
            ),
            (
                "kwic",
                {"keyword": "health", "max_matches": 10},
                {"keyword": "health", "max_matches": 11},
            ),
            (
                "dictionary",
                {"dictionary_content_checksum": "a"},
                {"dictionary_content_checksum": "b"},
            ),
            (
                "dictionary",
                {"hierarchy": {"a": ["x"]}},
                {"hierarchy": {"a": ["y"]}},
            ),
            (
                "cooccurrence",
                {"window_size": 5, "directional": False},
                {"window_size": 5, "directional": True},
            ),
            (
                "cooccurrence",
                {"association_method": "pmi", "min_count": 1},
                {"association_method": "npmi", "min_count": 1},
            ),
            (
                "similarity",
                {
                    "method": "tfidf_cosine",
                    "query_text": "one",
                    "centroid_target": "between_groups",
                    "top_k": 10,
                },
                {
                    "method": "tfidf_cosine",
                    "query_text": "two",
                    "centroid_target": "item_to_own_group",
                    "top_k": 11,
                },
            ),
            (
                "similarity",
                {"method": "tfidf_cosine", "min_score": 0.1},
                {"method": "tfidf_cosine", "min_score": 0.2},
            ),
            (
                "clustering",
                {"n_clusters": 5, "random_seed": 1},
                {"n_clusters": 6, "random_seed": 1},
            ),
            (
                "clustering",
                {"algorithm": "kmeans", "use_svd": False, "n_svd_components": 50},
                {"algorithm": "kmeans", "use_svd": True, "n_svd_components": 50},
            ),
            (
                "dimensionality_reduction",
                {"method": "svd", "n_components": 2},
                {"method": "svd", "n_components": 3},
            ),
            (
                "duplicate_detection",
                {"lexical_threshold": 0.85, "use_minhash": False},
                {"lexical_threshold": 0.9, "use_minhash": False},
            ),
            (
                "duplicate_detection",
                {"use_minhash": True, "minhash_num_perm": 64},
                {"use_minhash": True, "minhash_num_perm": 128},
            ),
            (
                "keyness",
                {"method": "log_likelihood", "min_frequency": 1},
                {"method": "log_likelihood", "min_frequency": 2},
            ),
            (
                "keyness",
                {"filters_a": {"language": "en"}, "filters_b": {"language": "tr"}},
                {"filters_a": {"language": "de"}, "filters_b": {"language": "tr"}},
            ),
        )
        for operation, initial, mutated in cases:
            with self.subTest(operation=operation, initial=initial, mutated=mutated):
                self.assertNotEqual(self._hash(operation, initial), self._hash(operation, mutated))

    def test_normalize_drops_non_scientific_controls(self) -> None:
        from backend.modules.text_research.application.analysis_identity import (
            normalize_analysis_parameters,
        )

        payload = normalize_analysis_parameters(
            "dfm",
            {"weighting": "bm25", "k1": 1.2, "run_async": True, "force_inline": True},
        )
        self.assertNotIn("run_async", payload)
        self.assertNotIn("force_inline", payload)
        self.assertEqual(payload["weighting"], "bm25")
        self.assertEqual(payload["k1"], 1.2)

    def test_scheduling_control_does_not_change_identity(self) -> None:
        # The normalizer deliberately excludes non-scientific request controls.
        initial = self._hash("dfm", {"weighting": "bm25", "k1": 1.2, "run_async": False})
        scheduled = self._hash("dfm", {"weighting": "bm25", "k1": 1.2, "run_async": True})
        self.assertEqual(initial, scheduled)

    def test_topic_holdout_and_embedding_version_change_identity(self) -> None:
        baseline = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="topic_model",
            analysis_parameters={
                "holdout_fraction": 0.1,
                "embedding_identity": {"provider": "hashing", "revision": "one"},
            },
        )
        mutated = baseline.model_copy(
            update={
                "analysis": baseline.analysis.model_copy(
                    update={
                        "parameters": {
                            "holdout_fraction": 0.2,
                            "embedding_identity": {"provider": "hashing", "revision": "two"},
                        }
                    }
                )
            }
        )
        self.assertNotEqual(baseline.spec_hash(), mutated.spec_hash())

    def test_classifier_tuning_controls_change_identity(self) -> None:
        baseline = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            analysis_parameters={
                "val_size": 0.2,
                "n_bootstrap": 100,
                "calibration_method": "sigmoid",
                "use_word_ngrams": True,
                "use_char_ngrams": False,
                "tune_hyperparameters": False,
                "validation_strategy": "holdout",
                "nested_cv_outer_splits": 5,
            },
        )
        mutated = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            analysis_parameters={
                "val_size": 0.3,
                "n_bootstrap": 200,
                "calibration_method": "isotonic",
                "use_word_ngrams": True,
                "use_char_ngrams": True,
                "tune_hyperparameters": True,
                "validation_strategy": "nested_grouped_cv",
                "nested_cv_outer_splits": 3,
            },
        )
        self.assertNotEqual(baseline.spec_hash(), mutated.spec_hash())

    def test_classifier_feature_and_nested_validation_change_identity(self) -> None:
        baseline = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            feature={
                "type": "tfidf",
                "ngram_range": (1, 1),
                "selection": {"method": "none", "k": "all"},
            },
            model={"family": "logistic_regression"},
            validation={"strategy": "grouped_holdout", "test_size": 0.2},
            analysis_parameters={"val_size": 0.2},
        )
        feature_mutated = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            feature={
                "type": "tfidf",
                "ngram_range": (1, 2),
                "selection": {"method": "chi2", "k": 100},
            },
            model={"family": "logistic_regression"},
            validation={"strategy": "grouped_holdout", "test_size": 0.2},
            analysis_parameters={"val_size": 0.2},
        )
        nested = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="classification",
            feature={
                "type": "tfidf",
                "ngram_range": (1, 1),
                "selection": {"method": "none", "k": "all"},
            },
            model={"family": "logistic_regression"},
            validation={
                "strategy": "nested_grouped_cv",
                "outer_splits": 5,
                "inner_splits": 3,
            },
            analysis_parameters={
                "val_size": 0.2,
                "validation_strategy": "nested_grouped_cv",
                "nested_cv_outer_splits": 5,
                "nested_cv_inner_splits": 3,
            },
        )
        self.assertNotEqual(baseline.spec_hash(), feature_mutated.spec_hash())
        self.assertNotEqual(baseline.spec_hash(), nested.spec_hash())

    def test_topic_embedding_model_name_changes_identity(self) -> None:
        baseline = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="topic_model",
            analysis_parameters={
                "holdout_fraction": 0.1,
                "embedding_model_name": "model-a",
                "embedding_identity": {"provider": "hashing", "model_name": "model-a"},
            },
        )
        mutated = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="topic_model",
            analysis_parameters={
                "holdout_fraction": 0.1,
                "embedding_model_name": "model-b",
                "embedding_identity": {"provider": "hashing", "model_name": "model-b"},
            },
        )
        self.assertNotEqual(baseline.spec_hash(), mutated.spec_hash())

    def test_similarity_embedding_identity_changes_scientific_hash(self) -> None:
        baseline = self._hash(
            "similarity",
            {
                "method": "embedding_cosine",
                "embedding_artifact_id": "emb-1",
                "embedding_identity": {
                    "provider": "hashing",
                    "model_name": "sklearn.feature_hasher",
                    "dimension": 256,
                },
            },
        )
        mutated = self._hash(
            "similarity",
            {
                "method": "embedding_cosine",
                "embedding_artifact_id": "emb-1",
                "embedding_identity": {
                    "provider": "hashing",
                    "model_name": "sklearn.feature_hasher",
                    "dimension": 512,
                },
            },
        )
        self.assertNotEqual(baseline, mutated)

    def test_queued_parameters_are_versioned_and_have_schema_defaults(self) -> None:
        params = normalize_quantitative_run_parameters(
            "ngrams", {"unit_type": "paragraph", "filters": {"language": "en"}}
        )
        self.assertEqual(params["parameter_schema_version"], QUANTITATIVE_PARAMETER_SCHEMA_VERSION)
        self.assertEqual(
            {key: params[key] for key in ("n", "top_n", "rate_per", "skip")},
            {"n": 2, "top_n": 50, "rate_per": 1000, "skip": 0},
        )

    def test_duplicate_detection_config_is_sole_default_owner(self) -> None:
        from backend.modules.text_research.domain.quantitative_configs import (
            DuplicateDetectionConfig,
            migrate_quantitative_parameters,
        )

        dumped = DuplicateDetectionConfig.model_validate({}).model_dump()
        migrated = migrate_quantitative_parameters("duplicate_detection", {"unit_type": "document"})
        for key, value in dumped.items():
            self.assertEqual(migrated[key], value)
        self.assertEqual(
            migrated["parameter_schema_version"], QUANTITATIVE_PARAMETER_SCHEMA_VERSION
        )


class QuantitativeWorkerParityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _worker_run(operation: str, parameters: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.RUNNING.value,
            created_by="user-1",
            corpus_id="corpus-1",
            parameters_json=dumps({"_quantitative_operation": operation, **parameters}),
        )

    async def test_cooccurrence_worker_forwards_all_persisted_parameters(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QUANT_OP_KEY,
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.RUNNING.value,
            created_by="user-1",
            corpus_id="corpus-1",
            parameters_json=dumps(
                {
                    QUANT_OP_KEY: "cooccurrence",
                    "unit_type": "paragraph",
                    "filters": {"language": "en"},
                    "window_size": 7,
                    "top_n": 11,
                    "association_method": "npmi",
                    "directional": True,
                    "min_frequency": 2,
                    "min_count": 3,
                    "include_network": False,
                }
            ),
        )
        service.repo.get_run = AsyncMock(return_value=run)
        service.cooccurrence = AsyncMock(return_value=run)

        await service.execute_quantitative("run-1")

        service.cooccurrence.assert_awaited_once_with(
            "corpus-1",
            user_id="user-1",
            unit_type="paragraph",
            preprocessing_profile_id=None,
            window_size=7,
            top_n=11,
            association_method="npmi",
            directional=True,
            min_frequency=2,
            min_count=3,
            include_network=False,
            force_inline=True,
            existing_run_id="run-1",
            language="en",
        )

    async def test_incomplete_cooccurrence_run_uses_canonical_defaults(self) -> None:
        """Legacy/incomplete persisted params must not invent drifted worker defaults."""
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = self._worker_run(
            "cooccurrence",
            {
                "unit_type": "paragraph",
                "window_size": 7,
                "top_n": 11,
                # association_method / directional / min_* / include_network omitted
            },
        )
        service.repo.get_run = AsyncMock(return_value=run)
        service.cooccurrence = AsyncMock(return_value=run)

        await service.execute_quantitative(run.id)

        kwargs = service.cooccurrence.await_args.kwargs
        self.assertEqual(kwargs["association_method"], "pmi")
        self.assertFalse(kwargs["directional"])
        self.assertEqual(kwargs["min_frequency"], 1)
        self.assertEqual(kwargs["min_count"], 1)
        self.assertTrue(kwargs["include_network"])

    async def test_incomplete_duplicate_run_uses_canonical_not_drifted_defaults(
        self,
    ) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )
        from backend.modules.text_research.application.run_adapters import (
            DuplicateDetectionAdapter,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = self._worker_run("duplicate_detection", {"unit_type": "document"})
        service.repo.get_run = AsyncMock(return_value=run)
        service.duplicate_detection = AsyncMock(return_value=run)

        await service.execute_quantitative(run.id)

        kwargs = service.duplicate_detection.await_args.kwargs
        self.assertEqual(kwargs["lexical_threshold"], 0.85)
        self.assertEqual(kwargs["char_ngram_size"], 5)
        self.assertEqual(kwargs["minhash_shingle_size"], 3)
        self.assertEqual(kwargs["max_pairs"], 1000)

        adapter_params = DuplicateDetectionAdapter().normalize({"unit_type": "paragraph"})
        self.assertEqual(adapter_params["lexical_threshold"], 0.85)
        self.assertEqual(adapter_params["char_ngram_size"], 5)
        self.assertEqual(adapter_params["minhash_shingle_size"], 3)
        self.assertEqual(adapter_params["max_pairs"], 1000)

        migrated = normalize_quantitative_run_parameters(
            "duplicate_detection", {"unit_type": "document"}
        )
        self.assertEqual(
            migrated["parameter_schema_version"], QUANTITATIVE_PARAMETER_SCHEMA_VERSION
        )
        self.assertEqual(migrated["max_pairs"], 1000)
        # Explicit None (unlimited) is preserved through Config validation.
        unlimited = normalize_quantitative_run_parameters(
            "duplicate_detection", {"unit_type": "document", "max_pairs": None}
        )
        self.assertIsNone(unlimited["max_pairs"])

    async def test_dfm_worker_forwards_non_default_parameters(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = self._worker_run(
            "dfm",
            {
                "unit_type": "sentence",
                "filters": {"language": "tr"},
                "weighting": "bm25",
                "k1": 1.7,
                "b": 0.3,
                "smooth_idf": False,
                "force_sparse_only": True,
                "trim": {"top_n": 12},
            },
        )
        service.repo.get_run = AsyncMock(return_value=run)
        service.dfm = AsyncMock(return_value=run)

        await service.execute_quantitative(run.id)

        self.assertEqual(
            service.dfm.await_args.kwargs,
            {
                "user_id": "user-1",
                "unit_type": "sentence",
                "weighting": "bm25",
                "k1": 1.7,
                "b": 0.3,
                "smooth_idf": False,
                "preprocessing_profile_id": None,
                "force_sparse_only": True,
                "trim": {"top_n": 12},
                "force_inline": True,
                "existing_run_id": "run-1",
                "language": "tr",
            },
        )

    async def test_duplicate_worker_forwards_non_default_parameters(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = self._worker_run(
            "duplicate_detection",
            {
                "unit_type": "document",
                "methods": ["lexical", "minhash"],
                "lexical_threshold": 0.91,
                "char_ngram_size": 4,
                "use_minhash": True,
                "minhash_num_perm": 32,
                "minhash_shingle_size": 2,
                "minhash_threshold": 0.73,
                "max_pairs": 8,
            },
        )
        service.repo.get_run = AsyncMock(return_value=run)
        service.duplicate_detection = AsyncMock(return_value=run)

        await service.execute_quantitative(run.id)

        kwargs = service.duplicate_detection.await_args.kwargs
        self.assertEqual(kwargs["methods"], ["lexical", "minhash"])
        self.assertEqual(kwargs["lexical_threshold"], 0.91)
        self.assertEqual(kwargs["char_ngram_size"], 4)
        self.assertTrue(kwargs["use_minhash"])
        self.assertEqual(kwargs["minhash_num_perm"], 32)
        self.assertEqual(kwargs["minhash_shingle_size"], 2)
        self.assertEqual(kwargs["minhash_threshold"], 0.73)
        self.assertEqual(kwargs["max_pairs"], 8)

    async def test_clustering_and_dimensionality_workers_forward_non_default_parameters(
        self,
    ) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        clustering_run = self._worker_run(
            "clustering",
            {
                "unit_type": "paragraph",
                "n_clusters": 7,
                "algorithm": "agglomerative",
                "use_svd": True,
                "n_svd_components": 15,
                "top_terms": 9,
                "random_seed": 3,
            },
        )
        service.repo.get_run = AsyncMock(return_value=clustering_run)
        service.clustering = AsyncMock(return_value=clustering_run)
        await service.execute_quantitative(clustering_run.id)
        self.assertEqual(service.clustering.await_args.kwargs["n_clusters"], 7)
        self.assertEqual(service.clustering.await_args.kwargs["algorithm"], "agglomerative")
        self.assertTrue(service.clustering.await_args.kwargs["use_svd"])
        self.assertEqual(service.clustering.await_args.kwargs["n_svd_components"], 15)
        self.assertEqual(service.clustering.await_args.kwargs["top_terms"], 9)
        self.assertEqual(service.clustering.await_args.kwargs["random_seed"], 3)

        dimensionality_run = self._worker_run(
            "dimensionality_reduction",
            {"unit_type": "paragraph", "method": "tsne", "n_components": 3, "random_seed": 9},
        )
        service.repo.get_run = AsyncMock(return_value=dimensionality_run)
        service.dimensionality_reduction = AsyncMock(return_value=dimensionality_run)
        await service.execute_quantitative(dimensionality_run.id)
        self.assertEqual(service.dimensionality_reduction.await_args.kwargs["method"], "tsne")
        self.assertEqual(service.dimensionality_reduction.await_args.kwargs["n_components"], 3)
        self.assertEqual(service.dimensionality_reduction.await_args.kwargs["random_seed"], 9)

    async def test_raw_embeddings_are_never_enqueued(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        corpus = SimpleNamespace(id="corpus-1", project_id="project-1")
        units = [
            SimpleNamespace(id="u1", text="one", corpus_document_id="d1"),
            SimpleNamespace(id="u2", text="two", corpus_document_id="d2"),
        ]
        service._select = AsyncMock(return_value=(corpus, units, []))
        service._enqueue_quantitative = AsyncMock()

        with self.assertRaises(HTTPException) as raised:
            await service.similarity(
                "corpus-1",
                user_id="user-1",
                unit_type="paragraph",
                method="embedding_cosine",
                embeddings={"u1": [1.0, 0.0], "u2": [0.0, 1.0]},
                run_async=True,
            )

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("managed embedding artifact", raised.exception.detail.lower())
        service._enqueue_quantitative.assert_not_awaited()

    async def test_raw_embeddings_blocked_when_workload_would_auto_enqueue(self) -> None:
        from unittest.mock import patch

        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        corpus = SimpleNamespace(id="corpus-1", project_id="project-1")
        units = [
            SimpleNamespace(id="u1", text="one", corpus_document_id="d1"),
            SimpleNamespace(id="u2", text="two", corpus_document_id="d2"),
        ]
        service._select = AsyncMock(return_value=(corpus, units, []))
        service._enqueue_quantitative = AsyncMock()

        with (
            patch(
                "backend.modules.text_research.application.similarity_analysis_service.should_enqueue_cpu_job",
                return_value=True,
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await service.similarity(
                "corpus-1",
                user_id="user-1",
                unit_type="paragraph",
                method="embedding_cosine",
                embeddings={"u1": [1.0, 0.0], "u2": [0.0, 1.0]},
                run_async=False,
            )

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("cannot be queued", raised.exception.detail.lower())
        service._enqueue_quantitative.assert_not_awaited()

    async def test_worker_rejects_legacy_raw_embedding_cosine_run(self) -> None:
        from unittest.mock import patch

        from backend.modules.text_research.application.quantitative_analysis_service import (
            QUANT_OP_KEY,
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        run = SimpleNamespace(
            id="run-legacy",
            status=AnalysisRunStatus.RUNNING.value,
            created_by="user-1",
            corpus_id="corpus-1",
            parameters_json=dumps(
                {
                    QUANT_OP_KEY: "similarity",
                    "unit_type": "paragraph",
                    "method": "embedding_cosine",
                    "mode": "pairwise",
                    "top_k": 10,
                }
            ),
        )
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run_if_active = AsyncMock(return_value=run)
        service.db.commit = AsyncMock()
        service.similarity = AsyncMock()

        with (
            patch(
                "backend.modules.text_research.application.run_lifecycle.fail_if_active",
                new=AsyncMock(return_value=run),
            ),
            self.assertRaises(ValueError) as raised,
        ):
            await service.execute_quantitative(run.id)

        self.assertIn("managed embedding artifact", str(raised.exception).lower())
        service.similarity.assert_not_awaited()

    async def test_managed_embedding_artifact_can_be_queued(self) -> None:
        from backend.modules.text_research.application.quantitative_analysis_service import (
            QuantitativeAnalysisService,
        )

        service = QuantitativeAnalysisService(MagicMock())
        corpus = SimpleNamespace(id="corpus-1", project_id="project-1")
        units = [SimpleNamespace(id="u1", text="one"), SimpleNamespace(id="u2", text="two")]
        service._select = AsyncMock(return_value=(corpus, units, []))
        queued = SimpleNamespace(id="run-1")
        service._enqueue_quantitative = AsyncMock(return_value=queued)

        result = await service.similarity(
            "corpus-1",
            user_id="user-1",
            unit_type="paragraph",
            method="embedding_cosine",
            embedding_artifact_id="embeddings:managed",
            run_async=True,
        )

        self.assertIs(result, queued)
        self.assertEqual(
            service._enqueue_quantitative.await_args.kwargs["parameters"]["embedding_artifact_id"],
            "embeddings:managed",
        )


class ManagedEmbeddingArtifactTests(unittest.TestCase):
    def test_ownership_and_dimension_are_validated(self) -> None:
        from backend.modules.text_research.infrastructure.embeddings import (
            create_managed_embedding_artifact,
            load_managed_embedding_artifact,
        )

        artifact_id = create_managed_embedding_artifact(
            project_id="project-1",
            corpus_id="corpus-1",
            embeddings={"u1": [1.0, 0.0], "u2": [0.0, 1.0]},
            provider="hashing",
            model="test",
        )
        with self.assertRaises(PermissionError):
            load_managed_embedding_artifact(
                artifact_id, project_id="project-2", corpus_id="corpus-1", unit_ids=["u1", "u2"]
            )
        with self.assertRaises(ValueError):
            load_managed_embedding_artifact(
                artifact_id,
                project_id="project-1",
                corpus_id="corpus-1",
                unit_ids=["u1", "u2"],
                expected_dim=3,
            )
