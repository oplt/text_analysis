"""TASK-013: API operators and StageRunner/CLI share deterministic analysis math."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.analysis_operators import (
    list_registered_operators,
    run_clustering_operator,
    run_cooccurrence_operator,
    run_corpus_stats_operator,
    run_dictionary_operator,
    run_dimensionality_reduction_operator,
    run_duplicate_detection_operator,
    run_frequencies_operator,
    run_keyness_operator,
    run_readability_operator,
    run_similarity_operator,
)
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class OperatorParityTests(unittest.TestCase):
    def test_registry_includes_core_operators(self) -> None:
        names = set(list_registered_operators())
        self.assertTrue(
            {
                "frequencies",
                "ngrams",
                "dfm",
                "kwic",
                "readability",
                "corpus_stats",
                "dictionary",
                "keyness",
                "cooccurrence",
                "similarity",
                "duplicate_detection",
                "clustering",
                "dimensionality_reduction",
            }
            <= names
        )

    def test_frequencies_operator_matches_direct_quantitative(self) -> None:
        from backend.modules.text_research.infrastructure import quantitative

        texts = ["alpha beta gamma", "beta gamma delta"]
        prepared = prepare_texts(texts, PreprocessingConfig(), unit_ids=["u1", "u2"])
        via_operator = run_frequencies_operator(prepared, top_n=5, rate_per=1000)
        via_direct = quantitative.term_frequency_report(
            [list(t) for t in prepared.token_sequences],
            top_n=5,
            rate_per=1000,
            unit_ids=list(prepared.unit_ids),
        )
        self.assertEqual(via_operator["frequencies"], via_direct["frequencies"])

    def test_readability_operator_stable(self) -> None:
        texts = ["This is a simple sentence. Another one follows."]
        prepared = prepare_texts(texts, PreprocessingConfig(), unit_ids=["u1"])
        result = run_readability_operator(prepared)
        self.assertIn("corpus", result)
        self.assertEqual(result["n_units"], 1)

    def test_new_operators_match_their_infrastructure_engines(self) -> None:
        from backend.modules.text_research.infrastructure import quantitative, similarity
        from backend.modules.text_research.infrastructure.clustering import (
            build_tfidf_matrix,
            run_clustering,
        )
        from backend.modules.text_research.infrastructure.collocation import collocation_report
        from backend.modules.text_research.infrastructure.dictionary_matcher import (
            match_dictionary,
            parse_dictionary_payload,
        )
        from backend.modules.text_research.infrastructure.dimensionality import reduce_dimensions
        from backend.modules.text_research.infrastructure.duplicate_detection import (
            duplicate_report,
        )
        from backend.modules.text_research.infrastructure.keyness import keyness_report

        texts = ["alpha beta gamma", "alpha beta delta", "gamma delta epsilon"]
        prepared = prepare_texts(texts, PreprocessingConfig(), unit_ids=["u1", "u2", "u3"])
        tokens = [list(item) for item in prepared.token_sequences]

        self.assertEqual(
            run_corpus_stats_operator(prepared),
            quantitative.corpus_statistics(texts, tokens),
        )
        dictionary_spec = parse_dictionary_payload({"terms": ["alpha", "gamma"]})
        self.assertEqual(
            run_dictionary_operator(prepared, dictionary_spec=dictionary_spec),
            match_dictionary(tokens, dictionary_spec, unit_ids=["u1", "u2", "u3"]),
        )
        self.assertEqual(
            run_keyness_operator(prepared, prepared, top_n=3),
            keyness_report(tokens, tokens, top_n=3),
        )
        self.assertEqual(
            run_cooccurrence_operator(prepared, top_n=3),
            collocation_report(tokens, window=5, top_n=3),
        )
        self.assertEqual(
            run_similarity_operator(prepared, top_k=3),
            similarity.pairwise_similarity(["u1", "u2", "u3"], tokenized=tokens, top_k=3),
        )
        self.assertEqual(
            run_duplicate_detection_operator(prepared, methods=["exact"]),
            duplicate_report(
                [{"id": f"u{i + 1}", "text": text} for i, text in enumerate(texts)],
                methods=["exact"],
            ),
        )
        expected_cluster = run_clustering(
            list(prepared.texts_joined),
            ["u1", "u2", "u3"],
            n_clusters=2,
            config=prepared.preprocessing_profile,
            random_seed=42,
        )
        actual_cluster = run_clustering_operator(prepared, n_clusters=2)
        self.assertEqual(
            {key: value for key, value in actual_cluster.items() if key != "tfidf_matrix"},
            {key: value for key, value in expected_cluster.items() if key != "tfidf_matrix"},
        )
        matrix, _ = build_tfidf_matrix(texts, prepared.preprocessing_profile)
        self.assertEqual(
            run_dimensionality_reduction_operator(prepared),
            reduce_dimensions(matrix, ["u1", "u2", "u3"]),
        )


if __name__ == "__main__":
    unittest.main()
