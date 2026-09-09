"""Unit tests for statistical modeling and measurement triangulation (§50–51)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure.measurement_validation import (
    compare_measurements,
)
from backend.modules.text_research.infrastructure.statistical_modeling import (
    fit_ols,
    fit_statistical_model,
)


class StatisticalModelingTests(unittest.TestCase):
    def test_ols_recovers_known_line(self) -> None:
        rows = [{"x": float(i), "y": 2.0 * i + 1.0} for i in range(10)]
        result = fit_ols(rows, dependent_var="y", independent_vars=["x"])
        by_term = {row["term"]: row for row in result["coefficients"]}
        self.assertAlmostEqual(by_term["Intercept"]["coefficient"], 1.0, places=6)
        self.assertAlmostEqual(by_term["x"]["coefficient"], 2.0, places=6)
        self.assertGreater(result["r_squared"], 0.999)
        self.assertFalse(result["causal_claim"])

    def test_logistic_optional(self) -> None:
        rows = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.2, "y": 0.0},
            {"x": 0.4, "y": 1.0},
            {"x": 0.5, "y": 0.0},
            {"x": 0.6, "y": 1.0},
            {"x": 0.8, "y": 1.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.2, "y": 1.0},
            {"x": 1.4, "y": 1.0},
            {"x": 1.6, "y": 1.0},
        ]
        try:
            result = fit_statistical_model(
                rows,
                model="logistic",
                dependent_var="y",
                independent_vars=["x"],
            )
        except ValueError as exc:
            self.assertIn("statsmodels", str(exc))
            return
        self.assertEqual(result["model"], "logistic")
        self.assertEqual(result["n_observations"], 10)


class MeasurementValidationTests(unittest.TestCase):
    def test_categorical_agreement(self) -> None:
        result = compare_measurements(
            "human",
            ["a", "a", "b", "b"],
            "classifier",
            ["a", "b", "b", "b"],
            value_kind="categorical",
        )
        self.assertEqual(result["n_paired"], 4)
        self.assertAlmostEqual(result["agreement_rate"], 0.75)
        self.assertFalse(result["equated_concepts"])
        self.assertEqual(result["confusion"]["labels"], ["a", "b"])

    def test_continuous_correlation(self) -> None:
        result = compare_measurements(
            "dictionary",
            [1.0, 2.0, 3.0, 4.0],
            "topic",
            [1.1, 1.9, 3.2, 3.8],
            value_kind="continuous",
        )
        self.assertIsNotNone(result["correlation"]["r"])
        self.assertGreater(result["correlation"]["r"], 0.9)


class AnalysisSpecificationTests(unittest.TestCase):
    def test_from_flat_roundtrip(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="classification",
            unit_type="sentence",
            filters={"language": "en"},
            feature={"type": "tfidf", "ngram_range": (1, 2)},
            validation={"strategy": "grouped_cv", "group_field": "organization"},
            random_seed=7,
        )
        params = spec.to_run_parameters()
        self.assertEqual(params["corpus"]["corpus_id"], "c1")
        self.assertEqual(params["unit_type"], "sentence")
        self.assertEqual(params["filters"]["language"], "en")
        self.assertEqual(params["random_seed"], 7)
        self.assertEqual(params["analysis"]["type"], "classification")


if __name__ == "__main__":
    unittest.main()
