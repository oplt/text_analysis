"""LATEST-015: statistical/measurement inputs as immutable checksummed artifacts."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application.input_dataset_artifacts import (
    build_measurement_input_payload,
    build_statistical_input_payload,
    input_dataset_checksum,
    measurement_input_metadata,
    statistical_input_metadata,
    store_input_dataset,
)
from backend.modules.text_research.application.measurement_validation_service import (
    MeasurementValidationService,
)
from backend.modules.text_research.application.statistical_modeling_service import (
    StatisticalModelingService,
)
from backend.modules.text_research.domain.models import loads


def _corpus() -> SimpleNamespace:
    return SimpleNamespace(id="corpus-1", project_id="project-1")


class InputDatasetArtifactTests(unittest.TestCase):
    def test_identical_statistical_input_has_stable_checksum(self) -> None:
        rows = [{"y": 1.0, "x": 2.0}, {"y": 2.0, "x": 3.0}, {"y": 3.0, "x": 4.0}]
        first = build_statistical_input_payload(
            model="ols",
            dependent_var="y",
            independent_vars=["x"],
            rows=rows,
            add_intercept=True,
        )
        second = build_statistical_input_payload(
            model="ols",
            dependent_var="y",
            independent_vars=["x"],
            rows=list(rows),
            add_intercept=True,
        )
        self.assertEqual(input_dataset_checksum(first), input_dataset_checksum(second))

    def test_changed_observation_changes_statistical_checksum(self) -> None:
        base = build_statistical_input_payload(
            model="ols",
            dependent_var="y",
            independent_vars=["x"],
            rows=[{"y": 1.0, "x": 2.0}, {"y": 2.0, "x": 3.0}, {"y": 3.0, "x": 4.0}],
            add_intercept=True,
        )
        changed = build_statistical_input_payload(
            model="ols",
            dependent_var="y",
            independent_vars=["x"],
            rows=[{"y": 1.0, "x": 2.0}, {"y": 2.0, "x": 3.0}, {"y": 9.0, "x": 4.0}],
            add_intercept=True,
        )
        self.assertNotEqual(input_dataset_checksum(base), input_dataset_checksum(changed))

    def test_identical_measurement_input_has_stable_checksum(self) -> None:
        first = build_measurement_input_payload(
            source_a="human",
            values_a=["yes", "no", "yes"],
            source_b="model",
            values_b=["yes", "no", "no"],
            ids=["u1", "u2", "u3"],
            value_kind="categorical",
            subgroup=None,
        )
        second = build_measurement_input_payload(
            source_a="human",
            values_a=["yes", "no", "yes"],
            source_b="model",
            values_b=["yes", "no", "no"],
            ids=["u1", "u2", "u3"],
            value_kind="categorical",
            subgroup=None,
        )
        self.assertEqual(input_dataset_checksum(first), input_dataset_checksum(second))

    def test_changed_measurement_value_changes_checksum(self) -> None:
        base = build_measurement_input_payload(
            source_a="a",
            values_a=[1.0, 2.0, 3.0],
            source_b="b",
            values_b=[1.1, 2.1, 3.1],
            ids=None,
            value_kind="continuous",
            subgroup=None,
        )
        changed = build_measurement_input_payload(
            source_a="a",
            values_a=[1.0, 2.0, 3.0],
            source_b="b",
            values_b=[1.1, 2.1, 9.9],
            ids=None,
            value_kind="continuous",
            subgroup=None,
        )
        self.assertNotEqual(input_dataset_checksum(base), input_dataset_checksum(changed))

    def test_store_records_schema_and_source_metadata(self) -> None:
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
        ):
            payload = build_statistical_input_payload(
                model="ols",
                dependent_var="y",
                independent_vars=["x"],
                rows=[{"y": 1.0, "x": 2.0}, {"y": 2.0, "x": 3.0}, {"y": 3.0, "x": 5.0}],
                add_intercept=True,
            )
            descriptor = store_input_dataset(payload, metadata=statistical_input_metadata(payload))
            self.assertEqual(descriptor.metadata["kind"], "statistical_model_input")
            self.assertEqual(descriptor.metadata["row_count"], 3)
            self.assertEqual(descriptor.metadata["variable_names"], ["y", "x"])
            self.assertIn("schema", descriptor.metadata)

            m_payload = build_measurement_input_payload(
                source_a="human",
                values_a=["a", "b"],
                source_b="model",
                values_b=["a", "a"],
                ids=None,
                value_kind="categorical",
                subgroup=None,
            )
            m_desc = store_input_dataset(m_payload, metadata=measurement_input_metadata(m_payload))
            self.assertEqual(m_desc.metadata["source_labels"], ["human", "model"])
            self.assertEqual(m_desc.metadata["row_count"], 2)


class StatisticalMeasurementRerunTests(unittest.IsolatedAsyncioTestCase):
    async def test_statistical_rerun_loads_artifact_and_is_deterministic(self) -> None:
        rows = [
            {"y": 1.0, "x": 1.0},
            {"y": 2.0, "x": 2.0},
            {"y": 3.0, "x": 3.0},
            {"y": 3.5, "x": 4.0},
        ]
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
        ):
            payload = build_statistical_input_payload(
                model="ols",
                dependent_var="y",
                independent_vars=["x"],
                rows=rows,
                add_intercept=True,
            )
            artifact = store_input_dataset(payload, metadata=statistical_input_metadata(payload))

            created: list[SimpleNamespace] = []

            async def _create(run):
                created.append(run)
                return run

            service = StatisticalModelingService(MagicMock())
            service.get_corpus_or_404 = AsyncMock(return_value=_corpus())
            service.repo.create_run = AsyncMock(side_effect=_create)

            first = await service.fit_from_artifact(
                "corpus-1", user_id="user-1", input_artifact_id=artifact.artifact_id
            )
            second = await service.fit_from_artifact(
                "corpus-1", user_id="user-1", input_artifact_id=artifact.artifact_id
            )

        params = loads(first.parameters_json, {})
        self.assertEqual(params["input_artifact_id"], artifact.artifact_id)
        self.assertEqual(params["input_artifact_checksum"], artifact.checksum)
        self.assertNotIn("rows", params)
        self.assertEqual(params["n_rows"], 4)

        first_results = loads(first.results_json, {})
        second_results = loads(second.results_json, {})
        self.assertEqual(first_results.get("r_squared"), second_results.get("r_squared"))
        self.assertEqual(
            first_results.get("coefficients"),
            second_results.get("coefficients"),
        )

    async def test_measurement_rerun_loads_artifact_and_is_deterministic(self) -> None:
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
        ):
            payload = build_measurement_input_payload(
                source_a="human",
                values_a=["yes", "no", "yes", "yes"],
                source_b="model",
                values_b=["yes", "no", "no", "yes"],
                ids=["1", "2", "3", "4"],
                value_kind="categorical",
                subgroup=None,
            )
            artifact = store_input_dataset(payload, metadata=measurement_input_metadata(payload))

            async def _create(run):
                return run

            service = MeasurementValidationService(MagicMock())
            service.get_corpus_or_404 = AsyncMock(return_value=_corpus())
            service.repo.create_run = AsyncMock(side_effect=_create)

            first = await service.compare_from_artifact(
                "corpus-1", user_id="user-1", input_artifact_id=artifact.artifact_id
            )
            second = await service.compare_from_artifact(
                "corpus-1", user_id="user-1", input_artifact_id=artifact.artifact_id
            )

        params = loads(first.parameters_json, {})
        self.assertEqual(params["input_artifact_id"], artifact.artifact_id)
        self.assertNotIn("values_a", params)
        self.assertNotIn("values_b", params)
        self.assertEqual(loads(first.results_json, {}), loads(second.results_json, {}))

    async def test_fit_persists_artifact_ids_without_raw_rows(self) -> None:
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": artifact_dir}),
        ):
            service = StatisticalModelingService(MagicMock())
            service.get_corpus_or_404 = AsyncMock(return_value=_corpus())
            service.repo.create_run = AsyncMock(side_effect=lambda run: run)

            run = await service.fit(
                "corpus-1",
                user_id="user-1",
                model="ols",
                dependent_var="y",
                independent_vars=["x"],
                rows=[
                    {"y": 1.0, "x": 1.0},
                    {"y": 2.0, "x": 2.0},
                    {"y": 3.0, "x": 3.0},
                    {"y": 4.0, "x": 4.0},
                ],
            )

        params = loads(run.parameters_json, {})
        self.assertTrue(params["input_artifact_id"].startswith("manifest:"))
        self.assertEqual(len(params["input_artifact_checksum"]), 64)
        self.assertNotIn("rows", params)
        self.assertEqual(params["n_rows"], 4)


if __name__ == "__main__":
    unittest.main()
