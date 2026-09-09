"""Unit tests for contextual CSV parsing helpers and pearson join math."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from backend.modules.text_research.application.contextual_dataset_service import (
    ContextualDatasetService,
    _parse_numeric,
    _pearson,
)


class ContextualHelpersTests(unittest.TestCase):
    def test_parse_numeric_handles_common_missing_tokens(self):
        self.assertIsNone(_parse_numeric("NA"))
        self.assertIsNone(_parse_numeric(""))
        self.assertEqual(_parse_numeric("12.5"), 12.5)
        self.assertEqual(_parse_numeric("1,200"), 1200.0)

    def test_pearson_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(_pearson(xs, ys) or 0.0, 1.0)

    def test_pearson_requires_three_points(self):
        self.assertIsNone(_pearson([1.0, 2.0], [1.0, 2.0]))


class ContextualDatasetServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_datasets_uses_grouped_observation_counts(self):
        dataset = SimpleNamespace(
            id="dataset-1",
            project_id="project-1",
            name="Indicators",
            description=None,
            created_by="user-1",
            created_at=datetime.now(UTC),
        )

        class Repository:
            async def list_contextual_datasets_with_counts(self, project_id: str):
                self.project_id = project_id
                return [(dataset, 3)]

        service = ContextualDatasetService.__new__(ContextualDatasetService)
        service.repo = Repository()

        async def allow_access(*, user_id: str, project_id: str):
            self.assertEqual((user_id, project_id), ("user-1", "project-1"))

        service.ensure_project_access = allow_access
        rows = await service.list_datasets(project_id="project-1", user_id="user-1")

        self.assertEqual(rows[0]["observation_count"], 3)
        self.assertEqual(service.repo.project_id, "project-1")


if __name__ == "__main__":
    unittest.main()
