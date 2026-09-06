"""Unit tests for contextual CSV parsing helpers and pearson join math."""

from __future__ import annotations

import unittest

from backend.modules.text_research.application.contextual_dataset_service import (
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


if __name__ == "__main__":
    unittest.main()
