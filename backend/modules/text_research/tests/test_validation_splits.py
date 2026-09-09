"""Unit tests for generic robustness/domain-shift and temporal validation
splits (§38, §39). Pure functions — no DB, no sklearn — run without asyncio
plugin autoload:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
      backend/modules/text_research/tests/test_validation_splits.py
"""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.validation_splits import (
    class_prevalence,
    eligible_group_values,
    expanding_window_splits,
    leave_one_group_out,
    rolling_window_splits,
    temporal_holdout_split,
    transfer_split,
)


class EligibleGroupValuesTests(unittest.TestCase):
    def test_drops_missing_and_dedupes(self):
        values = ["west", "east", None, "", "west", "unspecified"]
        self.assertEqual(eligible_group_values(values), ["east", "west"])


class LeaveOneGroupOutTests(unittest.TestCase):
    def test_generic_field_not_hardcoded_to_organization(self):
        # group_field could be "region", "country", or any custom facet —
        # the split logic itself is field-agnostic.
        regions = ["west", "west", "east", "east", "north"]
        splits = leave_one_group_out(regions)
        held_out = {s.held_out_value for s in splits}
        self.assertEqual(held_out, {"west", "east", "north"})
        west_split = next(s for s in splits if s.held_out_value == "west")
        self.assertEqual(west_split.test_index, [0, 1])
        self.assertEqual(west_split.train_index, [2, 3, 4])

    def test_missing_values_excluded_from_train_and_test(self):
        regions = ["west", None, "east", "west", "east"]
        splits = leave_one_group_out(regions)
        west_split = next(s for s in splits if s.held_out_value == "west")
        self.assertNotIn(1, west_split.train_index)
        self.assertNotIn(1, west_split.test_index)

    def test_single_group_yields_unevaluable_split(self):
        # Only one distinct value: the "leave it out" split has nothing left
        # to train on. Callers must check for an empty train_index rather
        # than assume every value yields a usable split.
        splits = leave_one_group_out(["only", "only", "only"])
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].train_index, [])
        self.assertEqual(splits[0].test_index, [0, 1, 2])

    def test_max_groups_caps_sweep(self):
        values = [str(i) for i in range(10)]
        splits = leave_one_group_out(values, max_groups=3)
        self.assertEqual(len(splits), 3)


class TransferSplitTests(unittest.TestCase):
    def test_train_where_a_test_where_b(self):
        values = ["A", "A", "B", "B", "C"]
        split = transfer_split(values, train_values=["A"], test_values=["B"])
        self.assertEqual(split.train_index, [0, 1])
        self.assertEqual(split.test_index, [2, 3])
        self.assertEqual(split.excluded_index, [4])

    def test_overlapping_train_test_values_rejected(self):
        with self.assertRaises(ValueError):
            transfer_split(["A", "B"], train_values=["A"], test_values=["A"])

    def test_empty_filters_rejected(self):
        with self.assertRaises(ValueError):
            transfer_split(["A", "B"], train_values=[], test_values=["B"])


class TemporalHoldoutTests(unittest.TestCase):
    def test_train_before_test_split(self):
        years = [2019, 2019, 2020, 2021, 2021, 2022]
        split = temporal_holdout_split(years)
        self.assertTrue(all(years[i] <= split.train_period for i in split.train_index))
        self.assertTrue(all(years[i] > split.train_period for i in split.test_index))
        self.assertEqual(set(split.train_index) | set(split.test_index), set(range(len(years))))

    def test_explicit_split_at(self):
        years = [2018, 2019, 2020, 2021]
        split = temporal_holdout_split(years, split_at=2019)
        self.assertEqual(split.train_index, [0, 1])
        self.assertEqual(split.test_index, [2, 3])

    def test_insufficient_periods_returns_empty_split(self):
        split = temporal_holdout_split([2020, 2020, 2020])
        self.assertEqual(split.train_index, [])
        self.assertEqual(split.test_index, [])

    def test_missing_values_excluded(self):
        years = [2019, None, 2020, None, 2021]
        split = temporal_holdout_split(years)
        self.assertEqual(set(split.excluded_index), {1, 3})

    def test_not_hardcoded_to_publication_year_field_name(self):
        # The function only ever sees raw values — caller decides which
        # metadata field they came from (§39).
        custom_field_values = ["2001-01-01", "2005-06-01", "2010-01-01", "2015-01-01"]
        split = temporal_holdout_split(custom_field_values)
        self.assertTrue(split.train_index)
        self.assertTrue(split.test_index)


class ExpandingWindowSplitsTests(unittest.TestCase):
    def test_train_grows_each_window(self):
        years = [2018, 2019, 2020, 2021]
        windows = expanding_window_splits(years, min_windows=3)
        self.assertGreaterEqual(len(windows), 2)
        sizes = [len(w.train_index) for w in windows]
        self.assertEqual(sizes, sorted(sizes))  # non-decreasing train size

    def test_too_few_periods_returns_empty(self):
        self.assertEqual(expanding_window_splits([2020, 2020, 2021]), [])


class RollingWindowSplitsTests(unittest.TestCase):
    def test_window_size_bounds_train_periods(self):
        years = [2018, 2019, 2020, 2021, 2022]
        windows = rolling_window_splits(years, window_size=2, min_windows=3)
        self.assertTrue(windows)
        for window in windows:
            self.assertLessEqual(len(set(window.train_period)), 2)


class ClassPrevalenceTests(unittest.TestCase):
    def test_prevalence_fractions(self):
        labels = [["a"], ["a", "b"], ["b"], []]
        prevalence = class_prevalence(labels)
        self.assertAlmostEqual(prevalence["a"], 0.5)
        self.assertAlmostEqual(prevalence["b"], 0.5)

    def test_empty_input(self):
        self.assertEqual(class_prevalence([]), {})


if __name__ == "__main__":
    unittest.main()
