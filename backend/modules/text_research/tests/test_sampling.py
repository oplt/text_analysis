"""Tests for stratified/random annotation sampling (prompt.txt §24)."""

from __future__ import annotations

import unittest

from backend.modules.text_research.infrastructure.sampling import (
    SamplingItem,
    build_sampling_plan,
)


def _fake_items(n: int) -> list[SamplingItem]:
    """Synthetic units with two independent, generic metadata fields.

    Deliberately named field_1 / field_2 (not geo/org) to prove
    stratification is not hardcoded to any particular dimension.
    """
    field_1_values = ["alpha", "beta", "gamma"]
    field_2_values = ["x", "y"]
    items = []
    for i in range(n):
        items.append(
            SamplingItem(
                id=f"unit-{i}",
                document_id=f"doc-{i // 5}",  # 5 units per document
                metadata={
                    "field_1": field_1_values[i % len(field_1_values)],
                    "field_2": field_2_values[i % len(field_2_values)],
                },
            )
        )
    return items


class SeededReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_identical_selection(self):
        items = _fake_items(50)
        plan_a = build_sampling_plan(items, sample_size=10, random_seed=42)
        plan_b = build_sampling_plan(items, sample_size=10, random_seed=42)
        self.assertEqual(plan_a["selected_unit_ids"], plan_b["selected_unit_ids"])
        self.assertEqual(plan_a["random_seed"], 42)
        self.assertEqual(plan_b["random_seed"], 42)

    def test_different_seed_can_produce_different_selection(self):
        items = _fake_items(50)
        plan_a = build_sampling_plan(items, sample_size=10, random_seed=1)
        plan_b = build_sampling_plan(items, sample_size=10, random_seed=2)
        self.assertNotEqual(plan_a["selected_unit_ids"], plan_b["selected_unit_ids"])

    def test_no_seed_generates_and_returns_one(self):
        items = _fake_items(20)
        plan = build_sampling_plan(items, sample_size=5)
        self.assertIsInstance(plan["random_seed"], int)
        # Reproducing with the returned seed gives the exact same sample.
        replay = build_sampling_plan(items, sample_size=5, random_seed=plan["random_seed"])
        self.assertEqual(plan["selected_unit_ids"], replay["selected_unit_ids"])

    def test_no_stratify_by_is_plain_seeded_random_sample(self):
        items = _fake_items(30)
        plan = build_sampling_plan(items, sample_size=12, random_seed=7)
        self.assertEqual(plan["sampling_config"]["stratify_by"], [])
        self.assertEqual(len(plan["selected_unit_ids"]), 12)
        self.assertEqual(len(set(plan["selected_unit_ids"])), 12)
        # Every selected id must come from the candidate pool.
        pool = {item.id for item in items}
        self.assertTrue(set(plan["selected_unit_ids"]).issubset(pool))


class StratificationTests(unittest.TestCase):
    def test_stratifies_by_generic_fake_metadata_fields(self):
        items = _fake_items(60)
        plan = build_sampling_plan(
            items,
            sample_size=18,
            random_seed=5,
            stratify_by=["field_1", "field_2"],
            stratum_mode="proportional",
        )
        # 3 field_1 values x 2 field_2 values = 6 strata.
        self.assertEqual(len(plan["strata"]), 6)
        stratum_fields = {frozenset(s["stratum"].keys()) for s in plan["strata"]}
        self.assertEqual(stratum_fields, {frozenset({"field_1", "field_2"})})
        self.assertEqual(sum(s["selected"] for s in plan["strata"]), 18)
        # Proportional: strata are equally sized here, so each gets exactly 3.
        for stratum in plan["strata"]:
            self.assertEqual(stratum["available"], 10)
            self.assertEqual(stratum["selected"], 3)

    def test_proportional_allocation_favors_larger_strata(self):
        # 40 items in field_1=alpha, 10 items in field_1=beta.
        items = [
            SamplingItem(id=f"a{i}", document_id=f"doc-a{i}", metadata={"field_1": "alpha"})
            for i in range(40)
        ] + [
            SamplingItem(id=f"b{i}", document_id=f"doc-b{i}", metadata={"field_1": "beta"})
            for i in range(10)
        ]
        plan = build_sampling_plan(
            items, sample_size=10, random_seed=3, stratify_by=["field_1"], stratum_mode="proportional"
        )
        by_value = {s["stratum"]["field_1"]: s["selected"] for s in plan["strata"]}
        self.assertEqual(by_value["alpha"], 8)
        self.assertEqual(by_value["beta"], 2)

    def test_equal_per_stratum_allocation_ignores_stratum_size(self):
        items = [
            SamplingItem(id=f"a{i}", document_id=f"doc-a{i}", metadata={"field_1": "alpha"})
            for i in range(40)
        ] + [
            SamplingItem(id=f"b{i}", document_id=f"doc-b{i}", metadata={"field_1": "beta"})
            for i in range(10)
        ]
        plan = build_sampling_plan(
            items, sample_size=10, random_seed=3, stratify_by=["field_1"], stratum_mode="equal"
        )
        by_value = {s["stratum"]["field_1"]: s["selected"] for s in plan["strata"]}
        self.assertEqual(by_value["alpha"], 5)
        self.assertEqual(by_value["beta"], 5)

    def test_equal_mode_redistributes_when_small_stratum_is_exhausted(self):
        # beta only has 2 members; equal split of 10 across 2 strata wants 5
        # each, so the 3 unfillable beta slots must go back to alpha.
        items = [
            SamplingItem(id=f"a{i}", document_id=f"doc-a{i}", metadata={"field_1": "alpha"})
            for i in range(40)
        ] + [
            SamplingItem(id=f"b{i}", document_id=f"doc-b{i}", metadata={"field_1": "beta"})
            for i in range(2)
        ]
        plan = build_sampling_plan(
            items, sample_size=10, random_seed=9, stratify_by=["field_1"], stratum_mode="equal"
        )
        by_value = {s["stratum"]["field_1"]: s["selected"] for s in plan["strata"]}
        self.assertEqual(by_value["beta"], 2)
        self.assertEqual(by_value["alpha"], 8)
        self.assertEqual(plan["total_selected"], 10)

    def test_missing_stratify_field_forms_its_own_stratum(self):
        items = [
            SamplingItem(id="u1", document_id="d1", metadata={"field_1": "alpha"}),
            SamplingItem(id="u2", document_id="d2", metadata={}),
        ]
        plan = build_sampling_plan(
            items, sample_size=2, random_seed=1, stratify_by=["field_1"]
        )
        self.assertEqual(len(plan["strata"]), 2)
        values = {s["stratum"]["field_1"] for s in plan["strata"]}
        self.assertEqual(values, {"alpha", None})

    def test_unknown_stratum_mode_raises(self):
        items = _fake_items(5)
        with self.assertRaises(ValueError):
            build_sampling_plan(items, sample_size=2, stratum_mode="bogus")

    def test_unknown_sampling_level_raises(self):
        items = _fake_items(5)
        with self.assertRaises(ValueError):
            build_sampling_plan(items, sample_size=2, sampling_level="bogus")


class MaxUnitsPerDocumentTests(unittest.TestCase):
    def test_caps_units_selected_per_document_unit_level(self):
        # 5 documents x 10 units each, no stratification.
        items = [
            SamplingItem(id=f"u{doc}-{i}", document_id=f"doc-{doc}", metadata={})
            for doc in range(5)
            for i in range(10)
        ]
        plan = build_sampling_plan(
            items, sample_size=30, random_seed=11, max_units_per_document=3
        )
        counts: dict[str, int] = {}
        for unit_id in plan["selected_unit_ids"]:
            doc_id = unit_id.rsplit("-", 1)[0]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        for doc_id, count in counts.items():
            self.assertLessEqual(count, 3)
        # 5 docs * cap 3 = 15 available under the cap, less than sample_size.
        self.assertEqual(plan["total_selected"], 15)

    def test_caps_units_selected_per_document_within_stratum(self):
        items = [
            SamplingItem(
                id=f"u{doc}-{i}",
                document_id=f"doc-{doc}",
                metadata={"field_1": "alpha" if doc % 2 == 0 else "beta"},
            )
            for doc in range(6)
            for i in range(10)
        ]
        plan = build_sampling_plan(
            items,
            sample_size=20,
            random_seed=4,
            stratify_by=["field_1"],
            max_units_per_document=2,
        )
        counts: dict[str, int] = {}
        for unit_id in plan["selected_unit_ids"]:
            doc_id = unit_id.rsplit("-", 1)[0]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        for doc_id, count in counts.items():
            self.assertLessEqual(count, 2)

    def test_document_level_sampling_caps_units_per_selected_document(self):
        items = [
            SamplingItem(id=f"u{doc}-{i}", document_id=f"doc-{doc}", metadata={"field_1": "alpha"})
            for doc in range(4)
            for i in range(10)
        ]
        plan = build_sampling_plan(
            items,
            sample_size=2,
            random_seed=6,
            sampling_level="document",
            max_units_per_document=4,
        )
        self.assertEqual(plan["sampling_config"]["sampling_level"], "document")
        self.assertEqual(len(plan["selected_document_ids"]), 2)
        self.assertEqual(len(plan["selected_unit_ids"]), 8)
        counts: dict[str, int] = {}
        for unit_id in plan["selected_unit_ids"]:
            doc_id = unit_id.rsplit("-", 1)[0]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        for doc_id, count in counts.items():
            self.assertEqual(count, 4)

    def test_invalid_max_units_per_document_raises(self):
        items = _fake_items(5)
        with self.assertRaises(ValueError):
            build_sampling_plan(items, sample_size=2, max_units_per_document=0)


if __name__ == "__main__":
    unittest.main()
