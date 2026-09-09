"""Tests for `TrainingDatasetSnapshot` freeze-time metadata capture.

Plain ``unittest.TestCase`` tests against the pure helpers in
`dataset_builder_service.py` plus the `TrainingDatasetSnapshot` accessor
static methods, which only read `class_distribution_json` — no DB/FastAPI
dependency required.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from backend.modules.text_research.application.dataset_builder_service import (
    DatasetBuilderService,
    aggregate_checksum,
    build_snapshot_metadata,
)
from backend.modules.text_research.domain.models import TrainingDatasetSnapshot, dumps


class AggregateChecksumTests(unittest.TestCase):
    def test_none_when_no_checksums(self):
        self.assertIsNone(aggregate_checksum([]))
        self.assertIsNone(aggregate_checksum([None, None]))

    def test_deterministic_and_order_independent(self):
        a = aggregate_checksum(["hash-1", "hash-2", "hash-3"])
        b = aggregate_checksum(["hash-3", "hash-1", "hash-2"])
        self.assertEqual(a, b)
        self.assertIsInstance(a, str)
        self.assertEqual(len(a), 64)  # sha256 hex digest

    def test_changes_when_a_checksum_changes(self):
        original = aggregate_checksum(["hash-1", "hash-2"])
        changed = aggregate_checksum(["hash-1", "hash-2-modified"])
        self.assertNotEqual(original, changed)

    def test_ignores_missing_entries(self):
        with_none = aggregate_checksum(["hash-1", None, "hash-2"])
        without_none = aggregate_checksum(["hash-1", "hash-2"])
        self.assertEqual(with_none, without_none)


class BuildSnapshotMetadataTests(unittest.TestCase):
    def test_captures_resolution_strategy_and_codebook_version(self):
        frozen_at = datetime(2026, 1, 1, tzinfo=UTC)
        metadata = build_snapshot_metadata(
            codebook_version="2.0",
            annotation_source="majority_vote",
            selected_annotator_id=None,
            minimum_agreement=0.75,
            frozen_at=frozen_at,
        )
        self.assertEqual(metadata["codebook_version"], "2.0")
        self.assertEqual(metadata["annotation_resolution_strategy"], "majority_vote")
        self.assertEqual(metadata["minimum_agreement"], 0.75)
        self.assertEqual(metadata["frozen_at"], frozen_at.isoformat())


class SnapshotAccessorTests(unittest.TestCase):
    """Static accessors read straight from a persisted snapshot's JSON payload."""

    def _snapshot(self, **payload) -> TrainingDatasetSnapshot:
        return TrainingDatasetSnapshot(class_distribution_json=dumps(payload))

    def test_text_hashes_round_trip(self):
        snapshot = self._snapshot(text_hashes={"unit-1": "hash-a", "unit-2": "hash-b"})
        self.assertEqual(
            DatasetBuilderService.text_hashes(snapshot),
            {"unit-1": "hash-a", "unit-2": "hash-b"},
        )

    def test_text_hashes_defaults_to_empty_dict_for_legacy_snapshots(self):
        # Snapshots frozen before this change have no "text_hashes" key.
        snapshot = self._snapshot(distribution={}, unit_labels={})
        self.assertEqual(DatasetBuilderService.text_hashes(snapshot), {})

    def test_corpus_checksums_and_aggregate_round_trip(self):
        snapshot = self._snapshot(
            corpus_checksums={"doc-1": "chk-1", "doc-2": "chk-2"},
            corpus_checksum_aggregate=aggregate_checksum(["chk-1", "chk-2"]),
        )
        self.assertEqual(
            DatasetBuilderService.corpus_checksums(snapshot),
            {"doc-1": "chk-1", "doc-2": "chk-2"},
        )
        self.assertEqual(
            DatasetBuilderService.corpus_checksum_aggregate(snapshot),
            aggregate_checksum(["chk-1", "chk-2"]),
        )

    def test_snapshot_metadata_round_trip(self):
        metadata_payload = build_snapshot_metadata(
            codebook_version="1.0",
            annotation_source="adjudicated_only",
            selected_annotator_id=None,
            minimum_agreement=None,
        )
        snapshot = self._snapshot(metadata=metadata_payload)
        self.assertEqual(DatasetBuilderService.snapshot_metadata(snapshot), metadata_payload)

    def test_existing_unit_labels_and_label_names_accessors_still_work(self):
        # New keys must be additive; must not disturb pre-existing accessors.
        snapshot = self._snapshot(
            distribution={"LabelA": {"yes": 2, "no": 1}},
            unit_labels={"unit-1": ["LabelA"]},
            text_hashes={"unit-1": "hash-a"},
        )
        self.assertEqual(DatasetBuilderService.label_names(snapshot), ["LabelA"])
        self.assertEqual(DatasetBuilderService.unit_labels(snapshot), {"unit-1": ["LabelA"]})


if __name__ == "__main__":
    unittest.main()
