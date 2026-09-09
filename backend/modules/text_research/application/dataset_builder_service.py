"""Build and freeze `TrainingDatasetSnapshot`s from human annotations.

Models are never trained against mutable "current" annotations — only
against a frozen snapshot whose exact unit IDs, document IDs, and per-unit
gold labels are persisted at freeze time (inside `class_distribution_json`,
alongside the aggregate class distribution), so retraining against the same
snapshot is always reproducible even if annotations change afterward.

`class_distribution_json` additionally captures, so the snapshot is fully
self-describing without joining back to mutable tables:

* ``text_hashes`` — ``{unit_id: TextUnit.text_hash}`` at freeze time.
* ``corpus_checksums`` — ``{corpus_document_id: canonical_text_checksum}``
  for every source document contributing a unit, plus
  ``corpus_checksum_aggregate``, a single deterministic hash of those
  checksums that changes if any contributing document's canonical text
  changes (only available for corpora ingested through the canonical-source
  pipeline; ``None`` when no canonical sources exist).
* ``metadata`` — codebook version, annotation resolution strategy, and the
  freeze timestamp (mirroring the equivalent model columns, for consumers
  that only have the JSON payload).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import (
    TextUnit,
    TrainingDatasetSnapshot,
    dumps,
    loads,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def aggregate_checksum(checksums: Iterable[str | None]) -> str | None:
    """Deterministic combined checksum for a set of document checksums.

    Sorting before hashing makes the result independent of document
    ordering; ``None``/empty values are dropped. Returns ``None`` if no
    non-empty checksums are provided (e.g. no canonical sources exist yet).
    """
    present = sorted(c for c in checksums if c)
    if not present:
        return None
    return hashlib.sha256("|".join(present).encode("utf-8")).hexdigest()


def build_snapshot_metadata(
    *,
    codebook_version: str,
    annotation_source: str,
    selected_annotator_id: str | None,
    minimum_agreement: float | None,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    """Pure builder for the ``metadata`` block stored in a frozen snapshot."""
    return {
        "codebook_version": codebook_version,
        "annotation_resolution_strategy": annotation_source,
        "selected_annotator_id": selected_annotator_id,
        "minimum_agreement": minimum_agreement,
        "frozen_at": (frozen_at or _utcnow()).isoformat(),
    }


class DatasetBuilderService(ResearchAccessMixin):
    async def _resolve_gold_values(
        self,
        corpus_id: str,
        *,
        user_id: str,
        codebook_id: str,
        label_ids: list[str],
        annotation_source: str,
        selected_annotator_id: str | None = None,
        minimum_agreement: float | None = None,
    ) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        annotations = await self.repo.list_annotations_for_corpus(corpus_id)
        annotations = [
            a
            for a in annotations
            if a.codebook_version == codebook.version and a.label_id in label_ids
        ]
        adjudications = await self.repo.list_adjudications_for_corpus(corpus_id)
        adjudication_lookup = {(a.text_unit_id, a.label_id): a.final_value for a in adjudications}

        # label_id -> unit_id -> annotator_id -> value
        by_label_unit: dict[str, dict[str, dict[str, str]]] = {}
        for a in annotations:
            by_label_unit.setdefault(a.label_id, {}).setdefault(a.text_unit_id, {})[
                a.annotator_id
            ] = a.value

        resolved: dict[str, dict[str, str]] = {}
        excluded_disagreements: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []

        for label_id in label_ids:
            per_unit = by_label_unit.get(label_id, {})
            for unit_id, coder_values in per_unit.items():
                if annotation_source == "adjudicated_only":
                    adjudicated = adjudication_lookup.get((unit_id, label_id))
                    if adjudicated is not None:
                        resolved.setdefault(label_id, {})[unit_id] = adjudicated
                        continue
                    distinct = set(coder_values.values())
                    if len(distinct) == 1:
                        resolved.setdefault(label_id, {})[unit_id] = next(iter(distinct))
                    else:
                        excluded_disagreements.append(
                            {"text_unit_id": unit_id, "label_id": label_id}
                        )
                elif annotation_source == "majority_vote":
                    votes = Counter(coder_values.values())
                    top_value, top_count = votes.most_common(1)[0]
                    agreement = top_count / sum(votes.values())
                    if minimum_agreement is not None and agreement < minimum_agreement:
                        excluded_disagreements.append(
                            {"text_unit_id": unit_id, "label_id": label_id}
                        )
                        continue
                    resolved.setdefault(label_id, {})[unit_id] = top_value
                elif annotation_source == "selected_annotator":
                    if selected_annotator_id in coder_values:
                        resolved.setdefault(label_id, {})[unit_id] = coder_values[
                            selected_annotator_id
                        ]
                    else:
                        missing.append({"text_unit_id": unit_id, "label_id": label_id})
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unknown annotation_source: {annotation_source}"
                    )

        return resolved, excluded_disagreements, missing

    async def preview(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        codebook_id: str,
        label_ids: list[str],
        annotation_source: str,
        selected_annotator_id: str | None = None,
        minimum_agreement: float | None = None,
    ) -> dict[str, Any]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        if not label_ids:
            raise HTTPException(status_code=400, detail="At least one label is required")

        resolved, excluded, missing = await self._resolve_gold_values(
            corpus_id,
            user_id=user_id,
            codebook_id=codebook_id,
            label_ids=label_ids,
            annotation_source=annotation_source,
            selected_annotator_id=selected_annotator_id,
            minimum_agreement=minimum_agreement,
        )

        fully_labeled_unit_ids = (
            set.intersection(*(set(resolved.get(label_id, {})) for label_id in label_ids))
            if label_ids
            else set()
        )

        candidate_units: list[TextUnit] = await self.repo.list_text_units_by_ids(
            list(fully_labeled_unit_ids)
        )
        candidate_units = [u for u in candidate_units if u.unit_type == unit_type]
        unit_ids = sorted(u.id for u in candidate_units)
        document_ids = sorted({u.corpus_document_id for u in candidate_units})
        text_hashes = {u.id: u.text_hash for u in candidate_units}

        canonical_sources = await self.repo.list_canonical_sources_for_corpus(corpus_id)
        corpus_checksums = {
            source.corpus_document_id: source.canonical_text_checksum
            for source in canonical_sources
            if source.corpus_document_id in document_ids
        }
        corpus_checksum_aggregate = aggregate_checksum(corpus_checksums.values())

        labels = {label_id: await self.repo.get_label(label_id) for label_id in label_ids}
        class_distribution: dict[str, dict[str, int]] = {}
        unit_labels: dict[str, list[str]] = {unit_id: [] for unit_id in unit_ids}
        for label_id in label_ids:
            label = labels[label_id]
            label_name = label.name if label else label_id
            values = [resolved[label_id][unit_id] for unit_id in unit_ids]
            class_distribution[label_name] = dict(Counter(values))
            for unit_id in unit_ids:
                if resolved[label_id][unit_id] == "yes":
                    unit_labels[unit_id].append(label_name)

        warnings: list[str] = []
        if len(unit_ids) < 20:
            warnings.append(
                f"Only {len(unit_ids)} fully-labeled units are available; results will be unstable."
            )
        for label_name, dist in class_distribution.items():
            if len(dist) < 2:
                warnings.append(f"Label '{label_name}' has only one observed class value.")
            elif dist.get("yes", 0) < 5:
                warnings.append(f"Label '{label_name}' has very few positive ('yes') examples.")

        annotator_ids: set[str] = set()
        for per_unit in await self.repo.list_annotations_for_units(unit_ids) if unit_ids else []:
            annotator_ids.add(per_unit.annotator_id)

        return {
            "unit_count": len(unit_ids),
            "document_count": len(document_ids),
            "unit_ids": unit_ids,
            "document_ids": document_ids,
            "class_distribution": class_distribution,
            "unit_labels": unit_labels,
            "missing_labels": missing,
            "excluded_disagreements": excluded,
            "annotator_coverage": sorted(annotator_ids),
            "warnings": warnings,
            "text_hashes": text_hashes,
            "corpus_checksums": corpus_checksums,
            "corpus_checksum_aggregate": corpus_checksum_aggregate,
        }

    async def freeze(
        self,
        corpus_id: str,
        *,
        user_id: str,
        name: str,
        unit_type: str,
        codebook_id: str,
        label_ids: list[str],
        annotation_source: str,
        selected_annotator_id: str | None = None,
        minimum_agreement: float | None = None,
    ) -> TrainingDatasetSnapshot:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        preview = await self.preview(
            corpus_id,
            user_id=user_id,
            unit_type=unit_type,
            codebook_id=codebook_id,
            label_ids=label_ids,
            annotation_source=annotation_source,
            selected_annotator_id=selected_annotator_id,
            minimum_agreement=minimum_agreement,
        )
        if preview["unit_count"] == 0:
            raise HTTPException(
                status_code=422,
                detail="No fully-labeled text units available for the requested configuration",
            )

        snapshot = await self.repo.create_snapshot(
            TrainingDatasetSnapshot(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                name=name,
                unit_type=unit_type,
                codebook_id=codebook_id,
                codebook_version=codebook.version,
                annotation_source=annotation_source,
                minimum_agreement=minimum_agreement,
                unit_ids_json=dumps(preview["unit_ids"]),
                document_ids_json=dumps(preview["document_ids"]),
                labels_json=dumps(label_ids),
                class_distribution_json=dumps(
                    {
                        "distribution": preview["class_distribution"],
                        "unit_labels": preview["unit_labels"],
                        "text_hashes": preview["text_hashes"],
                        "corpus_checksums": preview["corpus_checksums"],
                        "corpus_checksum_aggregate": preview["corpus_checksum_aggregate"],
                        "metadata": build_snapshot_metadata(
                            codebook_version=codebook.version,
                            annotation_source=annotation_source,
                            selected_annotator_id=selected_annotator_id,
                            minimum_agreement=minimum_agreement,
                        ),
                    }
                ),
                created_by=user_id,
            )
        )
        await self.db.commit()
        return snapshot

    async def list_snapshots(self, *, project_id: str, user_id: str, corpus_id: str | None = None):
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_snapshots(project_id, corpus_id=corpus_id)

    async def get_snapshot(self, snapshot_id: str, *, user_id: str) -> TrainingDatasetSnapshot:
        return await self.get_snapshot_or_404(snapshot_id, user_id=user_id)

    @staticmethod
    def unit_labels(snapshot: TrainingDatasetSnapshot) -> dict[str, list[str]]:
        """Frozen per-unit positive label names, exactly as computed at freeze time."""
        data = loads(snapshot.class_distribution_json, {})
        return data.get("unit_labels", {})

    @staticmethod
    def label_names(snapshot: TrainingDatasetSnapshot) -> list[str]:
        data = loads(snapshot.class_distribution_json, {})
        return sorted(data.get("distribution", {}).keys())

    @staticmethod
    def text_hashes(snapshot: TrainingDatasetSnapshot) -> dict[str, str]:
        """Frozen ``{unit_id: text_hash}`` exactly as computed at freeze time."""
        data = loads(snapshot.class_distribution_json, {})
        return data.get("text_hashes", {})

    @staticmethod
    def corpus_checksums(snapshot: TrainingDatasetSnapshot) -> dict[str, str | None]:
        """Frozen ``{corpus_document_id: canonical_text_checksum}``."""
        data = loads(snapshot.class_distribution_json, {})
        return data.get("corpus_checksums", {})

    @staticmethod
    def corpus_checksum_aggregate(snapshot: TrainingDatasetSnapshot) -> str | None:
        """Single deterministic hash summarizing all contributing document checksums."""
        data = loads(snapshot.class_distribution_json, {})
        return data.get("corpus_checksum_aggregate")

    @staticmethod
    def snapshot_metadata(snapshot: TrainingDatasetSnapshot) -> dict[str, Any]:
        """Codebook version, resolution strategy, and freeze timestamp."""
        data = loads(snapshot.class_distribution_json, {})
        return data.get("metadata", {})
