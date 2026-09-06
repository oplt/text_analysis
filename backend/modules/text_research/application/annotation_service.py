"""Annotation assignment, multilabel saving, annotator queues, and progress
statistics.

Annotation history is never silently deleted: `save_annotations` always
updates-in-place by the `(text_unit_id, label_id, annotator_id,
codebook_version)` unique key via `ResearchRepository.upsert_annotation`.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.assignment_planning import (
    assignment_pairs,
    plan_annotation_assignment,
)
from backend.modules.text_research.domain.enums import AnnotationTaskStatus
from backend.modules.text_research.domain.models import Annotation, AnnotationTask


class AnnotationService(ResearchAccessMixin):
    async def assign_tasks(
        self, *, user_id: str, text_unit_ids: list[str], annotator_ids: list[str]
    ) -> list[AnnotationTask]:
        unique_unit_ids = list(dict.fromkeys(text_unit_ids))
        unit_corpora = await self.repo.corpus_ids_for_text_units(unique_unit_ids)
        missing_ids = set(unique_unit_ids) - set(unit_corpora)
        if missing_ids:
            raise HTTPException(status_code=404, detail="One or more text units were not found")
        for corpus_id in set(unit_corpora.values()):
            await self.get_corpus_or_404(corpus_id, user_id=user_id)
        pairs = [(unit_id, annotator_id) for unit_id in unique_unit_ids for annotator_id in annotator_ids]
        created = await self.repo.bulk_create_tasks(pairs)
        # Also return already-existing tasks for the requested pairs.
        existing = await self.repo.list_tasks_for_units(unique_unit_ids)
        wanted = {(unit_id, annotator_id) for unit_id, annotator_id in pairs}
        matched = [
            task
            for task in existing
            if (task.text_unit_id, task.annotator_id) in wanted
        ]
        await self.db.commit()
        return matched if matched else created

    async def assign_corpus_tasks(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        annotator_ids: list[str] | None = None,
        sample_size: int = 50,
        strategy: str = "overlap",
        overlap_count: int | None = None,
        overlap_percent: float | None = None,
    ) -> dict[str, Any]:
        """Assign corpus text units using an explicit sampling / overlap strategy."""
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        targets = annotator_ids or [user_id]
        if not targets:
            raise HTTPException(status_code=422, detail="At least one annotator is required")

        units = await self.repo.list_text_units_for_corpus(corpus_id, unit_type=unit_type)
        if not units:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No {unit_type} units found. Segment the corpus before creating "
                    "annotation tasks."
                ),
            )

        existing = await self.repo.list_tasks_for_units([unit.id for unit in units])
        already_assigned = {(task.text_unit_id, task.annotator_id) for task in existing}

        # Prefer units that are not already fully assigned to every target annotator.
        candidate_ids = [
            unit.id
            for unit in units
            if not all((unit.id, annotator_id) in already_assigned for annotator_id in targets)
        ]
        if not candidate_ids:
            raise HTTPException(
                status_code=422,
                detail="All available units already have annotation tasks for these annotators.",
            )

        try:
            plan = plan_annotation_assignment(
                candidate_ids,
                targets,
                sample_size=sample_size,
                strategy=strategy,
                overlap_count=overlap_count,
                overlap_percent=overlap_percent,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        pairs = assignment_pairs(plan)
        if not pairs:
            raise HTTPException(status_code=422, detail="Assignment plan produced no tasks.")

        created = await self.repo.bulk_create_tasks(pairs)
        await self.db.commit()

        unique_units = {unit_id for unit_id, _ in pairs}
        # Overlap = units assigned to every annotator in the plan
        overlap_units = 0
        if len(targets) > 1:
            membership: dict[str, set[str]] = {}
            for annotator_id, unit_ids in plan.items():
                for unit_id in unit_ids:
                    membership.setdefault(unit_id, set()).add(annotator_id)
            overlap_units = sum(
                1 for annotators in membership.values() if len(annotators) == len(targets)
            )

        return {
            "assigned_count": len(created),
            "unique_units": len(unique_units),
            "overlap_units": overlap_units,
            "strategy": strategy,
            "unit_type": unit_type,
            "per_annotator": {annotator_id: len(unit_ids) for annotator_id, unit_ids in plan.items()},
        }

    async def list_queue(
        self,
        *,
        requesting_user_id: str,
        annotator_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Annotators may only list their own queue in this MVP."""
        target_annotator = annotator_id or requesting_user_id
        if target_annotator != requesting_user_id:
            raise HTTPException(status_code=403, detail="Cannot view another annotator's queue")

        tasks, total = await self.repo.paginate_tasks_for_annotator(
            target_annotator, status=status, limit=limit, offset=offset
        )
        units = {
            unit.id: unit
            for unit in await self.repo.list_text_units_by_ids([t.text_unit_id for t in tasks])
        }
        return [
            {
                "task": {
                    "id": task.id,
                    "text_unit_id": task.text_unit_id,
                    "annotator_id": task.annotator_id,
                    "status": task.status,
                    "assigned_at": task.assigned_at,
                    "completed_at": task.completed_at,
                },
                "text_unit": (
                    {
                        "id": unit.id,
                        "corpus_document_id": unit.corpus_document_id,
                        "unit_type": unit.unit_type,
                        "position": unit.position,
                        "page_number": unit.page_number,
                        "paragraph_number": unit.paragraph_number,
                        "sentence_number": unit.sentence_number,
                        "text": unit.text,
                        "text_hash": unit.text_hash,
                        "created_at": unit.created_at,
                    }
                    if (unit := units.get(task.text_unit_id)) is not None
                    else None
                ),
            }
            for task in tasks
            if task.text_unit_id in units
        ], total

    async def save_annotations(
        self,
        *,
        user_id: str,
        text_unit_id: str,
        codebook_id: str,
        values: list[dict[str, Any]],
        mark_task_complete: bool = True,
    ) -> list[Annotation]:
        """`values` items: {label_id, value, confidence?, comment?}. Multilabel:
        pass one item per label the annotator wants to record."""
        await self.get_text_unit_or_404(text_unit_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)

        results: list[Annotation] = []
        for item in values:
            label = await self.repo.get_label(item["label_id"])
            if label is None or label.codebook_id != codebook_id:
                raise HTTPException(
                    status_code=400, detail=f"Label {item['label_id']} not in this codebook"
                )
            annotation = await self.repo.upsert_annotation(
                text_unit_id=text_unit_id,
                label_id=item["label_id"],
                annotator_id=user_id,
                codebook_version=codebook.version,
                value=item["value"],
                confidence=item.get("confidence"),
                comment=item.get("comment"),
            )
            results.append(annotation)

        if mark_task_complete:
            task = await self.repo.get_task(text_unit_id, user_id)
            if task is None:
                task = await self.repo.create_task(
                    text_unit_id=text_unit_id,
                    annotator_id=user_id,
                    status=AnnotationTaskStatus.COMPLETED.value,
                )
            else:
                await self.repo.update_task_status(task, AnnotationTaskStatus.COMPLETED.value)

        await self.db.commit()
        return results

    async def list_annotations_for_unit(
        self, text_unit_id: str, *, user_id: str
    ) -> list[Annotation]:
        await self.get_text_unit_or_404(text_unit_id, user_id=user_id)
        return await self.repo.list_annotations_for_unit(text_unit_id)

    async def get_unit_context(
        self, text_unit_id: str, *, user_id: str, window: int = 2
    ) -> dict[str, Any]:
        unit = await self.get_text_unit_or_404(text_unit_id, user_id=user_id)
        siblings = await self.repo.list_text_units_for_document(
            unit.corpus_document_id, unit_type=unit.unit_type
        )
        index = next((i for i, sibling in enumerate(siblings) if sibling.id == unit.id), 0)
        before = siblings[max(0, index - window) : index]
        after = siblings[index + 1 : index + 1 + window]
        document = await self.repo.get_document(unit.corpus_document_id)

        def _unit_payload(row) -> dict[str, Any]:
            return {
                "id": row.id,
                "corpus_document_id": row.corpus_document_id,
                "unit_type": row.unit_type,
                "position": row.position,
                "text": row.text,
            }

        return {
            "unit": _unit_payload(unit),
            "document": (
                {
                    "id": document.id,
                    "title": document.title,
                    "organization": document.organization,
                    "publication_year": document.publication_year,
                    "country": document.country,
                    "language": document.language,
                }
                if document
                else None
            ),
            "before": [_unit_payload(row) for row in before],
            "after": [_unit_payload(row) for row in after],
        }

    async def progress(self, corpus_id: str, *, user_id: str) -> dict[str, Any]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        progress = await self.repo.annotation_progress_for_corpus(corpus_id)
        total_tasks = progress["total_tasks"]
        progress["completion_rate"] = (
            progress["completed_tasks"] / total_tasks if total_tasks else 0.0
        )
        return progress
