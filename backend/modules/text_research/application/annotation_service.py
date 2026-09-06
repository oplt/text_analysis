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
from backend.modules.text_research.domain.enums import AnnotationTaskStatus
from backend.modules.text_research.domain.models import Annotation, AnnotationTask


class AnnotationService(ResearchAccessMixin):
    async def assign_tasks(
        self, *, user_id: str, text_unit_ids: list[str], annotator_ids: list[str]
    ) -> list[AnnotationTask]:
        tasks: list[AnnotationTask] = []
        for text_unit_id in text_unit_ids:
            await self.get_text_unit_or_404(text_unit_id, user_id=user_id)
            for annotator_id in annotator_ids:
                task = await self.repo.get_task(text_unit_id, annotator_id)
                if task is None:
                    task = await self.repo.create_task(
                        text_unit_id=text_unit_id, annotator_id=annotator_id
                    )
                tasks.append(task)
        await self.db.commit()
        return tasks

    async def list_queue(
        self,
        *,
        requesting_user_id: str,
        annotator_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Annotators may only list their own queue in this MVP."""
        target_annotator = annotator_id or requesting_user_id
        if target_annotator != requesting_user_id:
            raise HTTPException(status_code=403, detail="Cannot view another annotator's queue")

        tasks = await self.repo.list_tasks_for_annotator(target_annotator)
        if status:
            tasks = [t for t in tasks if t.status == status]
        units = {
            unit.id: unit
            for unit in await self.repo.list_text_units_by_ids([t.text_unit_id for t in tasks])
        }
        return [
            {"task": task, "text_unit": units.get(task.text_unit_id)}
            for task in tasks
            if task.text_unit_id in units
        ]

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

    async def progress(self, corpus_id: str, *, user_id: str) -> dict[str, Any]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        units = await self.repo.list_text_units_for_corpus(corpus_id)
        unit_ids = [u.id for u in units]
        tasks = await self.repo.list_tasks_for_units(unit_ids)

        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == AnnotationTaskStatus.COMPLETED.value)
        by_annotator: dict[str, dict[str, int]] = {}
        for task in tasks:
            bucket = by_annotator.setdefault(task.annotator_id, {"assigned": 0, "completed": 0})
            bucket["assigned"] += 1
            if task.status == AnnotationTaskStatus.COMPLETED.value:
                bucket["completed"] += 1

        annotated_unit_ids = {t.text_unit_id for t in tasks if t.status == "completed"}
        return {
            "total_units": len(unit_ids),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks else 0.0,
            "units_with_completed_annotation": len(annotated_unit_ids),
            "by_annotator": by_annotator,
        }
