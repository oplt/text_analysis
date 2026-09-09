"""Lifecycle transitions for trained classifiers (candidate → approved → deprecated)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy import select

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import ModelLifecycleStatus
from backend.modules.text_research.domain.models import TrainedModel, loads


def _utcnow() -> datetime:
    return datetime.now(UTC)


def label_set_hash(label_ids_json: str) -> str:
    """Stable hash for a model's fitted label set."""
    labels = loads(label_ids_json, [])
    canonical = "|".join(sorted(str(label) for label in labels))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ModelLifecycleService(ResearchAccessMixin):
    async def set_status(
        self,
        model_id: str,
        *,
        user_id: str,
        status: str,
        notes: str | None = None,
        deprecate_others: bool = False,
    ) -> TrainedModel:
        try:
            lifecycle = ModelLifecycleStatus(status)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid lifecycle status {status!r}; "
                    "expected candidate, approved, or deprecated"
                ),
            ) from exc

        model = await self.get_model_or_404(model_id, user_id=user_id)
        now = _utcnow()

        if lifecycle == ModelLifecycleStatus.APPROVED and deprecate_others:
            await self._deprecate_siblings(model, exclude_id=model.id, now=now)

        model.lifecycle_status = lifecycle.value
        model.lifecycle_notes = notes
        model.lifecycle_updated_at = now
        await self.db.flush()
        await self.db.commit()
        refreshed = await self.repo.get_model(model_id)
        assert refreshed is not None
        return refreshed

    async def list_by_status(
        self,
        project_id: str,
        *,
        user_id: str,
        status: str | None = None,
        corpus_id: str | None = None,
    ) -> list[TrainedModel]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        if status is not None:
            try:
                ModelLifecycleStatus(status)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid lifecycle status {status!r}",
                ) from exc
        return await self.repo.list_models(project_id, corpus_id=corpus_id, lifecycle_status=status)

    async def _deprecate_siblings(
        self,
        model: TrainedModel,
        *,
        exclude_id: str,
        now: datetime,
    ) -> None:
        target_hash = label_set_hash(model.label_ids_json)
        stmt = select(TrainedModel).where(
            TrainedModel.project_id == model.project_id,
            TrainedModel.corpus_id == model.corpus_id,
            TrainedModel.task_type == model.task_type,
            TrainedModel.id != exclude_id,
            TrainedModel.lifecycle_status != ModelLifecycleStatus.DEPRECATED.value,
        )
        result = await self.db.execute(stmt)
        for sibling in result.scalars().all():
            if label_set_hash(sibling.label_ids_json) != target_hash:
                continue
            sibling.lifecycle_status = ModelLifecycleStatus.DEPRECATED.value
            sibling.lifecycle_notes = (
                sibling.lifecycle_notes or f"Auto-deprecated when model {exclude_id} was approved"
            )
            sibling.lifecycle_updated_at = now
