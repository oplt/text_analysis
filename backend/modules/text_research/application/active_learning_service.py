"""Active learning loop: rank uncertain predictions and enqueue them for
human annotation."""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.annotation_service import AnnotationService
from backend.modules.text_research.domain.models import AnnotationTask


class ActiveLearningService(ResearchAccessMixin):
    async def uncertain_queue(
        self,
        model_id: str,
        *,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        model = await self.get_model_or_404(model_id, user_id=user_id)
        predictions, total = await self.repo.list_predictions_for_model(
            model.id, limit=limit, offset=offset, order_by_uncertainty=True
        )
        units = {
            unit.id: unit
            for unit in await self.repo.list_text_units_by_ids(
                [p.text_unit_id for p in predictions]
            )
        }
        items = [
            {"prediction": prediction, "text_unit": units.get(prediction.text_unit_id)}
            for prediction in predictions
            if prediction.text_unit_id in units
        ]
        return items, total

    async def send_to_annotation(
        self,
        model_id: str,
        *,
        user_id: str,
        text_unit_ids: list[str],
        annotator_ids: list[str],
    ) -> list[AnnotationTask]:
        await self.get_model_or_404(model_id, user_id=user_id)
        annotation_service = AnnotationService(self.db)
        return await annotation_service.assign_tasks(
            user_id=user_id, text_unit_ids=text_unit_ids, annotator_ids=annotator_ids
        )
