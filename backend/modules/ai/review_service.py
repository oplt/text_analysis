from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.ai.base_service import AiBaseService
from backend.modules.identity_access.models import User


class AiReviewService(AiBaseService):
    async def create_review(self, user: User, run_id: str, assigned_to_user_id: str | None):
        run = await self.repo.get_run_for_user(user.id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="AI run not found")
        review = await self.repo.create_review(
            run_id=run.id,
            requested_by_user_id=user.id,
            assigned_to_user_id=assigned_to_user_id,
            status="pending",
        )
        run.review_status = "pending"
        await self.db.commit()
        await self.db.refresh(review)
        return review

    async def list_reviews(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        return await self.repo.list_reviews_for_user(user.id, limit=limit, offset=offset)

    async def decide_review(self, user: User, review_id: str, payload: dict[str, Any]):
        review = await self.repo.get_review(review_id)
        if not review:
            raise HTTPException(status_code=404, detail="Review item not found")
        if not user.is_admin and user.id not in {
            review.requested_by_user_id,
            review.assigned_to_user_id,
        }:
            raise HTTPException(status_code=403, detail="You are not allowed to decide this review")
        run = await self.repo.get_run_for_user(review.requested_by_user_id, review.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="AI run not found")
        review.status = payload["status"]
        review.reviewed_by_user_id = user.id
        review.reviewer_notes = payload.get("reviewer_notes")
        review.corrected_output = payload.get("corrected_output")
        run.review_status = payload["status"]
        if payload.get("corrected_output"):
            run.output_text = payload["corrected_output"]
        await self.db.commit()
        await self.db.refresh(review)
        return review

    async def add_feedback(
        self,
        user: User,
        run_id: str,
        rating: int,
        comment: str | None,
        corrected_output: str | None,
    ):
        run = await self.repo.get_run_for_user(user.id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="AI run not found")
        feedback = await self.repo.create_feedback(
            run_id=run.id,
            user_id=user.id,
            rating=rating,
            comment=comment,
            corrected_output=corrected_output,
        )
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def list_feedback(
        self,
        user: User,
        run_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        run = await self.repo.get_run_for_user(user.id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="AI run not found")
        return await self.repo.list_feedback_for_run(run.id, limit=limit, offset=offset)
