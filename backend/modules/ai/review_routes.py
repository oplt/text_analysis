from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.ai.review_service import AiReviewService
from backend.modules.ai.schemas import (
    AiFeedbackCreate,
    AiFeedbackResponse,
    AiReviewCreate,
    AiReviewDecision,
    AiReviewItemResponse,
)
from backend.modules.ai.serializers import _feedback_to_response, _review_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/reviews", response_model=PaginatedResponse[AiReviewItemResponse])
async def list_reviews(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiReviewService(db)
    reviews, total = await service.list_reviews(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [_review_to_response(item) for item in reviews],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/runs/{run_id}/reviews", response_model=AiReviewItemResponse, status_code=201)
async def create_review(
    run_id: str,
    payload: AiReviewCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiReviewService(db)
    review = await service.create_review(current_user, run_id, payload.assigned_to_user_id)
    return _review_to_response(review)


@router.post("/reviews/{review_id}/decision", response_model=AiReviewItemResponse)
async def decide_review(
    review_id: str,
    payload: AiReviewDecision,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiReviewService(db)
    review = await service.decide_review(current_user, review_id, payload.model_dump())
    return _review_to_response(review)


@router.get("/runs/{run_id}/feedback", response_model=PaginatedResponse[AiFeedbackResponse])
async def list_feedback(
    run_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiReviewService(db)
    feedback_items, total = await service.list_feedback(
        current_user,
        run_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_feedback_to_response(item) for item in feedback_items],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/runs/{run_id}/feedback", response_model=AiFeedbackResponse, status_code=201)
async def create_feedback(
    run_id: str,
    payload: AiFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiReviewService(db)
    feedback = await service.add_feedback(
        current_user,
        run_id,
        payload.rating,
        payload.comment,
        payload.corrected_output,
    )
    return _feedback_to_response(feedback)
