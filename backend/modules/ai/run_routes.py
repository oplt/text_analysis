import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.ai.dependencies import enforce_ai_generation_rate_limit
from backend.modules.ai.run_service import AiRunService
from backend.modules.ai.schemas import AiRunRequest, AiRunResponse
from backend.modules.ai.serializers import run_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/runs", response_model=PaginatedResponse[AiRunResponse])
async def list_runs(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiRunService(db)
    runs, total = await service.list_runs(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [run_to_response(item) for item in runs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/runs", response_model=AiRunResponse, status_code=201)
async def create_run(
    payload: AiRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_ai_generation_rate_limit),
):
    service = AiRunService(db)
    try:
        run = await asyncio.wait_for(
            service.run_prompt(
                current_user,
                prompt_template_key=payload.prompt_template_key,
                prompt_version_id=payload.prompt_version_id,
                variables=payload.variables,
                retrieval_query=payload.retrieval_query,
                document_ids=payload.document_ids,
                top_k=payload.top_k,
                review_required=payload.review_required,
            ),
            timeout=settings.AI_REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="AI run timed out") from exc
    return run_to_response(run)
