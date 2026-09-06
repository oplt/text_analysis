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
from backend.modules.ai.prompt_service import AiPromptService
from backend.modules.ai.schemas import (
    AiPromptTemplateCreate,
    AiPromptTemplateResponse,
    AiPromptTemplateUpdate,
    AiPromptVersionCreate,
    AiPromptVersionResponse,
    AiPromptVersionUpdate,
)
from backend.modules.ai.serializers import _prompt_template_to_response, _prompt_version_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/prompts", response_model=PaginatedResponse[AiPromptTemplateResponse])
async def list_prompt_templates(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    templates, total = await service.list_prompt_templates(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [_prompt_template_to_response(item) for item in templates],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/prompts", response_model=AiPromptTemplateResponse, status_code=201)
async def create_prompt_template(
    payload: AiPromptTemplateCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    template = await service.create_prompt_template(
        current_user, payload.key, payload.name, payload.description
    )
    return _prompt_template_to_response(template)


@router.patch("/prompts/{template_id}", response_model=AiPromptTemplateResponse)
async def update_prompt_template(
    template_id: str,
    payload: AiPromptTemplateUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    template = await service.update_prompt_template(
        current_user,
        template_id,
        payload.model_dump(exclude_unset=True),
    )
    return _prompt_template_to_response(template)


@router.get(
    "/prompts/{template_id}/versions",
    response_model=PaginatedResponse[AiPromptVersionResponse],
)
async def list_prompt_versions(
    template_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    versions, total = await service.list_prompt_versions(
        current_user,
        template_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_prompt_version_to_response(item) for item in versions],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post(
    "/prompts/{template_id}/versions",
    response_model=AiPromptVersionResponse,
    status_code=201,
)
async def create_prompt_version(
    template_id: str,
    payload: AiPromptVersionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    version = await service.create_prompt_version(current_user, template_id, payload.model_dump())
    return _prompt_version_to_response(version)


@router.patch(
    "/prompts/{template_id}/versions/{version_id}",
    response_model=AiPromptVersionResponse,
)
async def update_prompt_version(
    template_id: str,
    version_id: str,
    payload: AiPromptVersionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiPromptService(db)
    version = await service.update_prompt_version(
        current_user,
        template_id,
        version_id,
        payload.model_dump(exclude_unset=True),
    )
    return _prompt_version_to_response(version)
