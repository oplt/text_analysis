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
from backend.modules.identity_access.models import User
from backend.modules.platform.api_key_service import ApiKeyService
from backend.modules.platform.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
)

router = APIRouter()


def _api_key_to_response(api_key) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        last_used_at=api_key.last_used_at,
        revoked_at=api_key.revoked_at,
        created_at=api_key.created_at,
    )


@router.get("/api-keys", response_model=PaginatedResponse[ApiKeyResponse])
async def list_api_keys(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ApiKeyService(db)
    keys, total = await service.list_api_keys_for_user(
        current_user,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_api_key_to_response(item) for item in keys],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ApiKeyService(db)
    api_key, plaintext_key = await service.create_api_key_for_user(current_user, payload.name)
    response = _api_key_to_response(api_key)
    return ApiKeyCreateResponse(**response.model_dump(), plaintext_key=plaintext_key)


@router.delete("/api-keys/{api_key_id}", response_model=ApiKeyResponse)
async def revoke_api_key(
    api_key_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ApiKeyService(db)
    api_key = await service.revoke_api_key_for_user(current_user, api_key_id)
    return _api_key_to_response(api_key)
