from fastapi import APIRouter, Depends, Response
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
from backend.modules.platform.schemas import (
    WebhookEndpointCreate,
    WebhookEndpointCreateResponse,
    WebhookEndpointResponse,
    WebhookEndpointUpdate,
    WebhookTestResponse,
)
from backend.modules.platform.webhook_service import WebhookService

router = APIRouter()


def _webhook_to_response(webhook) -> WebhookEndpointResponse:
    return WebhookEndpointResponse(
        id=webhook.id,
        target_url=webhook.target_url,
        description=webhook.description,
        is_active=webhook.is_active,
        events=webhook.events_json,
        last_tested_at=webhook.last_tested_at,
        last_response_status=webhook.last_response_status,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
    )


@router.get("/webhooks", response_model=PaginatedResponse[WebhookEndpointResponse])
async def list_webhooks(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = WebhookService(db)
    webhooks, total = await service.list_webhooks_for_user(
        current_user,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_webhook_to_response(item) for item in webhooks],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/webhooks", response_model=WebhookEndpointCreateResponse, status_code=201)
async def create_webhook(
    payload: WebhookEndpointCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = WebhookService(db)
    webhook = await service.create_webhook_for_user(
        current_user,
        target_url=str(payload.target_url),
        description=payload.description,
        events=payload.events,
    )
    response = _webhook_to_response(webhook)
    return WebhookEndpointCreateResponse(**response.model_dump(), signing_secret=webhook.secret)


@router.patch("/webhooks/{webhook_id}", response_model=WebhookEndpointResponse)
async def update_webhook(
    webhook_id: str,
    payload: WebhookEndpointUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = WebhookService(db)
    updates = payload.model_dump(exclude_unset=True)
    if "target_url" in updates and updates["target_url"] is not None:
        updates["target_url"] = str(updates["target_url"])
    webhook = await service.update_webhook_for_user(current_user, webhook_id, updates)
    return _webhook_to_response(webhook)


@router.delete("/webhooks/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = WebhookService(db)
    await service.delete_webhook_for_user(current_user, webhook_id)
    return Response(status_code=204)


@router.post("/webhooks/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = WebhookService(db)
    return WebhookTestResponse(**(await service.test_webhook_for_user(current_user, webhook_id)))
