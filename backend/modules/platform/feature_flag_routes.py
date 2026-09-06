from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.admin import get_admin_user
from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.audit.request_logging import log_request_audit_event
from backend.modules.identity_access.models import User
from backend.modules.platform.feature_flag_service import FeatureFlagService
from backend.modules.platform.schemas import (
    EffectiveFeatureFlagResponse,
    FeatureFlagCreate,
    FeatureFlagResponse,
    FeatureFlagUpdate,
)

router = APIRouter()


def _feature_flag_to_response(flag) -> FeatureFlagResponse:
    return FeatureFlagResponse(
        id=flag.id,
        key=flag.key,
        name=flag.name,
        description=flag.description,
        module_key=flag.module_key,
        is_enabled=flag.is_enabled,
        rollout_percentage=flag.rollout_percentage,
        updated_at=flag.updated_at,
    )


@router.get("/feature-flags", response_model=list[EffectiveFeatureFlagResponse])
async def list_effective_feature_flags(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = FeatureFlagService(db)
    return await service.list_effective_feature_flags_for_user(current_user)


@router.get("/admin/feature-flags", response_model=list[FeatureFlagResponse])
async def list_feature_flags(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    service = FeatureFlagService(db)
    return [_feature_flag_to_response(flag) for flag in await service.list_feature_flags()]


@router.post("/admin/feature-flags", response_model=FeatureFlagResponse, status_code=201)
async def create_feature_flag(
    payload: FeatureFlagCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = FeatureFlagService(db)
    flag = await service.create_feature_flag(payload.model_dump())
    await log_request_audit_event(
        db,
        request,
        action="admin.feature_flag_created",
        actor_user_id=admin.id,
        resource_type="feature_flag",
        resource_id=flag.id,
        metadata={"key": flag.key},
    )
    await db.commit()
    return _feature_flag_to_response(flag)


@router.patch("/admin/feature-flags/{feature_flag_id}", response_model=FeatureFlagResponse)
async def update_feature_flag(
    feature_flag_id: str,
    payload: FeatureFlagUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = FeatureFlagService(db)
    flag = await service.update_feature_flag(
        feature_flag_id, payload.model_dump(exclude_unset=True)
    )
    await log_request_audit_event(
        db,
        request,
        action="admin.feature_flag_updated",
        actor_user_id=admin.id,
        resource_type="feature_flag",
        resource_id=flag.id,
    )
    await db.commit()
    return _feature_flag_to_response(flag)
