from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.admin import get_admin_user
from backend.api.deps.db import get_db
from backend.modules.audit.request_logging import log_request_audit_event
from backend.modules.identity_access.models import User
from backend.modules.settings.schemas import (
    ConfigSettingsResponse,
    ConfigSettingsUpdateRequest,
    DatabaseSettingCreate,
    DatabaseSettingResponse,
    DatabaseSettingUpdate,
)
from backend.modules.settings.service import SettingsService

router = APIRouter()


@router.get("/config", response_model=ConfigSettingsResponse)
async def get_config_settings(_: User = Depends(get_admin_user)):
    return await SettingsService.list_config_entries()


@router.put("/config", response_model=ConfigSettingsResponse)
async def update_config_settings(
    payload: ConfigSettingsUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    response = await SettingsService.update_config_entries(payload.items)
    await log_request_audit_event(
        db,
        request,
        action="admin.config_updated",
        actor_user_id=admin.id,
        resource_type="config",
    )
    await db.commit()
    return response


@router.get("/database", response_model=list[DatabaseSettingResponse])
async def list_database_settings(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    service = SettingsService(db)
    return await service.list_database_settings()


@router.post("/database", response_model=DatabaseSettingResponse, status_code=201)
async def create_database_setting(
    payload: DatabaseSettingCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = SettingsService(db)
    setting = await service.create_database_setting(payload.key, payload.value, payload.description)
    await log_request_audit_event(
        db,
        request,
        action="admin.database_setting_created",
        actor_user_id=admin.id,
        resource_type="app_setting",
        resource_id=setting.id,
    )
    await db.commit()
    return setting


@router.patch("/database/{setting_id}", response_model=DatabaseSettingResponse)
async def update_database_setting(
    setting_id: str,
    payload: DatabaseSettingUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = SettingsService(db)
    setting = await service.update_database_setting(
        setting_id, payload.model_dump(exclude_unset=True)
    )
    await log_request_audit_event(
        db,
        request,
        action="admin.database_setting_updated",
        actor_user_id=admin.id,
        resource_type="app_setting",
        resource_id=setting.id,
    )
    await db.commit()
    return setting


@router.delete("/database/{setting_id}", status_code=204)
async def delete_database_setting(
    setting_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    service = SettingsService(db)
    await service.delete_database_setting(setting_id)
    await log_request_audit_event(
        db,
        request,
        action="admin.database_setting_deleted",
        actor_user_id=admin.id,
        resource_type="app_setting",
        resource_id=setting_id,
    )
    await db.commit()
    return Response(status_code=204)
