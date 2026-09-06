import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.admin import get_admin_user
from backend.api.deps.db import get_db
from backend.modules.admin.schemas import (
    AdminUserListResponse,
    AdminUserResponse,
    AdminUserStatusUpdate,
    AuditLogResponse,
    MetricsResponse,
)
from backend.modules.audit.repository import AuditRepository
from backend.modules.identity_access.models import User
from backend.modules.identity_access.repository import IdentityRepository
from backend.modules.notifications.models import Notification

router = APIRouter()


def _user_to_response(user: User) -> AdminUserResponse:
    roles = ["user"]
    if user.is_admin:
        roles.append("admin")

    return AdminUserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        roles=roles,
        is_active=user.is_active,
        is_verified=user.is_verified,
        is_admin=user.is_admin,
        mfa_enabled=user.mfa_enabled,
        created_at=user.created_at,
    )


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None, min_length=1),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    filters = []
    if search:
        search_term = f"%{search.strip()}%"
        filters.append(
            or_(
                User.email.ilike(search_term),
                User.full_name.ilike(search_term),
            )
        )

    total_query = select(func.count()).select_from(User)
    data_query = select(User).order_by(User.created_at.desc())

    if filters:
        total_query = total_query.where(*filters)
        data_query = data_query.where(*filters)

    total = await db.scalar(total_query)
    result = await db.execute(
        data_query.offset((page - 1) * page_size).limit(page_size)
    )

    return AdminUserListResponse(
        items=[_user_to_response(user) for user in result.scalars().all()],
        total=total or 0,
        page=page,
        page_size=page_size,
    )


@router.get("/users/{user_id}", response_model=AdminUserResponse)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    repo = IdentityRepository(db)
    user = await repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_response(user)


@router.patch("/users/{user_id}/status", response_model=AdminUserResponse)
async def update_user_status(
    user_id: str,
    payload: AdminUserStatusUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own status")
    repo = IdentityRepository(db)
    user = await repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = payload.is_active
    audit_repo = AuditRepository(db)
    await audit_repo.log(
        action="admin.update_user_status",
        user_id=admin.id,
        resource_type="user",
        resource_id=user.id,
        metadata={"is_active": payload.is_active},
    )
    await db.commit()
    return _user_to_response(user)


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    repo = AuditRepository(db)
    logs = await repo.list_recent(limit=100)
    return [
        AuditLogResponse(
            id=log.id, user_id=log.user_id, action=log.action,
            resource_type=log.resource_type, resource_id=log.resource_id,
            ip_address=log.ip_address, created_at=log.created_at,
        )
        for log in logs
    ]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_admin_user),
):
    total, verified, active, notifs = await asyncio.gather(
        db.scalar(select(func.count()).select_from(User)),
        db.scalar(
            select(func.count()).select_from(User).where(User.is_verified.is_(True))
        ),
        db.scalar(select(func.count()).select_from(User).where(User.is_active.is_(True))),
        db.scalar(select(func.count()).select_from(Notification)),
    )

    return MetricsResponse(
        total_users=total or 0,
        verified_users=verified or 0,
        active_users=active or 0,
        total_notifications=notifs or 0,
    )
