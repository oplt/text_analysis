from datetime import datetime

from pydantic import BaseModel, EmailStr

from backend.core.schemas import RequestModel


class AdminUserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None
    roles: list[str]
    is_active: bool
    is_verified: bool
    is_admin: bool
    mfa_enabled: bool
    created_at: datetime


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]
    total: int
    page: int
    page_size: int


class AdminUserStatusUpdate(RequestModel):
    is_active: bool


class AuditLogResponse(BaseModel):
    id: str
    user_id: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    ip_address: str | None
    created_at: datetime


class MetricsResponse(BaseModel):
    total_users: int
    verified_users: int
    active_users: int
    total_notifications: int
