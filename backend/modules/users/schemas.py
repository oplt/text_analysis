from datetime import datetime

from pydantic import BaseModel, Field

from backend.core.schemas import RequestModel


class UserProfileUpdate(RequestModel):
    full_name: str | None = None


class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_verified: bool
    mfa_enabled: bool


class UserDirectoryResponse(BaseModel):
    id: str
    email: str
    full_name: str | None


class PasswordChangeRequest(RequestModel):
    current_password: str
    new_password: str = Field(min_length=8)


class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    expires_at: datetime
