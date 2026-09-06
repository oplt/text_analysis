from datetime import datetime

from pydantic import BaseModel

from backend.core.schemas import RequestModel


class NotificationResponse(BaseModel):
    id: str
    type: str
    title: str
    body: str | None
    is_read: bool
    created_at: datetime


class NotificationPreferenceResponse(BaseModel):
    email_enabled: bool
    push_enabled: bool
    marketing_enabled: bool


class NotificationPreferenceUpdate(RequestModel):
    email_enabled: bool | None = None
    push_enabled: bool | None = None
    marketing_enabled: bool | None = None
