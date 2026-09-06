from pydantic import BaseModel

from backend.core.schemas import RequestModel


class ProfileResponse(BaseModel):
    user_id: str
    bio: str | None
    avatar_url: str | None
    location: str | None
    website: str | None


class ProfileUpdate(RequestModel):
    bio: str | None = None
    location: str | None = None
    website: str | None = None
