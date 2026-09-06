from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.core.schemas import RequestModel

ENV_KEY_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
SETTING_KEY_PATTERN = r"^[A-Za-z0-9_.:-]+$"


class ConfigEntryResponse(BaseModel):
    key: str
    value: str
    value_type: str
    description: str | None = None
    requires_restart: bool = False
    is_custom: bool = False
    is_secret: bool = False


class ConfigSettingsResponse(BaseModel):
    items: list[ConfigEntryResponse]
    notice: str


class ConfigEntryUpdate(RequestModel):
    key: str = Field(min_length=1, max_length=128, pattern=ENV_KEY_PATTERN)
    value: str = ""


class ConfigSettingsUpdateRequest(RequestModel):
    items: list[ConfigEntryUpdate]


class DatabaseSettingCreate(RequestModel):
    key: str = Field(min_length=1, max_length=128, pattern=SETTING_KEY_PATTERN)
    value: str = ""
    description: str | None = Field(default=None, max_length=1000)


class DatabaseSettingUpdate(RequestModel):
    value: str | None = None
    description: str | None = Field(default=None, max_length=1000)


class DatabaseSettingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    value: str
    description: str | None
    updated_at: datetime
