from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.core.schemas import RequestModel

MODULE_KEY_PATTERN = r"^[a-z0-9_]+$"


class ModuleCatalogItem(BaseModel):
    key: str
    label: str
    description: str
    user_visible: bool
    enabled: bool


class ModulePackResponse(BaseModel):
    key: str
    label: str
    description: str
    modules: list[str]


class PlatformMetadataResponse(BaseModel):
    app_name: str
    core_domain_singular: str
    core_domain_plural: str
    module_pack: str
    enabled_modules: list[str]
    module_catalog: list[ModuleCatalogItem]
    available_module_packs: list[ModulePackResponse]
    mfa_enabled: bool = False


class PlatformConfigResponse(PlatformMetadataResponse):
    module_overrides: dict[str, bool]


class PlatformConfigUpdateRequest(RequestModel):
    app_name: str | None = Field(default=None, min_length=1, max_length=255)
    core_domain_singular: str | None = Field(default=None, min_length=1, max_length=64)
    core_domain_plural: str | None = Field(default=None, min_length=1, max_length=64)
    module_pack: str | None = None
    module_overrides: dict[str, bool] | None = None
    mfa_enabled: bool | None = None


class SubscriptionPlanCreate(RequestModel):
    code: str = Field(min_length=2, max_length=64, pattern=MODULE_KEY_PATTERN)
    name: str = Field(min_length=2, max_length=255)
    description: str | None = None
    price_cents: int = Field(ge=0)
    interval: str = Field(min_length=3, max_length=32)
    is_default: bool = False
    features: list[str] = Field(default_factory=list)


class SubscriptionPlanUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = None
    price_cents: int | None = Field(default=None, ge=0)
    interval: str | None = Field(default=None, min_length=3, max_length=32)
    is_active: bool | None = None
    is_default: bool | None = None
    features: list[str] | None = None


class SubscriptionPlanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str
    name: str
    description: str | None
    price_cents: int
    interval: str
    is_active: bool
    is_default: bool
    features: list[str]
    created_at: datetime
    updated_at: datetime


class UserSubscriptionResponse(BaseModel):
    id: str
    status: str
    cancel_at_period_end: bool
    started_at: datetime
    current_period_end: datetime | None
    created_at: datetime
    updated_at: datetime
    plan: SubscriptionPlanResponse


class SubscriptionSelectionRequest(RequestModel):
    plan_code: str = Field(min_length=2, max_length=64, pattern=MODULE_KEY_PATTERN)


class ApiKeyCreateRequest(RequestModel):
    name: str = Field(min_length=2, max_length=255)


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    key_prefix: str
    last_used_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class ApiKeyCreateResponse(ApiKeyResponse):
    plaintext_key: str


class WebhookEndpointCreate(RequestModel):
    target_url: HttpUrl
    description: str | None = None
    events: list[str] = Field(default_factory=list)


class WebhookEndpointUpdate(RequestModel):
    target_url: HttpUrl | None = None
    description: str | None = None
    events: list[str] | None = None
    is_active: bool | None = None


class WebhookEndpointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    target_url: str
    description: str | None
    is_active: bool
    events: list[str]
    last_tested_at: datetime | None
    last_response_status: int | None
    created_at: datetime
    updated_at: datetime


class WebhookEndpointCreateResponse(WebhookEndpointResponse):
    signing_secret: str


class WebhookTestResponse(BaseModel):
    delivered: bool
    status_code: int | None = None
    response_preview: str | None = None
    error: str | None = None


class FeatureFlagCreate(RequestModel):
    key: str = Field(min_length=2, max_length=128, pattern=MODULE_KEY_PATTERN)
    name: str = Field(min_length=2, max_length=255)
    description: str | None = None
    module_key: str | None = Field(default=None, pattern=MODULE_KEY_PATTERN)
    is_enabled: bool = False
    rollout_percentage: int = Field(default=100, ge=0, le=100)


class FeatureFlagUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = None
    module_key: str | None = Field(default=None, pattern=MODULE_KEY_PATTERN)
    is_enabled: bool | None = None
    rollout_percentage: int | None = Field(default=None, ge=0, le=100)


class FeatureFlagResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    name: str
    description: str | None
    module_key: str | None
    is_enabled: bool
    rollout_percentage: int
    updated_at: datetime


class EffectiveFeatureFlagResponse(FeatureFlagResponse):
    effective_enabled: bool


class EmailTemplateCreate(RequestModel):
    key: str = Field(min_length=2, max_length=128)
    name: str = Field(min_length=2, max_length=255)
    subject_template: str = Field(min_length=1, max_length=500)
    html_template: str = Field(min_length=1)
    text_template: str | None = None
    is_active: bool = True


class EmailTemplateUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    subject_template: str | None = Field(default=None, min_length=1, max_length=500)
    html_template: str | None = None
    text_template: str | None = None
    is_active: bool | None = None


class EmailTemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    name: str
    subject_template: str
    html_template: str
    text_template: str | None
    is_active: bool
    updated_at: datetime
