from backend.modules.platform.api_key_service import ApiKeyService
from backend.modules.platform.billing_service import BillingService
from backend.modules.platform.email_template_service import EmailTemplateService
from backend.modules.platform.feature_flag_service import FeatureFlagService
from backend.modules.platform.webhook_service import WebhookService


class PlatformService(
    BillingService,
    ApiKeyService,
    WebhookService,
    FeatureFlagService,
    EmailTemplateService,
):
    """Compatibility facade for platform capability services."""
