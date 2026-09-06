from typing import TypedDict


class ModuleCatalogDefinition(TypedDict):
    key: str
    label: str
    description: str
    user_visible: bool


class ModulePackDefinition(TypedDict):
    label: str
    description: str
    modules: list[str]


class PlanDefinition(TypedDict):
    code: str
    name: str
    description: str
    price_cents: int
    interval: str
    is_default: bool
    features_json: list[str]


class FeatureFlagDefinition(TypedDict):
    key: str
    name: str
    description: str
    module_key: str
    is_enabled: bool
    rollout_percentage: int


class EmailTemplateDefinition(TypedDict):
    key: str
    name: str
    subject_template: str
    html_template: str
    text_template: str
    is_active: bool


MODULE_CATALOG: tuple[ModuleCatalogDefinition, ...] = (
    {
        "key": "ai",
        "label": "AI",
        "description": "Prompt ops, document retrieval, reviews, and evaluations.",
        "user_visible": True,
    },
    {
        "key": "billing",
        "label": "Billing",
        "description": "Plan catalog and subscription management.",
        "user_visible": True,
    },
    {
        "key": "api_keys",
        "label": "API Keys",
        "description": "User-managed credentials for integrations and automation.",
        "user_visible": True,
    },
    {
        "key": "webhooks",
        "label": "Webhooks",
        "description": "Outbound event delivery to external systems.",
        "user_visible": True,
    },
    {
        "key": "feature_flags",
        "label": "Feature Flags",
        "description": "Runtime rollout controls for features and experiments.",
        "user_visible": True,
    },
    {
        "key": "email_templates",
        "label": "Email Templates",
        "description": "Customizable transactional email content.",
        "user_visible": False,
    },
)

MODULE_PACKS: dict[str, ModulePackDefinition] = {
    "lean_saas": {
        "label": "Lean SaaS",
        "description": "Billing plus integration basics for a straightforward SaaS clone.",
        "modules": ["billing", "api_keys", "feature_flags", "ai"],
    },
    "automation_suite": {
        "label": "Automation Suite",
        "description": "API keys, webhooks, flags, and templates for workflow-driven products.",
        "modules": ["api_keys", "webhooks", "feature_flags", "email_templates", "ai"],
    },
    "client_portal": {
        "label": "Client Portal",
        "description": "Subscription-led portal with flags and email customization.",
        "modules": ["billing", "feature_flags", "email_templates", "ai"],
    },
    "full_platform": {
        "label": "Full Platform",
        "description": "Enable every optional platform module.",
        "modules": [item["key"] for item in MODULE_CATALOG],
    },
}

DEFAULT_PLANS: tuple[PlanDefinition, ...] = (
    {
        "code": "free",
        "name": "Free",
        "description": "Starter plan for early validation and internal testing.",
        "price_cents": 0,
        "interval": "month",
        "is_default": True,
        "features_json": ["core_access", "community_support"],
    },
    {
        "code": "growth",
        "name": "Growth",
        "description": "Operational plan for customer-facing launches and smaller teams.",
        "price_cents": 4900,
        "interval": "month",
        "is_default": False,
        "features_json": [
            "core_access",
            "priority_support",
            "platform_webhooks",
            "platform_api_keys",
        ],
    },
    {
        "code": "enterprise",
        "name": "Enterprise",
        "description": "Premium plan for large deployments and white-label programs.",
        "price_cents": 19900,
        "interval": "month",
        "is_default": False,
        "features_json": [
            "core_access",
            "priority_support",
            "platform_webhooks",
            "platform_api_keys",
            "advanced_templates",
        ],
    },
)

DEFAULT_FEATURE_FLAGS: tuple[FeatureFlagDefinition, ...] = (
    {
        "key": "beta_dashboard",
        "name": "Beta Dashboard",
        "description": "Enable next-generation dashboard components.",
        "module_key": "feature_flags",
        "is_enabled": True,
        "rollout_percentage": 100,
    },
    {
        "key": "advanced_billing_controls",
        "name": "Advanced Billing Controls",
        "description": "Expose richer billing controls and internal finance actions.",
        "module_key": "billing",
        "is_enabled": False,
        "rollout_percentage": 0,
    },
    {
        "key": "webhook_replay",
        "name": "Webhook Replay",
        "description": "Prepare replay tooling for webhook troubleshooting workflows.",
        "module_key": "webhooks",
        "is_enabled": False,
        "rollout_percentage": 0,
    },
)

DEFAULT_EMAIL_TEMPLATES: tuple[EmailTemplateDefinition, ...] = (
    {
        "key": "auth.verify_email",
        "name": "Verify Email",
        "subject_template": "{{app_name}}: verify your email address",
        "html_template": (
            "<p>Thanks for joining {{app_name}}.</p>"
            '<p>Verify your email by opening <a href="{{action_url}}">this link</a>.</p>'
            "<p>If you did not create an account for"
            " {{recipient_email}}, you can ignore this email.</p>"
        ),
        "text_template": (
            "Thanks for joining {{app_name}}.\n"
            "Verify your email by opening: {{action_url}}\n"
            "If you did not create an account for {{recipient_email}}, ignore this email."
        ),
        "is_active": True,
    },
    {
        "key": "auth.reset_password",
        "name": "Reset Password",
        "subject_template": "{{app_name}}: reset your password",
        "html_template": (
            "<p>We received a password reset request for {{recipient_email}}.</p>"
            '<p>Use <a href="{{action_url}}">this secure link</a> to choose a new password.</p>'
            "<p>If you did not request this, you can ignore this email.</p>"
        ),
        "text_template": (
            "We received a password reset request for {{recipient_email}}.\n"
            "Use this secure link to choose a new password: {{action_url}}\n"
            "If you did not request this, you can ignore this email."
        ),
        "is_active": True,
    },
)

SETTING_APP_NAME = "platform.app_name"
SETTING_CORE_DOMAIN_SINGULAR = "platform.core_domain_singular"
SETTING_CORE_DOMAIN_PLURAL = "platform.core_domain_plural"
SETTING_MODULE_PACK = "platform.module_pack"
SETTING_MODULE_OVERRIDE_PREFIX = "platform.module_override."
SETTING_MFA_ENABLED = "platform.mfa_enabled"
