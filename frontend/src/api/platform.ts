import { apiFetch, apiFetchItems } from "./client";

export type ModuleCatalogItem = {
    key: string;
    label: string;
    description: string;
    user_visible: boolean;
    enabled: boolean;
};

export type ModulePack = {
    key: string;
    label: string;
    description: string;
    modules: string[];
};

export type PlatformMetadata = {
    app_name: string;
    core_domain_singular: string;
    core_domain_plural: string;
    module_pack: string;
    enabled_modules: string[];
    module_catalog: ModuleCatalogItem[];
    available_module_packs: ModulePack[];
    mfa_enabled: boolean;
};

export type PlatformConfig = PlatformMetadata & {
    module_overrides: Record<string, boolean>;
};

export type SubscriptionPlan = {
    id: string;
    code: string;
    name: string;
    description: string | null;
    price_cents: number;
    interval: string;
    is_active: boolean;
    is_default: boolean;
    features: string[];
    created_at: string;
    updated_at: string;
};

export type UserSubscription = {
    id: string;
    status: string;
    cancel_at_period_end: boolean;
    started_at: string;
    current_period_end: string | null;
    created_at: string;
    updated_at: string;
    plan: SubscriptionPlan;
};

export type ApiKey = {
    id: string;
    name: string;
    key_prefix: string;
    last_used_at: string | null;
    revoked_at: string | null;
    created_at: string;
};

export type CreatedApiKey = ApiKey & {
    plaintext_key: string;
};

export type WebhookEndpoint = {
    id: string;
    target_url: string;
    description: string | null;
    is_active: boolean;
    events: string[];
    last_tested_at: string | null;
    last_response_status: number | null;
    created_at: string;
    updated_at: string;
};

export type CreatedWebhookEndpoint = WebhookEndpoint & {
    signing_secret: string;
};

export type WebhookTestResult = {
    delivered: boolean;
    status_code: number | null;
    response_preview: string | null;
    error: string | null;
};

export type FeatureFlag = {
    id: string;
    key: string;
    name: string;
    description: string | null;
    module_key: string | null;
    is_enabled: boolean;
    rollout_percentage: number;
    updated_at: string;
};

export type EffectiveFeatureFlag = FeatureFlag & {
    effective_enabled: boolean;
};

export type EmailTemplate = {
    id: string;
    key: string;
    name: string;
    subject_template: string;
    html_template: string;
    text_template: string | null;
    is_active: boolean;
    updated_at: string;
};

export async function getPlatformMetadata(): Promise<PlatformMetadata> {
    return apiFetch("/platform/metadata");
}

export async function getPlatformConfig(): Promise<PlatformConfig> {
    return apiFetch("/platform/admin/config");
}

export async function updatePlatformConfig(payload: {
    app_name?: string;
    core_domain_singular?: string;
    core_domain_plural?: string;
    module_pack?: string;
    module_overrides?: Record<string, boolean>;
    mfa_enabled?: boolean;
}): Promise<PlatformConfig> {
    return apiFetch("/platform/admin/config", {
        method: "PUT",
        body: JSON.stringify(payload),
    });
}

export async function listSubscriptionPlans(): Promise<SubscriptionPlan[]> {
    return apiFetch("/platform/billing/plans");
}

export async function getMySubscription(): Promise<UserSubscription | null> {
    return apiFetch("/platform/billing/subscription");
}

export async function selectMyPlan(plan_code: string): Promise<UserSubscription> {
    return apiFetch("/platform/billing/subscription", {
        method: "PUT",
        body: JSON.stringify({ plan_code }),
    });
}

export async function listAdminPlans(): Promise<SubscriptionPlan[]> {
    return apiFetch("/platform/admin/plans");
}

export async function createAdminPlan(payload: {
    code: string;
    name: string;
    description?: string;
    price_cents: number;
    interval: string;
    is_default?: boolean;
    features: string[];
}): Promise<SubscriptionPlan> {
    return apiFetch("/platform/admin/plans", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateAdminPlan(
    planId: string,
    payload: Partial<{
        name: string;
        description: string | null;
        price_cents: number;
        interval: string;
        is_active: boolean;
        is_default: boolean;
        features: string[];
    }>
): Promise<SubscriptionPlan> {
    return apiFetch(`/platform/admin/plans/${planId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function listApiKeys(): Promise<ApiKey[]> {
    return apiFetchItems<ApiKey>("/platform/api-keys");
}

export async function createApiKey(name: string): Promise<CreatedApiKey> {
    return apiFetch("/platform/api-keys", {
        method: "POST",
        body: JSON.stringify({ name }),
    });
}

export async function revokeApiKey(apiKeyId: string): Promise<ApiKey> {
    return apiFetch(`/platform/api-keys/${apiKeyId}`, {
        method: "DELETE",
    });
}

export async function listWebhooks(): Promise<WebhookEndpoint[]> {
    return apiFetchItems<WebhookEndpoint>("/platform/webhooks");
}

export async function createWebhook(payload: {
    target_url: string;
    description?: string;
    events: string[];
}): Promise<CreatedWebhookEndpoint> {
    return apiFetch("/platform/webhooks", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateWebhook(
    webhookId: string,
    payload: Partial<{
        target_url: string;
        description: string | null;
        events: string[];
        is_active: boolean;
    }>
): Promise<WebhookEndpoint> {
    return apiFetch(`/platform/webhooks/${webhookId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deleteWebhook(webhookId: string): Promise<void> {
    return apiFetch(`/platform/webhooks/${webhookId}`, {
        method: "DELETE",
    });
}

export async function testWebhook(webhookId: string): Promise<WebhookTestResult> {
    return apiFetch(`/platform/webhooks/${webhookId}/test`, {
        method: "POST",
    });
}

export async function listMyFeatureFlags(): Promise<EffectiveFeatureFlag[]> {
    return apiFetch("/platform/feature-flags");
}

export async function listAdminFeatureFlags(): Promise<FeatureFlag[]> {
    return apiFetch("/platform/admin/feature-flags");
}

export async function createAdminFeatureFlag(payload: {
    key: string;
    name: string;
    description?: string;
    module_key?: string | null;
    is_enabled?: boolean;
    rollout_percentage?: number;
}): Promise<FeatureFlag> {
    return apiFetch("/platform/admin/feature-flags", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateAdminFeatureFlag(
    featureFlagId: string,
    payload: Partial<{
        name: string;
        description: string | null;
        module_key: string | null;
        is_enabled: boolean;
        rollout_percentage: number;
    }>
): Promise<FeatureFlag> {
    return apiFetch(`/platform/admin/feature-flags/${featureFlagId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function listAdminEmailTemplates(): Promise<EmailTemplate[]> {
    return apiFetch("/platform/admin/email-templates");
}

export async function createAdminEmailTemplate(payload: {
    key: string;
    name: string;
    subject_template: string;
    html_template: string;
    text_template?: string | null;
    is_active?: boolean;
}): Promise<EmailTemplate> {
    return apiFetch("/platform/admin/email-templates", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateAdminEmailTemplate(
    templateId: string,
    payload: Partial<{
        name: string;
        subject_template: string;
        html_template: string;
        text_template: string | null;
        is_active: boolean;
    }>
): Promise<EmailTemplate> {
    return apiFetch(`/platform/admin/email-templates/${templateId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}
