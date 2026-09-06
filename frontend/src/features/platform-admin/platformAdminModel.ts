import type { EmailTemplate, FeatureFlag, PlatformConfig, SubscriptionPlan } from "../../api/platform";
import type { FlagDraft, PlanDraft, TemplateDraft } from "./hooks/usePlatformAdminMutations";

export function buildConfigDraft(config: PlatformConfig) {
    return { app_name: config.app_name, core_domain_singular: config.core_domain_singular,
        core_domain_plural: config.core_domain_plural, module_pack: config.module_pack,
        module_states: Object.fromEntries(config.module_catalog.map((item) => [item.key, item.enabled])),
        mfa_enabled: config.mfa_enabled };
}
export type ConfigDraft = ReturnType<typeof buildConfigDraft>;

export function buildPlanDrafts(plans: SubscriptionPlan[]) {
    return Object.fromEntries(plans.map((plan) => [plan.id, { name: plan.name,
        description: plan.description ?? "", price_cents: String(plan.price_cents), interval: plan.interval,
        is_active: plan.is_active, is_default: plan.is_default, features: plan.features.join(", ") }])) as Record<string, PlanDraft>;
}
export function buildFlagDrafts(flags: FeatureFlag[]) {
    return Object.fromEntries(flags.map((flag) => [flag.id, { name: flag.name,
        description: flag.description ?? "", module_key: flag.module_key ?? "", is_enabled: flag.is_enabled,
        rollout_percentage: String(flag.rollout_percentage) }])) as Record<string, FlagDraft>;
}
export function buildTemplateDrafts(templates: EmailTemplate[]) {
    return Object.fromEntries(templates.map((template) => [template.id, { name: template.name,
        subject_template: template.subject_template, html_template: template.html_template,
        text_template: template.text_template ?? "", is_active: template.is_active }])) as Record<string, TemplateDraft>;
}

export function platformAdminKey(config: PlatformConfig, plans: SubscriptionPlan[], flags: FeatureFlag[], templates: EmailTemplate[]) {
    return [config.app_name, config.module_pack, config.module_catalog.map((item) => `${item.key}:${item.enabled}`).join("|"),
        plans.map((item) => `${item.id}:${item.updated_at}`).join("|"), flags.map((item) => `${item.id}:${item.updated_at}`).join("|"),
        templates.map((item) => `${item.id}:${item.updated_at}`).join("|")].join("::");
}
