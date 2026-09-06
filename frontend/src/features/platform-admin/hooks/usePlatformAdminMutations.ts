import type { Dispatch, SetStateAction } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
    createAdminEmailTemplate,
    createAdminFeatureFlag,
    createAdminPlan,
    updateAdminEmailTemplate,
    updateAdminFeatureFlag,
    updateAdminPlan,
    updatePlatformConfig,
} from "../../../api/platform";
import { useSnackbar } from "../../../app/snackbarContext";
import { queryKeys } from "../../../config/queryKeys";
import { useMutationErrorToast } from "../../../hooks/useMutationErrorToast";

export type PlanDraft = {
    name: string; description: string; price_cents: string; interval: string;
    is_active: boolean; is_default: boolean; features: string;
};
export type FlagDraft = {
    name: string; description: string; module_key: string;
    is_enabled: boolean; rollout_percentage: string;
};
export type TemplateDraft = {
    name: string; subject_template: string; html_template: string;
    text_template: string; is_active: boolean;
};
export type NewPlanDraft = Omit<PlanDraft, "is_active"> & { code: string };
export type NewFlagDraft = FlagDraft & { key: string };
export type NewTemplateDraft = TemplateDraft & { key: string };

export const EMPTY_NEW_PLAN: NewPlanDraft = {
    code: "", name: "", description: "", price_cents: "0", interval: "month",
    is_default: false, features: "",
};
export const EMPTY_NEW_FLAG: NewFlagDraft = {
    key: "", name: "", description: "", module_key: "", is_enabled: false,
    rollout_percentage: "100",
};
export const EMPTY_NEW_TEMPLATE: NewTemplateDraft = {
    key: "", name: "", subject_template: "", html_template: "", text_template: "",
    is_active: true,
};

export async function invalidatePlatformAdminCapability(
    queryClient: ReturnType<typeof useQueryClient>,
    capability: "all" | "plans" | "featureFlags" | "emailTemplates",
) {
    const queryKey = capability === "all"
        ? queryKeys.platform.all
        : queryKeys.platform.admin[capability];
    await queryClient.invalidateQueries({ queryKey });
}

export function usePlatformAdminMutations({
    setNewPlan, setNewFlag, setNewTemplate,
}: {
    setNewPlan: Dispatch<SetStateAction<NewPlanDraft>>;
    setNewFlag: Dispatch<SetStateAction<NewFlagDraft>>;
    setNewTemplate: Dispatch<SetStateAction<NewTemplateDraft>>;
}) {
    const queryClient = useQueryClient();
    const { showToast } = useSnackbar();
    const toastMutationError = useMutationErrorToast();
    const success = async (capability: "all" | "plans" | "featureFlags" | "emailTemplates", message: string) => {
        await invalidatePlatformAdminCapability(queryClient, capability);
        showToast({ message, severity: "success" });
    };

    const saveConfigMutation = useMutation({
        mutationFn: updatePlatformConfig,
        onSuccess: () => success("all", "Platform configuration updated."),
        onError: (error) => toastMutationError(error, "Failed to save platform configuration."),
    });
    const createPlanMutation = useMutation({
        mutationFn: createAdminPlan,
        onSuccess: async () => { setNewPlan(EMPTY_NEW_PLAN); await success("plans", "Plan created."); },
        onError: (error) => toastMutationError(error, "Failed to create plan."),
    });
    const updatePlanMutation = useMutation({
        mutationFn: ({ id, draft }: { id: string; draft: PlanDraft }) => updateAdminPlan(id, {
            name: draft.name, description: draft.description || null,
            price_cents: Number(draft.price_cents), interval: draft.interval,
            is_active: draft.is_active, is_default: draft.is_default,
            features: draft.features.split(",").map((item) => item.trim()).filter(Boolean),
        }),
        onSuccess: () => success("plans", "Plan updated."),
        onError: (error) => toastMutationError(error, "Failed to update plan."),
    });
    const createFlagMutation = useMutation({
        mutationFn: createAdminFeatureFlag,
        onSuccess: async () => { setNewFlag(EMPTY_NEW_FLAG); await success("featureFlags", "Feature flag created."); },
        onError: (error) => toastMutationError(error, "Failed to create feature flag."),
    });
    const updateFlagMutation = useMutation({
        mutationFn: ({ id, draft }: { id: string; draft: FlagDraft }) => updateAdminFeatureFlag(id, {
            name: draft.name, description: draft.description || null,
            module_key: draft.module_key || null, is_enabled: draft.is_enabled,
            rollout_percentage: Number(draft.rollout_percentage),
        }),
        onSuccess: () => success("featureFlags", "Feature flag updated."),
        onError: (error) => toastMutationError(error, "Failed to update feature flag."),
    });
    const createTemplateMutation = useMutation({
        mutationFn: createAdminEmailTemplate,
        onSuccess: async () => { setNewTemplate(EMPTY_NEW_TEMPLATE); await success("emailTemplates", "Email template created."); },
        onError: (error) => toastMutationError(error, "Failed to create email template."),
    });
    const updateTemplateMutation = useMutation({
        mutationFn: ({ id, draft }: { id: string; draft: TemplateDraft }) => updateAdminEmailTemplate(id, {
            name: draft.name, subject_template: draft.subject_template,
            html_template: draft.html_template, text_template: draft.text_template || null,
            is_active: draft.is_active,
        }),
        onSuccess: () => success("emailTemplates", "Email template updated."),
        onError: (error) => toastMutationError(error, "Failed to update email template."),
    });

    return { saveConfigMutation, createPlanMutation, updatePlanMutation, createFlagMutation,
        updateFlagMutation, createTemplateMutation, updateTemplateMutation };
}
