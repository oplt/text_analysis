import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { createApiKey, createWebhook, deleteWebhook, getMySubscription, listApiKeys, listMyFeatureFlags, listSubscriptionPlans, listWebhooks, revokeApiKey, selectMyPlan, testWebhook, updateWebhook, type WebhookEndpoint } from "../../../api/platform";
import { useSnackbar } from "../../../app/snackbarContext";
import { queryKeys } from "../../../config/queryKeys";
import { useMutationErrorToast } from "../../../hooks/useMutationErrorToast";
import { usePlatformMetadata } from "../../../hooks/usePlatformMetadata";

const urlSchema = z.string().url("Enter a valid URL (https://...)");
export function usePlatformView() {
    const client = useQueryClient(); const { showToast } = useSnackbar(); const toastError = useMutationErrorToast();
    const metadataQuery = usePlatformMetadata(); const modules = metadataQuery.data?.enabled_modules ?? [];
    const enabled = { billing: modules.includes("billing"), apiKeys: modules.includes("api_keys"), webhooks: modules.includes("webhooks"), flags: modules.includes("feature_flags") };
    const plansQuery = useQuery({ queryKey: queryKeys.platform.plans, queryFn: listSubscriptionPlans, enabled: enabled.billing });
    const subscriptionQuery = useQuery({ queryKey: queryKeys.platform.subscription, queryFn: getMySubscription, enabled: enabled.billing });
    const apiKeysQuery = useQuery({ queryKey: queryKeys.platform.apiKeys, queryFn: listApiKeys, enabled: enabled.apiKeys });
    const webhooksQuery = useQuery({ queryKey: queryKeys.platform.webhooks, queryFn: listWebhooks, enabled: enabled.webhooks });
    const flagsQuery = useQuery({ queryKey: queryKeys.platform.featureFlags, queryFn: listMyFeatureFlags, enabled: enabled.flags });
    const [apiKeyName, setApiKeyName] = useState(""); const [apiKeyNameError, setApiKeyNameError] = useState<string | null>(null); const [revealedKey, setRevealedKey] = useState<string | null>(null);
    const [webhookDraft, setWebhookDraft] = useState({ target_url: "", description: "", events: "platform.test" }); const [webhookUrlError, setWebhookUrlError] = useState<string | null>(null);
    const [revealedWebhookSecret, setRevealedWebhookSecret] = useState<string | null>(null); const [lastWebhookResult, setLastWebhookResult] = useState("");
    const selectPlanMutation = useMutation({ mutationFn: selectMyPlan, onSuccess: async () => { await client.invalidateQueries({ queryKey: queryKeys.platform.subscription }); showToast({ message: "Subscription updated.", severity: "success" }); }, onError: (e) => toastError(e, "Failed to update subscription.") });
    const createApiKeyMutation = useMutation({ mutationFn: createApiKey, onSuccess: async (data) => { setApiKeyName(""); setRevealedKey(data.plaintext_key); await client.invalidateQueries({ queryKey: queryKeys.platform.apiKeys }); }, onError: (e) => toastError(e, "Failed to create API key.") });
    const revokeApiKeyMutation = useMutation({ mutationFn: revokeApiKey, onSuccess: async () => { await client.invalidateQueries({ queryKey: queryKeys.platform.apiKeys }); showToast({ message: "API key revoked.", severity: "success" }); }, onError: (e) => toastError(e, "Failed to revoke API key.") });
    const createWebhookMutation = useMutation({ mutationFn: createWebhook, onSuccess: async (data) => { setWebhookDraft({ target_url: "", description: "", events: "platform.test" }); setRevealedWebhookSecret(data.signing_secret); await client.invalidateQueries({ queryKey: queryKeys.platform.webhooks }); showToast({ message: "Webhook created.", severity: "success" }); }, onError: (e) => toastError(e, "Failed to create webhook.") });
    const toggleWebhookMutation = useMutation({ mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) => updateWebhook(id, { is_active }), onMutate: async ({ id, is_active }) => { await client.cancelQueries({ queryKey: queryKeys.platform.webhooks }); const previous = client.getQueryData<WebhookEndpoint[]>(queryKeys.platform.webhooks); client.setQueryData<WebhookEndpoint[]>(queryKeys.platform.webhooks, (old) => old?.map((item) => item.id === id ? { ...item, is_active } : item)); return { previous }; }, onError: (e, _v, context) => { if (context?.previous) client.setQueryData(queryKeys.platform.webhooks, context.previous); toastError(e, "Failed to update webhook."); }, onSettled: () => void client.invalidateQueries({ queryKey: queryKeys.platform.webhooks }) });
    const deleteWebhookMutation = useMutation({ mutationFn: deleteWebhook, onSuccess: async () => { await client.invalidateQueries({ queryKey: queryKeys.platform.webhooks }); showToast({ message: "Webhook deleted.", severity: "success" }); }, onError: (e) => toastError(e, "Failed to delete webhook.") });
    const testWebhookMutation = useMutation({ mutationFn: testWebhook, onSuccess: (r) => setLastWebhookResult(r.delivered ? `Delivered with status ${r.status_code}.` : r.error ? `Delivery failed: ${r.error}` : `Received status ${r.status_code}.`), onError: (e) => toastError(e, "Failed to test webhook.") });
    const submitWebhook = () => { const target_url = webhookDraft.target_url.trim(); const parsed = urlSchema.safeParse(target_url); if (!parsed.success) { setWebhookUrlError(parsed.error.issues[0]?.message ?? "Invalid URL"); return; } createWebhookMutation.mutate({ target_url, description: webhookDraft.description.trim() || undefined, events: webhookDraft.events.split(",").map((item) => item.trim()).filter(Boolean) }); };
    return { metadataQuery, enabled, plansQuery, subscriptionQuery, apiKeysQuery, webhooksQuery, flagsQuery, apiKeyName, setApiKeyName, apiKeyNameError, setApiKeyNameError, revealedKey, webhookDraft, setWebhookDraft, webhookUrlError, setWebhookUrlError, revealedWebhookSecret, lastWebhookResult, selectPlanMutation, createApiKeyMutation, revokeApiKeyMutation, createWebhookMutation, toggleWebhookMutation, deleteWebhookMutation, testWebhookMutation, submitWebhook };
}
export type PlatformModel = ReturnType<typeof usePlatformView>;
