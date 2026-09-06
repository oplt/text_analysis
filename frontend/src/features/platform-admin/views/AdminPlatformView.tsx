import { useQuery } from "@tanstack/react-query";
import { Skeleton, Stack } from "@mui/material";
import { getPlatformConfig, listAdminEmailTemplates, listAdminFeatureFlags, listAdminPlans } from "../../../api/platform";
import { queryKeys } from "../../../config/queryKeys";
import { AdminSettingsTabs } from "../../../components/layout/AdminSettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { AdminPlatformContent } from "../components/AdminPlatformContent";
import { platformAdminKey } from "../platformAdminModel";

export default function AdminPlatformPage() {
    const config = useQuery({ queryKey: queryKeys.platform.admin.config, queryFn: getPlatformConfig });
    const plans = useQuery({ queryKey: queryKeys.platform.admin.plans, queryFn: listAdminPlans });
    const flags = useQuery({ queryKey: queryKeys.platform.admin.featureFlags, queryFn: listAdminFeatureFlags });
    const templates = useQuery({ queryKey: queryKeys.platform.admin.emailTemplates, queryFn: listAdminEmailTemplates });
    if ([config, plans, flags, templates].some((query) => query.isLoading)) return <PageShell maxWidth="xl"><Stack spacing={2}><Skeleton variant="rounded" height={180} sx={{ borderRadius: 6 }} /><Skeleton variant="rounded" height={240} sx={{ borderRadius: 6 }} /><Skeleton variant="rounded" height={240} sx={{ borderRadius: 6 }} /></Stack></PageShell>;
    if (!config.data || !plans.data || !flags.data || !templates.data) return <PageShell maxWidth="xl"><AdminSettingsTabs /><Stack spacing={2} sx={{ mt: 2 }}>
        {!config.data && <QueryErrorAlert error={config.error ?? new Error("Failed to load platform configuration.")} fallback="Failed to load platform configuration." title="Configuration" onRetry={() => void config.refetch()} />}
        {!plans.data && <QueryErrorAlert error={plans.error ?? new Error("Failed to load subscription plans.")} fallback="Failed to load subscription plans." title="Plans" onRetry={() => void plans.refetch()} />}
        {!flags.data && <QueryErrorAlert error={flags.error ?? new Error("Failed to load feature flags.")} fallback="Failed to load feature flags." title="Feature flags" onRetry={() => void flags.refetch()} />}
        {!templates.data && <QueryErrorAlert error={templates.error ?? new Error("Failed to load email templates.")} fallback="Failed to load email templates." title="Email templates" onRetry={() => void templates.refetch()} />}
    </Stack></PageShell>;
    return <AdminPlatformContent key={platformAdminKey(config.data, plans.data, flags.data, templates.data)} config={config.data} plans={plans.data} flags={flags.data} templates={templates.data} />;
}
