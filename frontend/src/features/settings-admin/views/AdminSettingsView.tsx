import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Skeleton, Stack } from "@mui/material";
import { getConfigSettings, listDatabaseSettings } from "../../../api/settings";
import { queryKeys } from "../../../config/queryKeys";
import { AdminSettingsTabs } from "../../../components/layout/AdminSettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { AdminSettingsContent } from "../components/AdminSettingsContent";
import { buildConfigGroups, settingsContentKey, type SettingsTabValue } from "../settingsModel";

export default function AdminSettingsPage() {
    const [activeTab, setActiveTab] = useState<SettingsTabValue>("application");
    const config = useQuery({ queryKey: queryKeys.settings.config, queryFn: getConfigSettings });
    const database = useQuery({ queryKey: queryKeys.settings.database, queryFn: listDatabaseSettings });
    if ((config.isLoading && !config.data) || (database.isLoading && !database.data)) {
        return <PageShell maxWidth="xl"><Stack spacing={2}><Skeleton variant="rounded" height={180} sx={{ borderRadius: 6 }} /><Skeleton variant="rounded" height={320} sx={{ borderRadius: 6 }} /></Stack></PageShell>;
    }
    if (!config.data) {
        return (
            <PageShell maxWidth="xl"><AdminSettingsTabs /><Stack spacing={2} sx={{ mt: 2 }}>
                <QueryErrorAlert error={config.error ?? new Error("Failed to load config values.")} fallback="Failed to load config values." title="Config" onRetry={() => void config.refetch()} />
                {database.isError && <QueryErrorAlert error={database.error} fallback="Failed to load database settings." title="Database settings" onRetry={() => void database.refetch()} />}
            </Stack></PageShell>
        );
    }
    const settings = database.data ?? [];
    const groups = buildConfigGroups(config.data.items);
    const resolvedTab = activeTab === "database" || groups.some((group) => group.id === activeTab) ? activeTab : groups[0]?.id ?? "database";
    return <AdminSettingsContent key={settingsContentKey(config.data, settings)} configData={config.data} databaseSettings={settings}
        configErrorMessage={getQueryErrorMessage(config.error, "Failed to load config values.")}
        databaseErrorMessage={getQueryErrorMessage(database.error, "Failed to load database settings.")}
        hasConfigError={config.isError} hasDatabaseError={database.isError || !database.data}
        activeTab={resolvedTab} onTabChange={setActiveTab} />;
}
