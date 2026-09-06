import { Alert, Box, Tab, Tabs } from "@mui/material";
import { RestartAlt as RestartIcon, SettingsSuggest as SettingsIcon, Storage as StorageIcon } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import type { ConfigSettingsResponse, DatabaseSetting } from "../../../api/settings";
import { AdminSettingsTabs } from "../../../components/layout/AdminSettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { StatCard } from "../../../components/ui/StatCard";
import type { SettingsTabValue } from "../settingsModel";
import { useAdminSettingsContent } from "../hooks/useAdminSettingsContent";
import { ConfigSettingsSection } from "./ConfigSettingsSection";
import { DatabaseSettingsSections } from "./DatabaseSettingsSections";

type Props = {
    configData: ConfigSettingsResponse; databaseSettings: DatabaseSetting[];
    configErrorMessage: string; databaseErrorMessage: string;
    hasConfigError: boolean; hasDatabaseError: boolean;
    activeTab: SettingsTabValue; onTabChange: (tab: SettingsTabValue) => void;
};

export function AdminSettingsContent(props: Props) {
    const model = useAdminSettingsContent(props.configData, props.databaseSettings, props.activeTab);
    return (
        <PageShell maxWidth="xl">
            <AdminSettingsTabs />
            <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", xl: "repeat(3, minmax(0, 1fr))" } }}>
                <StatCard label="Config variables" value={props.configData.items.length} description="Environment-backed values available in the settings file" icon={<SettingsIcon />} />
                <StatCard label="Restart-sensitive" value={props.configData.items.filter((item) => item.requires_restart).length} description="Values likely to require a backend restart after saving" icon={<RestartIcon />} color="warning" />
                <StatCard label="Runtime database settings" value={props.databaseSettings.length} description="Key/value records stored in the database for live updates" icon={<StorageIcon />} color="secondary" />
            </Box>
            {props.hasConfigError && <Alert severity="error">{props.configErrorMessage}</Alert>}
            {props.hasDatabaseError && <Alert severity="error">{props.databaseErrorMessage}</Alert>}
            <Box sx={(theme) => ({ p: 1, borderRadius: 4, border: `1px solid ${theme.palette.divider}`, backgroundColor: alpha(theme.palette.background.paper, 0.82) })}>
                <Tabs value={props.activeTab} onChange={(_, value: SettingsTabValue) => props.onTabChange(value)} variant="scrollable" scrollButtons="auto" allowScrollButtonsMobile>
                    {model.configGroups.map((group) => <Tab key={group.id} value={group.id} label={`${group.label} (${group.items.length})`} />)}
                    <Tab value="database" label={`Database settings (${props.databaseSettings.length})`} />
                </Tabs>
            </Box>
            {model.activeConfigGroup
                ? <ConfigSettingsSection configData={props.configData} model={model} />
                : <DatabaseSettingsSections settings={props.databaseSettings} model={model} />}
        </PageShell>
    );
}
