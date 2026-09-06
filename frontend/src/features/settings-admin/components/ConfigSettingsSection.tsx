import { Alert, Box, Button, Chip, Stack } from "@mui/material";
import type { ConfigSettingsResponse } from "../../../api/settings";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { AdminSettingsModel } from "../hooks/useAdminSettingsContent";
import { ConfigEntryEditor } from "./SettingEditors";

type Props = { configData: ConfigSettingsResponse; model: AdminSettingsModel };

export function ConfigSettingsSection({ configData, model }: Props) {
    const group = model.activeConfigGroup;
    if (!group) return null;
    return (
        <SectionCard title={group.label} description={group.description} action={
            <Button variant="contained" disabled={model.configMutation.isPending || model.changedConfigCount === 0}
                onClick={() => model.configMutation.mutate({ items: configData.items.map((item) => ({ key: item.key, value: model.configDrafts[item.key] ?? "" })) })}>
                {model.configMutation.isPending ? "Saving..." : "Save all config"}
            </Button>
        }>
            <Stack spacing={2}>
                <Alert severity="info">{configData.notice}</Alert>
                {model.configMutation.isSuccess && <Alert severity="success">Config saved. Restart the backend if a startup-bound value changed.</Alert>}
                {model.configMutation.isError && <Alert severity="error">{getQueryErrorMessage(model.configMutation.error, "Failed to save config.")}</Alert>}
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Chip label={`${group.items.length} variables in this group`} variant="outlined" />
                    <Chip label={model.changedConfigCount > 0 ? `${model.changedConfigCount} unsaved changes across all tabs` : "No unsaved config changes"}
                        color={model.changedConfigCount > 0 ? "warning" : "default"} variant="outlined" />
                </Stack>
                <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" }, alignItems: "start" }}>
                    {group.items.map((item) => <ConfigEntryEditor key={item.key} item={item}
                        value={model.configDrafts[item.key] ?? item.value}
                        onChange={(value) => model.setConfigDrafts((current) => ({ ...current, [item.key]: value }))} />)}
                </Box>
            </Stack>
        </SectionCard>
    );
}
