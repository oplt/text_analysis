import { Alert, Box, Button, Stack, TextField } from "@mui/material";
import { Storage as StorageIcon } from "@mui/icons-material";
import type { DatabaseSetting } from "../../../api/settings";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { AdminSettingsModel } from "../hooks/useAdminSettingsContent";
import { DatabaseSettingEditor } from "./SettingEditors";

export function DatabaseSettingsSections({ settings, model }: { settings: DatabaseSetting[]; model: AdminSettingsModel }) {
    const create = model.createDatabaseMutation;
    return (
        <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", lg: "minmax(320px, 0.9fr) minmax(0, 1.1fr)" }, alignItems: "start" }}>
            <SectionCard title="Add database setting" description="Store arbitrary key/value settings inside the database.">
                <Stack spacing={2}>
                    {create.isSuccess && <Alert severity="success">Database setting created.</Alert>}
                    {create.isError && <Alert severity="error">{getQueryErrorMessage(create.error, "Failed to create database setting.")}</Alert>}
                    <TextField label="Key" value={model.newSetting.key} onChange={(event) => model.setNewSetting((current) => ({ ...current, key: event.target.value }))} fullWidth />
                    <TextField label="Value" value={model.newSetting.value} onChange={(event) => model.setNewSetting((current) => ({ ...current, value: event.target.value }))} fullWidth multiline minRows={3} />
                    <TextField label="Description" value={model.newSetting.description} onChange={(event) => model.setNewSetting((current) => ({ ...current, description: event.target.value }))} fullWidth multiline minRows={3} />
                    <Button variant="contained" disabled={create.isPending || !model.newSetting.key.trim()}
                        onClick={() => create.mutate({ key: model.newSetting.key.trim(), value: model.newSetting.value, description: model.newSetting.description || undefined })}>
                        {create.isPending ? "Adding..." : "Add setting"}
                    </Button>
                </Stack>
            </SectionCard>
            <SectionCard title="Database settings" description="Review, edit, and delete runtime settings stored in the database.">
                {settings.length > 0 ? <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", xl: "repeat(2, minmax(0, 1fr))" }, alignItems: "start" }}>
                    {settings.map((item) => {
                        const draft = model.databaseDrafts[item.id] ?? { value: item.value, description: item.description ?? "" };
                        return <DatabaseSettingEditor key={item.id} item={item} draft={draft}
                            onDraftChange={(next) => model.setDatabaseDrafts((current) => ({ ...current, [item.id]: next }))}
                            onSave={() => model.updateDatabaseMutation.mutate({ id: item.id, ...draft })}
                            onDelete={() => model.deleteDatabaseMutation.mutate(item.id)}
                            isSaving={model.updateDatabaseMutation.isPending && model.updateDatabaseMutation.variables?.id === item.id}
                            isDeleting={model.deleteDatabaseMutation.isPending && model.deleteDatabaseMutation.variables === item.id} />;
                    })}
                </Box> : <EmptyState icon={<StorageIcon />} title="No database settings yet" description="Create a setting when you need runtime-configurable values stored in the database." />}
            </SectionCard>
        </Box>
    );
}
