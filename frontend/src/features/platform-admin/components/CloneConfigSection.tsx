import { Alert, Box, Button, MenuItem, Stack, Switch, TextField, Typography } from "@mui/material";
import type { PlatformConfig } from "../../../api/platform";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { PlatformAdminModel } from "../hooks/usePlatformAdminContent";

export function CloneConfigSection({ source, model }: { source: PlatformConfig; model: PlatformAdminModel }) {
    const draft = model.configDraft;
    const set = model.setConfigDraft;
    const activePack = source.available_module_packs.find((pack) => pack.key === draft.module_pack);
    return <SectionCard title="Clone configuration" description="Set the product name, core domain terminology, module pack, and module visibility defaults."
        action={<Button variant="contained" disabled={model.saveConfigMutation.isPending}
            onClick={() => model.saveConfigMutation.mutate({ app_name: draft.app_name, core_domain_singular: draft.core_domain_singular,
                core_domain_plural: draft.core_domain_plural, module_pack: draft.module_pack, module_overrides: draft.module_states, mfa_enabled: draft.mfa_enabled })}>
            {model.saveConfigMutation.isPending ? "Saving..." : "Save platform config"}</Button>}>
        <Stack spacing={2.5}>
            {model.saveConfigMutation.isError && <Alert severity="error">{getQueryErrorMessage(model.saveConfigMutation.error, "Failed to save platform config.")}</Alert>}
            <TextField label="App name" value={draft.app_name} onChange={(e) => set((v) => ({ ...v, app_name: e.target.value }))} fullWidth />
            <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" } }}>
                <TextField label="Core domain singular" value={draft.core_domain_singular} onChange={(e) => set((v) => ({ ...v, core_domain_singular: e.target.value }))} fullWidth />
                <TextField label="Core domain plural" value={draft.core_domain_plural} onChange={(e) => set((v) => ({ ...v, core_domain_plural: e.target.value }))} fullWidth />
            </Box>
            <TextField label="Module pack" select value={draft.module_pack} onChange={(e) => {
                const module_pack = e.target.value;
                const defaults = source.available_module_packs.find((pack) => pack.key === module_pack)?.modules ?? [];
                set((v) => ({ ...v, module_pack, module_states: Object.fromEntries(source.module_catalog.map((item) => [item.key, defaults.includes(item.key)])) }));
            }} fullWidth>{source.available_module_packs.map((pack) => <MenuItem key={pack.key} value={pack.key}>{pack.label}</MenuItem>)}</TextField>
            {activePack && <Alert severity="info">{activePack.description}</Alert>}
            <Box sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                <Stack direction="row" justifyContent="space-between" spacing={1.5}><Box><Typography variant="subtitle2">MFA authentication</Typography><Typography variant="body2" color="text.secondary">Show the authenticator code field on the login page.</Typography></Box>
                    <Switch checked={draft.mfa_enabled} onChange={(e) => set((v) => ({ ...v, mfa_enabled: e.target.checked }))} /></Stack>
            </Box>
            <Box><Typography variant="subtitle2" sx={{ mb: 1.25 }}>Module access</Typography>
                <Box sx={{ display: "grid", gap: 1.25, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" } }}>
                    {source.module_catalog.map((item) => <Box key={item.key} sx={(theme) => ({ p: 2, borderRadius: 4, border: `1px solid ${theme.palette.divider}` })}>
                        <Stack direction="row" justifyContent="space-between" spacing={1.5}><Box><Typography variant="subtitle2">{item.label}</Typography><Typography variant="body2" color="text.secondary">{item.description}</Typography></Box>
                            <Switch checked={draft.module_states[item.key] ?? false} onChange={(e) => set((v) => ({ ...v, module_states: { ...v.module_states, [item.key]: e.target.checked } }))} /></Stack>
                    </Box>)}
                </Box>
            </Box>
        </Stack>
    </SectionCard>;
}
