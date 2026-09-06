import { Box, Button, FormControlLabel, Stack, Switch, TextField, Typography } from "@mui/material";
import type { FeatureFlag } from "../../../api/platform";
import { SectionCard } from "../../../components/ui/SectionCard";
import type { PlatformAdminModel } from "../hooks/usePlatformAdminContent";

export function FlagsSection({ flags, model }: { flags: FeatureFlag[]; model: PlatformAdminModel }) {
    const n = model.newFlag;
    return <SectionCard title="Feature flags" description="Create rollout controls and tune existing flags."><Stack spacing={2.5}>
        <Box sx={(t) => ({ p: 2.5, borderRadius: 4, border: `1px solid ${t.palette.divider}` })}><Stack spacing={1.5}><Typography variant="subtitle2">Create flag</Typography>
            <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" } }}>
                <TextField label="Key" value={n.key} onChange={(e) => model.setNewFlag((v) => ({ ...v, key: e.target.value }))} /><TextField label="Name" value={n.name} onChange={(e) => model.setNewFlag((v) => ({ ...v, name: e.target.value }))} />
                <TextField label="Module key" value={n.module_key} onChange={(e) => model.setNewFlag((v) => ({ ...v, module_key: e.target.value }))} /><TextField label="Rollout %" value={n.rollout_percentage} onChange={(e) => model.setNewFlag((v) => ({ ...v, rollout_percentage: e.target.value }))} />
            </Box><TextField label="Description" value={n.description} onChange={(e) => model.setNewFlag((v) => ({ ...v, description: e.target.value }))} /><FormControlLabel control={<Switch checked={n.is_enabled} onChange={(e) => model.setNewFlag((v) => ({ ...v, is_enabled: e.target.checked }))} />} label="Enabled" />
            <Button variant="contained" disabled={model.createFlagMutation.isPending || n.key.trim().length < 2} onClick={() => model.createFlagMutation.mutate({ key: n.key.trim(), name: n.name.trim(), description: n.description.trim() || undefined, module_key: n.module_key.trim() || null, is_enabled: n.is_enabled, rollout_percentage: Number(n.rollout_percentage) })}>{model.createFlagMutation.isPending ? "Creating..." : "Create flag"}</Button>
        </Stack></Box>
        <Stack spacing={1.5}>{flags.map((flag) => { const d = model.flagDrafts[flag.id]; const saving = model.updateFlagMutation.isPending && model.updateFlagMutation.variables?.id === flag.id; return <Box key={flag.id} sx={(t) => ({ p: 2.5, borderRadius: 4, border: `1px solid ${t.palette.divider}` })}><Stack spacing={1.5}><Typography variant="subtitle2">{flag.key}</Typography>
            <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" } }}><TextField label="Name" value={d.name} onChange={(e) => model.setFlagDrafts((v) => ({ ...v, [flag.id]: { ...d, name: e.target.value } }))} /><TextField label="Module key" value={d.module_key} onChange={(e) => model.setFlagDrafts((v) => ({ ...v, [flag.id]: { ...d, module_key: e.target.value } }))} /><TextField label="Rollout %" value={d.rollout_percentage} onChange={(e) => model.setFlagDrafts((v) => ({ ...v, [flag.id]: { ...d, rollout_percentage: e.target.value } }))} /></Box>
            <TextField label="Description" value={d.description} onChange={(e) => model.setFlagDrafts((v) => ({ ...v, [flag.id]: { ...d, description: e.target.value } }))} /><FormControlLabel control={<Switch checked={d.is_enabled} onChange={(e) => model.setFlagDrafts((v) => ({ ...v, [flag.id]: { ...d, is_enabled: e.target.checked } }))} />} label="Enabled" /><Button variant="outlined" disabled={saving} onClick={() => model.updateFlagMutation.mutate({ id: flag.id, draft: d })}>{saving ? "Saving..." : "Save flag"}</Button>
        </Stack></Box>; })}</Stack>
    </Stack></SectionCard>;
}
