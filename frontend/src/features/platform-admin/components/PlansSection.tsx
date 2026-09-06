import { Box, Button, Chip, FormControlLabel, Stack, Switch, TextField, Typography } from "@mui/material";
import type { SubscriptionPlan } from "../../../api/platform";
import { SectionCard } from "../../../components/ui/SectionCard";
import type { PlatformAdminModel } from "../hooks/usePlatformAdminContent";

export function PlansSection({ plans, model }: { plans: SubscriptionPlan[]; model: PlatformAdminModel }) {
    const create = model.newPlan;
    return <SectionCard title="Subscription plans" description="Create new commercial tiers and tune existing plans."><Stack spacing={2.5}>
        <Box sx={(t) => ({ p: 2.5, borderRadius: 4, border: `1px solid ${t.palette.divider}` })}><Stack spacing={1.5}>
            <Typography variant="subtitle2">Create plan</Typography>
            <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" } }}>
                <TextField label="Code" value={create.code} onChange={(e) => model.setNewPlan((v) => ({ ...v, code: e.target.value }))} />
                <TextField label="Name" value={create.name} onChange={(e) => model.setNewPlan((v) => ({ ...v, name: e.target.value }))} />
                <TextField label="Price (cents)" value={create.price_cents} onChange={(e) => model.setNewPlan((v) => ({ ...v, price_cents: e.target.value }))} />
                <TextField label="Interval" value={create.interval} onChange={(e) => model.setNewPlan((v) => ({ ...v, interval: e.target.value }))} />
            </Box>
            <TextField label="Description" value={create.description} onChange={(e) => model.setNewPlan((v) => ({ ...v, description: e.target.value }))} />
            <TextField label="Features" value={create.features} onChange={(e) => model.setNewPlan((v) => ({ ...v, features: e.target.value }))} helperText="Comma-separated feature labels" />
            <FormControlLabel control={<Switch checked={create.is_default} onChange={(e) => model.setNewPlan((v) => ({ ...v, is_default: e.target.checked }))} />} label="Default plan" />
            <Button variant="contained" disabled={model.createPlanMutation.isPending || create.code.trim().length < 2} onClick={() => model.createPlanMutation.mutate({ code: create.code.trim(), name: create.name.trim(), description: create.description.trim() || undefined, price_cents: Number(create.price_cents), interval: create.interval.trim(), is_default: create.is_default, features: create.features.split(",").map((v) => v.trim()).filter(Boolean) })}>{model.createPlanMutation.isPending ? "Creating..." : "Create plan"}</Button>
        </Stack></Box>
        <Stack spacing={1.5}>{plans.map((plan) => { const d = model.planDrafts[plan.id]; const saving = model.updatePlanMutation.isPending && model.updatePlanMutation.variables?.id === plan.id; return <Box key={plan.id} sx={(t) => ({ p: 2.5, borderRadius: 4, border: `1px solid ${t.palette.divider}` })}><Stack spacing={1.5}>
            <Stack direction="row" spacing={1} alignItems="center"><Typography variant="subtitle2">{plan.code}</Typography>{plan.is_default && <Chip label="Default" size="small" color="primary" />}</Stack>
            <Box sx={{ display: "grid", gap: 1.5, gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" } }}>
                <TextField label="Name" value={d.name} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, name: e.target.value } }))} />
                <TextField label="Price (cents)" value={d.price_cents} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, price_cents: e.target.value } }))} />
                <TextField label="Interval" value={d.interval} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, interval: e.target.value } }))} />
                <TextField label="Features" value={d.features} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, features: e.target.value } }))} />
            </Box>
            <TextField label="Description" value={d.description} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, description: e.target.value } }))} />
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1}><FormControlLabel control={<Switch checked={d.is_active} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, is_active: e.target.checked } }))} />} label="Active" /><FormControlLabel control={<Switch checked={d.is_default} onChange={(e) => model.setPlanDrafts((v) => ({ ...v, [plan.id]: { ...d, is_default: e.target.checked } }))} />} label="Default" /></Stack>
            <Button variant="outlined" disabled={saving} onClick={() => model.updatePlanMutation.mutate({ id: plan.id, draft: d })}>{saving ? "Saving..." : "Save plan"}</Button>
        </Stack></Box>; })}</Stack>
    </Stack></SectionCard>;
}
