import { Box } from "@mui/material";
import { Extension as ExtensionIcon, Flag as FlagIcon, MailOutline as MailIcon, Sell as SellIcon } from "@mui/icons-material";
import type { EmailTemplate, FeatureFlag, SubscriptionPlan } from "../../../api/platform";
import { StatCard } from "../../../components/ui/StatCard";
import type { ConfigDraft } from "../platformAdminModel";

export function AdminPlatformStats({ config, plans, flags, templates }: { config: ConfigDraft; plans: SubscriptionPlan[]; flags: FeatureFlag[]; templates: EmailTemplate[] }) {
    return <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", xl: "repeat(4, minmax(0, 1fr))" } }}>
        <StatCard label="Enabled modules" value={Object.values(config.module_states).filter(Boolean).length} description="Modules currently exposed by the selected pack and overrides" icon={<ExtensionIcon />} />
        <StatCard label="Subscription plans" value={plans.length} description="Commercial tiers available across the platform" icon={<SellIcon />} color="secondary" />
        <StatCard label="Feature flags" value={flags.length} description="Flags available for rollout and experimentation" icon={<FlagIcon />} color="success" />
        <StatCard label="Email templates" value={templates.length} description="Transactional templates ready for automated delivery" icon={<MailIcon />} color="warning" />
    </Box>;
}
