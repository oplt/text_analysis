import type { EmailTemplate, FeatureFlag, PlatformConfig, SubscriptionPlan } from "../../../api/platform";
import { AdminSettingsTabs } from "../../../components/layout/AdminSettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { usePlatformAdminContent } from "../hooks/usePlatformAdminContent";
import { AdminPlatformStats } from "./AdminPlatformStats";
import { CloneConfigSection } from "./CloneConfigSection";
import { FlagsSection } from "./FlagsSection";
import { PlansSection } from "./PlansSection";
import { TemplatesSection } from "./TemplatesSection";

export function AdminPlatformContent({ config, plans, flags, templates }: { config: PlatformConfig; plans: SubscriptionPlan[]; flags: FeatureFlag[]; templates: EmailTemplate[] }) {
    const model = usePlatformAdminContent(config, plans, flags, templates);
    return <PageShell maxWidth="xl"><AdminSettingsTabs /><AdminPlatformStats config={model.configDraft} plans={plans} flags={flags} templates={templates} />
        <CloneConfigSection source={config} model={model} /><PlansSection plans={plans} model={model} />
        <FlagsSection flags={flags} model={model} /><TemplatesSection templates={templates} model={model} />
    </PageShell>;
}
