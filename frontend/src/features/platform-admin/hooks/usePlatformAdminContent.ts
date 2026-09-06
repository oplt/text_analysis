import { useState } from "react";
import type { EmailTemplate, FeatureFlag, PlatformConfig, SubscriptionPlan } from "../../../api/platform";
import { buildConfigDraft, buildFlagDrafts, buildPlanDrafts, buildTemplateDrafts } from "../platformAdminModel";
import { EMPTY_NEW_FLAG, EMPTY_NEW_PLAN, EMPTY_NEW_TEMPLATE, usePlatformAdminMutations } from "./usePlatformAdminMutations";

export function usePlatformAdminContent(config: PlatformConfig, plans: SubscriptionPlan[], flags: FeatureFlag[], templates: EmailTemplate[]) {
    const [configDraft, setConfigDraft] = useState(() => buildConfigDraft(config));
    const [planDrafts, setPlanDrafts] = useState(() => buildPlanDrafts(plans));
    const [flagDrafts, setFlagDrafts] = useState(() => buildFlagDrafts(flags));
    const [templateDrafts, setTemplateDrafts] = useState(() => buildTemplateDrafts(templates));
    const [newPlan, setNewPlan] = useState(EMPTY_NEW_PLAN);
    const [newFlag, setNewFlag] = useState(EMPTY_NEW_FLAG);
    const [newTemplate, setNewTemplate] = useState(EMPTY_NEW_TEMPLATE);
    const mutations = usePlatformAdminMutations({ setNewPlan, setNewFlag, setNewTemplate });
    return { configDraft, setConfigDraft, planDrafts, setPlanDrafts, flagDrafts, setFlagDrafts,
        templateDrafts, setTemplateDrafts, newPlan, setNewPlan, newFlag, setNewFlag,
        newTemplate, setNewTemplate, ...mutations };
}

export type PlatformAdminModel = ReturnType<typeof usePlatformAdminContent>;
