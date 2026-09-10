import { useMemo } from "react";

import { useAuth } from "./useAuth";
import { usePlatformMetadata } from "./usePlatformMetadata";

export type SettingsTab = {
    label: string;
    path: string;
};

type SettingsTabDefinition = SettingsTab & {
    requiresPlatformModule?: boolean;
    requiresAiModule?: boolean;
    adminOnly?: boolean;
};

const SETTINGS_TAB_DEFINITIONS: SettingsTabDefinition[] = [
    { label: "Profile", path: "/profile" },
    { label: "Platform", path: "/platform", requiresPlatformModule: true },
    { label: "AI Studio", path: "/ai", requiresAiModule: true },
    { label: "RAG", path: "/rag", requiresAiModule: true },
    { label: "Memory", path: "/memory", requiresAiModule: true },
    { label: "Agent", path: "/agent", requiresAiModule: true },
    { label: "Observability", path: "/observability" },
    { label: "Settings", path: "/admin/settings", adminOnly: true },
    { label: "Users", path: "/admin/users", adminOnly: true },
    { label: "Platform Admin", path: "/admin/platform", adminOnly: true },
];

export function getVisibleSettingsTabs({
    isAdmin,
    hasUserPlatformModule,
    hasAiModule,
}: {
    isAdmin: boolean;
    hasUserPlatformModule: boolean;
    hasAiModule: boolean;
}): SettingsTab[] {
    return SETTINGS_TAB_DEFINITIONS.filter((item) => {
        if (item.adminOnly && !isAdmin) return false;
        if (item.requiresPlatformModule && !hasUserPlatformModule) return false;
        if (item.requiresAiModule && !hasAiModule) return false;
        return true;
    });
}

export function useSettingsTabs(): SettingsTab[] {
    const { isAdmin } = useAuth();
    const { data: platformMetadata } = usePlatformMetadata();
    const hasUserPlatformModule =
        platformMetadata?.module_catalog.some((item) => item.user_visible && item.enabled) ?? false;
    const hasAiModule =
        platformMetadata?.module_catalog.some((item) => item.key === "ai" && item.enabled) ??
        false;

    return useMemo(
        () => getVisibleSettingsTabs({ isAdmin, hasUserPlatformModule, hasAiModule }),
        [hasAiModule, hasUserPlatformModule, isAdmin]
    );
}

export function getActiveSettingsTab(
    pathname: string,
    tabs: SettingsTab[]
): SettingsTab | undefined {
    return [...tabs]
        .sort((left, right) => right.path.length - left.path.length)
        .find((item) => pathname === item.path || pathname.startsWith(`${item.path}/`));
}

export function isSettingsHubPath(pathname: string, tabs: SettingsTab[]): boolean {
    return getActiveSettingsTab(pathname, tabs) !== undefined;
}

export function getSettingsHubLabel(
    pathname: string,
    tabs: SettingsTab[]
): string | undefined {
    return getActiveSettingsTab(pathname, tabs)?.label;
}
