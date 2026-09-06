import { describe, expect, it } from "vitest";

import { getVisibleSettingsTabs } from "./useSettingsTabs";

const paths = (isAdmin: boolean, hasUserPlatformModule: boolean, hasAiModule: boolean) =>
    getVisibleSettingsTabs({ isAdmin, hasUserPlatformModule, hasAiModule }).map((tab) => tab.path);

describe("settings tab visibility", () => {
    it("keeps capability and admin tabs hidden for a basic user", () => {
        expect(paths(false, false, false)).toEqual(["/profile", "/observability"]);
    });

    it("shows capability tabs independently", () => {
        expect(paths(false, true, false)).toContain("/platform");
        expect(paths(false, false, true)).toContain("/ai");
    });

    it("shows all admin tabs only to admins", () => {
        const adminPaths = paths(true, true, true);
        expect(adminPaths).toEqual(expect.arrayContaining([
            "/admin/settings",
            "/admin/users",
            "/admin/platform",
        ]));
        expect(paths(false, true, true).some((path) => path.startsWith("/admin/"))).toBe(false);
    });
});
