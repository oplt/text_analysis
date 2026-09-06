import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../../config/queryKeys";
import { invalidatePlatformAdminCapability } from "./usePlatformAdminMutations";

describe("platform-admin invalidation", () => {
    it("invalidates only selected capability", async () => {
        const client = new QueryClient();
        client.setQueryData(queryKeys.platform.admin.plans, []);
        client.setQueryData(queryKeys.platform.admin.featureFlags, []);
        await invalidatePlatformAdminCapability(client, "plans");
        expect(client.getQueryState(queryKeys.platform.admin.plans)?.isInvalidated).toBe(true);
        expect(client.getQueryState(queryKeys.platform.admin.featureFlags)?.isInvalidated).toBe(false);
    });

    it("invalidates platform prefix for config changes", async () => {
        const client = new QueryClient();
        client.setQueryData(queryKeys.platform.admin.config, {});
        client.setQueryData(queryKeys.platform.plans, []);
        await invalidatePlatformAdminCapability(client, "all");
        expect(client.getQueryState(queryKeys.platform.admin.config)?.isInvalidated).toBe(true);
        expect(client.getQueryState(queryKeys.platform.plans)?.isInvalidated).toBe(true);
    });
});
