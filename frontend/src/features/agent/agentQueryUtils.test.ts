import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../config/queryKeys";
import { invalidateAgentRunQueries } from "./agentQueryUtils";

describe("agent run query invalidation", () => {
    it("invalidates the persisted history and selected detail after a run", async () => {
        const client = new QueryClient();
        client.setQueryData(queryKeys.agent.runs, { items: [], total: 0, limit: 20, offset: 0 });
        client.setQueryData(queryKeys.agent.run("run-1"), { id: "run-1" });

        await invalidateAgentRunQueries(client, "run-1");

        expect(client.getQueryState(queryKeys.agent.runs)?.isInvalidated).toBe(true);
        expect(client.getQueryState(queryKeys.agent.run("run-1"))?.isInvalidated).toBe(true);
    });
});
