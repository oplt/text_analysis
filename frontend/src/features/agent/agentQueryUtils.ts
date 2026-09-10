import type { QueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../config/queryKeys";

export async function invalidateAgentRunQueries(queryClient: QueryClient, runId: string) {
    await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.agent.runs }),
        queryClient.invalidateQueries({ queryKey: queryKeys.agent.run(runId) }),
    ]);
}
