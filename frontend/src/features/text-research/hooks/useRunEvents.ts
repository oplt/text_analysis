import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { researchRunEventsUrl } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";

const RUN_EVENTS = [
    "queued",
    "started",
    "progress",
    "artifact-created",
    "completed",
    "failed",
    "cancelled",
] as const;

/** Keep active run queries current over SSE, falling back to polling on errors. */
export function useRunEvents(runId: string | null, projectId: string) {
    const queryClient = useQueryClient();
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        if (!runId || !projectId) {
            setConnected(false);
            return;
        }

        const source = new EventSource(researchRunEventsUrl(runId), { withCredentials: true });
        const invalidate = (event: Event) => {
            void Promise.all([
                queryClient.invalidateQueries({ queryKey: queryKeys.textResearch.run(runId) }),
                queryClient.invalidateQueries({ queryKey: ["text-research", projectId, "runs"] }),
            ]);
            if (event.type === "completed" || event.type === "failed" || event.type === "cancelled") {
                source.close();
                setConnected(false);
            }
        };

        source.onopen = () => setConnected(true);
        source.onerror = () => setConnected(false);
        for (const eventName of RUN_EVENTS) {
            source.addEventListener(eventName, invalidate);
        }

        return () => {
            source.close();
            setConnected(false);
        };
    }, [projectId, queryClient, runId]);

    return connected;
}
