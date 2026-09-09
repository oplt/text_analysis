import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { researchRunEventsUrl } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import type { AnalysisRun } from "../types";

const RUN_EVENTS = [
    "queued",
    "started",
    "progress",
    "artifact-created",
    "completed",
    "failed",
    "cancelled",
] as const;

const LIST_INVALIDATING_EVENTS = new Set([
    "queued",
    "started",
    "completed",
    "failed",
    "cancelled",
]);

function parseRunPayload(data: string): AnalysisRun | null {
    try {
        const parsed = JSON.parse(data) as unknown;
        if (!parsed || typeof parsed !== "object") return null;
        const run = parsed as AnalysisRun;
        return typeof run.id === "string" ? run : null;
    } catch {
        return null;
    }
}

/** Keep active run queries current over SSE, falling back to polling on errors. */
export function useRunEvents(runId: string | null, projectId: string) {
    const queryClient = useQueryClient();
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        if (!runId || !projectId) {
            return;
        }

        const source = new EventSource(researchRunEventsUrl(runId), { withCredentials: true });
        const onEvent = (event: Event) => {
            const message = event as MessageEvent<string>;
            const run = typeof message.data === "string" ? parseRunPayload(message.data) : null;
            if (run) {
                queryClient.setQueryData(queryKeys.textResearch.run(run.id), run);
            } else {
                void queryClient.invalidateQueries({ queryKey: queryKeys.textResearch.run(runId) });
            }

            if (LIST_INVALIDATING_EVENTS.has(event.type)) {
                void queryClient.invalidateQueries({
                    queryKey: ["text-research", projectId, "runs"],
                });
            }

            if (event.type === "completed" || event.type === "failed" || event.type === "cancelled") {
                source.close();
                setConnected(false);
            }
        };

        source.onopen = () => setConnected(true);
        source.onerror = () => setConnected(false);
        for (const eventName of RUN_EVENTS) {
            source.addEventListener(eventName, onEvent);
        }

        return () => {
            source.close();
            setConnected(false);
        };
    }, [projectId, queryClient, runId]);

    return Boolean(runId && projectId && connected);
}
