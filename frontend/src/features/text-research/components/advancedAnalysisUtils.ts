import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { getRun } from "../../../api/textResearch";
import { useSnackbar } from "../../../app/snackbarContext";
import { queryKeys } from "../../../config/queryKeys";
import { researchRunStaleTime } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval } from "../runPolling";
import type { AnalysisRun, UnitType } from "../types";

export type AdvancedAnalysisBasePayload = {
    unit_type: UnitType;
    preprocessing_profile_id?: string;
    [key: string]: unknown;
};

export function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

export function useAnalysisRun() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [runId, setRunId] = useState<string | null>(null);
    const sseConnected = useRunEvents(runId, ctx.projectId);
    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        staleTime: (query) => researchRunStaleTime(query.state.data?.status),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });
    return { ctx, run: runQuery.data, runQuery, setRunId, showToast };
}

export function useRunMutation(
    action: () => Promise<AnalysisRun>,
    label: string,
    setRunId: (runId: string) => void,
    showToast: ReturnType<typeof useSnackbar>["showToast"]
) {
    return useMutation({
        mutationFn: action,
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: `${label} started.`, severity: "success" });
        },
        onError: (error) =>
            showToast({ message: getQueryErrorMessage(error, `${label} failed.`), severity: "error" }),
    });
}
