import { useState, type ReactNode } from "react";
import { Alert, Stack, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { getRun } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { queryKeys } from "../../../config/queryKeys";
import { researchRunStaleTime } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import type { AnalysisRun, UnitType } from "../types";
import { ResultsInspector } from "./ResearchCharts";
import { ResearchResultsTable } from "./ResearchResults";
import { RunStatusChip } from "./ResearchShared";

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

function ResultOutput({ run }: { run: AnalysisRun | undefined }) {
    if (!run) return null;
    const results = asRecord(run.results);
    const firstArray = Object.values(results ?? {}).find((value) => Array.isArray(value)) as
        | unknown[]
        | undefined;
    const rows = (firstArray ?? [])
        .map(asRecord)
        .filter((row): row is Record<string, unknown> => Boolean(row))
        .map((row, index) => ({ ...row, id: index }));
    const columns = rows.length
        ? Object.keys(rows[0])
              .filter((key) => key !== "id")
              .slice(0, 7)
              .map((key) => ({
                  id: key,
                  label: key.replaceAll("_", " "),
                  value: (row: Record<string, unknown>) => {
                      const value = row[key];
                      return typeof value === "object" ? JSON.stringify(value) : String(value ?? "");
                  },
              }))
        : [];
    return (
        <Stack spacing={1.5}>
            <Typography variant="body2">
                Run <RunStatusChip status={run.status} />{" "}
                {run.progress_stage ? ` · ${run.progress_stage}` : ""}
            </Typography>
            {run.error_message ? <Alert severity="error">{run.error_message}</Alert> : null}
            {isActiveRunStatus(run.status) ? (
                <Typography color="text.secondary">Analysis in progress…</Typography>
            ) : null}
            {rows.length ? <ResearchResultsTable rows={rows} columns={columns} /> : null}
            {!isActiveRunStatus(run.status) ? (
                <ResultsInspector data={{ metrics: run.metrics, results: run.results }} />
            ) : null}
        </Stack>
    );
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

export function PanelBody({
    children,
    run,
    runQuery,
}: {
    children: ReactNode;
    run?: AnalysisRun;
    runQuery: ReturnType<typeof useQuery<AnalysisRun>>;
}) {
    return (
        <Stack spacing={2}>
            {children}
            <QueryBoundary
                isLoading={runQuery.isLoading && !run}
                isError={runQuery.isError}
                error={runQuery.error}
                onRetry={() => void runQuery.refetch()}
                variant="inline"
            >
                <ResultOutput run={run} />
            </QueryBoundary>
        </Stack>
    );
}
