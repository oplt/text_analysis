import type { ReactNode } from "react";
import { Alert, Stack, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { useResearchContext } from "../hooks/useResearchContext";
import { isActiveRunStatus } from "../runPolling";
import type { AnalysisRun } from "../types";
import { ActiveRunActions } from "./ActiveRunActions";
import { ResultsInspector } from "./ResearchCharts";
import { ResearchResultsTable } from "./ResearchResults";
import { RunStatusChip } from "./ResearchShared";
import { asRecord } from "./advancedAnalysisUtils";

function ResultOutput({ run }: { run: AnalysisRun | undefined }) {
    const ctx = useResearchContext();
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
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                <Typography variant="body2">
                    Run <RunStatusChip status={run.status} />{" "}
                    {run.progress_stage ? ` · ${run.progress_stage}` : ""}
                </Typography>
                <ActiveRunActions
                    run={run}
                    projectId={ctx.projectId}
                    corpusId={ctx.selectedCorpusId}
                />
            </Stack>
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
