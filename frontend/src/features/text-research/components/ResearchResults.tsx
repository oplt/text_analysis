import { useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
    Alert,
    Button,
    Stack,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import { Download as DownloadIcon } from "@mui/icons-material";
import {
    getRunResults,
    researchExportUrl,
    runResultsDownloadUrl,
} from "../../../api/textResearch";
import type { AnalysisRun } from "../types";
import { ActiveRunActions } from "./ActiveRunActions";
import { MetricCards, ResultsInspector } from "./ResearchCharts";
import { DataTable } from "../../../components/ui/DataTable";
import { RunStatusPanel } from "../../../components/ui/RunStatusPanel";
import { isActiveRunStatus } from "../runPolling";
import { ProvenanceDrawer } from "./ProvenanceDrawer";
import { ProvenancePanel } from "./ProvenancePanel";

export type ResearchResultsColumn<Row> = {
    id: string;
    label: string;
    value: (row: Row) => string | number | null | undefined;
    align?: "left" | "right" | "center";
};

export function ChartTableToggle({
    value,
    onChange,
}: {
    value: "chart" | "table" | "both";
    onChange: (value: "chart" | "table" | "both") => void;
}) {
    return (
        <ToggleButtonGroup
            size="small"
            exclusive
            value={value}
            onChange={(_, next) => next && onChange(next)}
            aria-label="Result display"
        >
            <ToggleButton value="chart">Chart</ToggleButton>
            <ToggleButton value="table">Table</ToggleButton>
            <ToggleButton value="both">Both</ToggleButton>
        </ToggleButtonGroup>
    );
}

export function ResearchResultsTable<Row extends { id?: string | number }>({
    columns,
    rows,
    emptyMessage = "No rows were returned for this analysis.",
    pageSize = 25,
}: {
    columns: Array<ResearchResultsColumn<Row>>;
    rows: Row[];
    emptyMessage?: string;
    pageSize?: number;
}) {
    const tableColumns = useMemo(
        () =>
            columns.map((column) => ({
                id: column.id,
                label: column.label,
                align: column.align,
                sortable: true,
                truncate: true as const,
                getSortValue: (row: Row) => column.value(row),
                render: (row: Row) => {
                    const value = column.value(row);
                    return value == null || value === "" ? "—" : String(value);
                },
            })),
        [columns]
    );

    return (
        <DataTable
            ariaLabel="Analysis result table"
            columns={tableColumns}
            rows={rows}
            getRowId={(row, index) => String(row.id ?? `row-${index}`)}
            density="compact"
            stickyHeader
            stickyFirstColumn
            clientSort
            pageSize={pageSize}
            emptyDescription={emptyMessage}
            showDensityToggle={rows.length > 20}
        />
    );
}

export function MethodsAndProvenanceContent({
    run,
    compact = false,
}: {
    run: AnalysisRun;
    compact?: boolean;
}) {
    return (
        <ProvenancePanel
            run={run}
            compact={compact}
            showRaw={!compact}
            actions={!compact ? <ResearchExportActions runId={run.id} /> : null}
        />
    );
}

export function MethodsAndProvenanceDrawer({ run }: { run: AnalysisRun }) {
    return (
        <ProvenanceDrawer
            run={run}
            title="Methods and provenance"
            tooltip="Methods and provenance"
            actions={<ResearchExportActions runId={run.id} />}
        />
    );
}

export function ResearchExportActions({ runId }: { runId: string }) {
    return <Button size="small" startIcon={<DownloadIcon />} href={researchExportUrl(`/research/runs/${runId}/export.json`)} target="_blank" rel="noopener">Export JSON</Button>;
}

function resultCollectionKey(results: Record<string, unknown> | null): string | undefined {
    if (!results) return undefined;
    return Object.keys(results).find((key) => key.endsWith("_total"))?.replace(/_total$/, "");
}

export function FullResultsActions({ run }: { run: AnalysisRun }) {
    const [open, setOpen] = useState(false);
    const [page, setPage] = useState(0);
    const results = run.results;
    const artifactized =
        typeof results?.results_artifact_id === "string" ||
        typeof run.metrics?.results_artifact_id === "string";
    const collectionKey = resultCollectionKey(results);
    const inlineLarge = Object.values(results ?? {}).some(
        (value) => Array.isArray(value) && value.length > 100
    );
    const resultsQuery = useQuery({
        queryKey: ["text-research", "run-results", run.id, collectionKey, page],
        queryFn: () =>
            getRunResults(run.id, {
                key: collectionKey,
                limit: 100,
                offset: page * 100,
            }),
        enabled: open,
    });

    if (!artifactized && !inlineLarge) return null;
    const resultPage = resultsQuery.data;
    const pageCount = resultPage?.total ? Math.ceil(resultPage.total / 100) : 1;
    return (
        <Stack spacing={1}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Button size="small" variant="outlined" onClick={() => setOpen(true)}>
                    View full results
                </Button>
                <Button
                    size="small"
                    startIcon={<DownloadIcon />}
                    href={runResultsDownloadUrl(run.id)}
                    target="_blank"
                    rel="noopener"
                >
                    Download full results
                </Button>
            </Stack>
            {open ? (
                resultsQuery.isLoading ? (
                    <Typography variant="body2" color="text.secondary">Loading full results…</Typography>
                ) : resultsQuery.isError ? (
                    <Typography variant="body2" color="error">Full results could not be loaded.</Typography>
                ) : (
                    <Stack spacing={1}>
                        <ResultsInspector
                            title={collectionKey ? `${collectionKey} (full results)` : "full results"}
                            data={resultPage?.items ?? resultPage?.data}
                        />
                        {resultPage?.total != null && pageCount > 1 ? (
                            <Stack direction="row" spacing={1} alignItems="center">
                                <Button size="small" disabled={page === 0} onClick={() => setPage(page - 1)}>
                                    Previous
                                </Button>
                                <Typography variant="caption">
                                    Page {page + 1} of {pageCount}
                                </Typography>
                                <Button
                                    size="small"
                                    disabled={page >= pageCount - 1}
                                    onClick={() => setPage(page + 1)}
                                >
                                    Next
                                </Button>
                            </Stack>
                        ) : null}
                    </Stack>
                )
            ) : null}
        </Stack>
    );
}

export function ResearchResultPanel({
    run,
    title,
    metricItems,
    children,
    projectId,
    corpusId,
}: {
    run: AnalysisRun;
    title: string;
    metricItems?: Array<{ label: string; value: string | number | null | undefined }>;
    children: ReactNode;
    projectId?: string | null;
    corpusId?: string | null;
}) {
    return (
        <Stack spacing={2}>
            <RunStatusPanel
                title={title}
                status={run.status}
                runId={run.id}
                stage={run.progress_stage}
                startedAt={run.started_at}
                completedAt={run.completed_at}
                createdAt={run.created_at}
                errorMessage={run.error_message}
                actions={
                    <>
                        <ActiveRunActions run={run} projectId={projectId} corpusId={corpusId} />
                        <MethodsAndProvenanceDrawer run={run} />
                        <ResearchExportActions runId={run.id} />
                    </>
                }
            />
            {isActiveRunStatus(run.status) ? (
                <Alert severity="info">Analysis in progress…</Alert>
            ) : null}
            {metricItems?.length ? <MetricCards items={metricItems} /> : null}
            {children}
            <FullResultsActions run={run} />
            <ResultsInspector title="raw JSON" data={{ metrics: run.metrics, results: run.results }} />
        </Stack>
    );
}
