import { useMemo, useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControlLabel,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Tooltip,
    Typography,
} from "@mui/material";
import {
    ContentCopy as CloneIcon,
    Download as DownloadIcon,
    PlayArrow as RerunIcon,
    Visibility as InspectIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    cloneRunParameters,
    compareRuns,
    getAnalysisCapabilities,
    getRun,
    getRunProvenance,
    listRuns,
    researchExportUrl,
    rerunRun,
    type ClonedRunParameters,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { HelpTooltip } from "../../../components/ui/HelpTooltip";
import { DataTable } from "../../../components/ui/DataTable";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ActiveRunActions } from "../components/ActiveRunActions";
import { ResultsInspector } from "../components/ResearchCharts";
import { FullResultsActions } from "../components/ResearchResults";
import { ProvenanceDrawer } from "../components/ProvenanceDrawer";
import { ProvenancePanel } from "../components/ProvenancePanel";
import { RunStatusChip } from "../../../components/ui/RunStatusChip";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { reproduceActionState } from "../reproduceAction";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import type { AnalysisRun } from "../types";

const RUNS_PAGE_SIZE_OPTIONS = [25, 50, 100];

function formatValue(value: unknown): string {
    if (value == null) return "—";
    if (typeof value === "string") return value || "—";
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

function ReproduceButton({
    run,
    pending,
    onReproduce,
}: {
    run: Pick<
        AnalysisRun,
        | "id"
        | "rerunnable"
        | "rerun_block_reason"
        | "replayable"
        | "exact_reproducible"
        | "exact_reproduce_block_reason"
    >;
    pending: boolean;
    onReproduce: (runId: string, exact: boolean) => void;
}) {
    const action = reproduceActionState(run);
    const button = (
        <span>
            <Button
                size="small"
                startIcon={<RerunIcon />}
                disabled={!action.replay.enabled || pending}
                onClick={() => onReproduce(run.id, false)}
            >
                Replay
            </Button>
        </span>
    );
    if (!action.replay.enabled && action.replay.reason) {
        return (
            <Tooltip title={action.replay.reason}>
                {button}
            </Tooltip>
        );
    }
    return (
        <>
            {button}
            <Tooltip title={action.exact.reason ?? "Reproduce with frozen original inputs."}>
                <span>
                    <Button
                        size="small"
                        startIcon={<RerunIcon />}
                        disabled={!action.exact.enabled || pending}
                        onClick={() => onReproduce(run.id, true)}
                    >
                        Exact reproduce
                    </Button>
                </span>
            </Tooltip>
        </>
    );
}

export default function RunsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [compareIds, setCompareIds] = useState<string[]>([]);
    const [clonedParams, setClonedParams] = useState<ClonedRunParameters | null>(null);
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(50);
    const [listCorpusId, setListCorpusId] = useState(ctx.selectedCorpusId);
    const [rerunAsync, setRerunAsync] = useState(true);
    const sseConnected = useRunEvents(selectedRunId, ctx.projectId);

    // Reset paging when corpus selection changes without setState-in-effect.
    if (listCorpusId !== ctx.selectedCorpusId) {
        setListCorpusId(ctx.selectedCorpusId);
        setPage(0);
    }

    const pageOffset = page * pageSize;

    const runsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId, undefined, {
            limit: pageSize,
            offset: pageOffset,
        }),
        queryFn: ({ signal }) =>
            listRuns(
                ctx.projectId,
                {
                    corpus_id: ctx.selectedCorpusId || undefined,
                    limit: pageSize,
                    offset: pageOffset,
                },
                signal
            ),
        enabled: Boolean(ctx.projectId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(selectedRunId ?? ""),
        queryFn: ({ signal }) => getRun(selectedRunId!, signal),
        enabled: Boolean(selectedRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });
    const capabilitiesQuery = useQuery({
        queryKey: ["text-research", "analysis-capabilities"],
        queryFn: ({ signal }) => getAnalysisCapabilities(signal),
    });

    const provenanceQuery = useQuery({
        queryKey: queryKeys.textResearch.runProvenance(selectedRunId ?? ""),
        queryFn: ({ signal }) => getRunProvenance(selectedRunId!, signal),
        enabled: Boolean(selectedRunId),
    });

    const compareReady = compareIds.length === 2;
    const compareQuery = useQuery({
        queryKey: ["text-research", "run-compare", compareIds[0], compareIds[1]],
        queryFn: ({ signal }) => compareRuns(compareIds[0], compareIds[1], signal),
        enabled: compareReady,
    });

    const invalidateRuns = async () => {
        await queryClient.invalidateQueries({
            queryKey: ["text-research", ctx.projectId, "runs"],
        });
    };

    const rerunMutation = useMutation({
        mutationFn: ({ runId, exact, runAsync }: { runId: string; exact: boolean; runAsync: boolean }) =>
            rerunRun(runId, runAsync, exact),
        onSuccess: async (run) => {
            setSelectedRunId(run.id);
            await invalidateRuns();
            showToast({ message: "Reproducible rerun started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Reproduce failed."),
                severity: "error",
            }),
    });

    const cloneMutation = useMutation({
        mutationFn: (runId: string) => cloneRunParameters(runId),
        onSuccess: (data) => {
            setClonedParams(data);
            showToast({ message: "Parameters cloned for inspection.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Clone failed."),
                severity: "error",
            }),
    });

    const runs = runsQuery.data?.items ?? [];
    const runsTotal = runsQuery.data?.total ?? 0;
    const selectedRun = runQuery.data;
    const supportsAsyncRerun = (run: AnalysisRun) =>
        capabilitiesQuery.data?.run_types[run.run_type]?.async === true;

    const toggleCompare = (runId: string) => {
        setCompareIds((prev) => {
            if (prev.includes(runId)) return prev.filter((id) => id !== runId);
            if (prev.length >= 2) return [prev[1], runId];
            return [...prev, runId];
        });
    };

    const changedParameters = useMemo(
        () => compareQuery.data?.changed_parameters ?? [],
        [compareQuery.data]
    );
    const changedMetrics = useMemo(
        () => compareQuery.data?.changed_metrics ?? [],
        [compareQuery.data]
    );

    const exportPath = selectedRunId
        ? researchExportUrl(`/research/runs/${selectedRunId}/export.json`)
        : null;

    return (
        <Stack spacing={2}>
            <SectionCard
                title={
                    <HelpTooltip termId="provenance" variant="label">
                        Runs & provenance
                    </HelpTooltip>
                }
                description="Inspect, reproduce, clone, cancel, and compare persisted research operations."
            >
                <QueryBoundary
                    isLoading={runsQuery.isLoading}
                    isError={runsQuery.isError}
                    error={runsQuery.error}
                    onRetry={() => void runsQuery.refetch()}
                >
                    {runs.length ? (
                        <DataTable
                            ariaLabel="Research runs"
                            columns={[
                                {
                                    id: "compare",
                                    label: "Compare",
                                    padding: "checkbox",
                                    hideable: false,
                                    render: (run) => (
                                        <Checkbox
                                            size="small"
                                            checked={compareIds.includes(run.id)}
                                            onChange={() => toggleCompare(run.id)}
                                            inputProps={{
                                                "aria-label": `Compare run ${run.id}`,
                                            }}
                                        />
                                    ),
                                },
                                {
                                    id: "type",
                                    label: "Type",
                                    sortable: true,
                                    sticky: "left",
                                    getSortValue: (run) => run.run_type,
                                    render: (run) => run.run_type,
                                },
                                {
                                    id: "status",
                                    label: "Status",
                                    sortable: true,
                                    getSortValue: (run) => run.status,
                                    render: (run) => <RunStatusChip status={run.status} />,
                                },
                                {
                                    id: "stage",
                                    label: "Stage",
                                    truncate: true,
                                    getSortValue: (run) => run.progress_stage ?? "",
                                    render: (run) => run.progress_stage ?? "—",
                                },
                                {
                                    id: "seed",
                                    label: "Seed",
                                    hideable: true,
                                    getSortValue: (run) => run.random_seed ?? "",
                                    render: (run) => run.random_seed ?? "—",
                                },
                                {
                                    id: "created",
                                    label: "Created",
                                    sortable: true,
                                    getSortValue: (run) => Date.parse(run.created_at),
                                    render: (run) =>
                                        new Date(run.created_at).toLocaleString(),
                                },
                                {
                                    id: "actions",
                                    label: "Actions",
                                    align: "right",
                                    sticky: "right",
                                    hideable: false,
                                    render: (run) => {
                                        const active = isActiveRunStatus(run.status);
                                        return (
                                            <Stack
                                                direction="row"
                                                spacing={0.5}
                                                justifyContent="flex-end"
                                                flexWrap="wrap"
                                                useFlexGap
                                                onClick={(event) => event.stopPropagation()}
                                            >
                                                <Button
                                                    size="small"
                                                    startIcon={<InspectIcon />}
                                                    onClick={() => setSelectedRunId(run.id)}
                                                >
                                                    Inspect
                                                </Button>
                                                <ReproduceButton
                                                    run={run}
                                                    pending={rerunMutation.isPending}
                                                    onReproduce={(runId, exact) =>
                                                        rerunMutation.mutate({
                                                            runId,
                                                            exact,
                                                            runAsync:
                                                                supportsAsyncRerun(run) &&
                                                                rerunAsync,
                                                        })
                                                    }
                                                />
                                                <Button
                                                    size="small"
                                                    startIcon={<CloneIcon />}
                                                    disabled={cloneMutation.isPending}
                                                    onClick={() => cloneMutation.mutate(run.id)}
                                                >
                                                    Clone
                                                </Button>
                                                {active ? (
                                                    <ActiveRunActions
                                                        run={run}
                                                        projectId={ctx.projectId}
                                                        corpusId={ctx.selectedCorpusId}
                                                    />
                                                ) : null}
                                            </Stack>
                                        );
                                    },
                                },
                            ]}
                            rows={runs}
                            getRowId={(run) => run.id}
                            density="compact"
                            showDensityToggle
                            showColumnVisibility
                            stickyHeader
                            stickyFirstColumn
                            clientSort
                            page={page}
                            pageSize={pageSize}
                            totalCount={runsTotal}
                            onPageChange={setPage}
                            onPageSizeChange={setPageSize}
                            rowsPerPageOptions={RUNS_PAGE_SIZE_OPTIONS}
                            selectedRowId={selectedRunId}
                            onRowClick={(run) => setSelectedRunId(run.id)}
                            emptyDescription="No persisted runs for this corpus yet."
                        />
                    ) : (
                        <Typography color="text.secondary">
                            No persisted runs for this corpus yet.
                        </Typography>
                    )}
                </QueryBoundary>
                {compareIds.length > 0 && compareIds.length < 2 ? (
                    <Alert severity="info" sx={{ mt: 2 }}>
                        Select a second run to compare against{" "}
                        <Typography component="span" variant="body2" fontFamily="monospace">
                            {compareIds[0]}
                        </Typography>
                        .
                    </Alert>
                ) : null}
            </SectionCard>

            {selectedRunId ? (
                <SectionCard title="Run detail" description="Live status, provenance fields, and results.">
                    <QueryBoundary
                        isLoading={runQuery.isLoading}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {selectedRun ? (
                            <Stack spacing={1.5}>
                                <Stack
                                    direction={{ xs: "column", sm: "row" }}
                                    spacing={1}
                                    alignItems={{ sm: "center" }}
                                    justifyContent="space-between"
                                >
                                    <Typography>
                                        {selectedRun.run_type} —{" "}
                                        <RunStatusChip status={selectedRun.status} />
                                    </Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        <ProvenanceDrawer
                                            run={selectedRun}
                                            detail={provenanceQuery.data}
                                            fetchDetail={false}
                                            actions={
                                                exportPath ? (
                                                    <Button
                                                        size="small"
                                                        startIcon={<DownloadIcon />}
                                                        onClick={() =>
                                                            window.open(exportPath, "_blank")
                                                        }
                                                    >
                                                        Export JSON
                                                    </Button>
                                                ) : null
                                            }
                                        />
                                        {exportPath ? (
                                            <Button
                                                size="small"
                                                startIcon={<DownloadIcon />}
                                                onClick={() => window.open(exportPath, "_blank")}
                                            >
                                                Export JSON
                                            </Button>
                                        ) : null}
                                        <ReproduceButton
                                            run={selectedRun}
                                            pending={rerunMutation.isPending}
                                            onReproduce={(runId, exact) =>
                                                rerunMutation.mutate({
                                                    runId,
                                                    exact,
                                                    runAsync: supportsAsyncRerun(selectedRun) && rerunAsync,
                                                })
                                            }
                                        />
                                        {isActiveRunStatus(selectedRun.status) ? (
                                            <ActiveRunActions
                                                run={selectedRun}
                                                projectId={ctx.projectId}
                                                corpusId={ctx.selectedCorpusId}
                                            />
                                        ) : null}
                                    </Stack>
                                </Stack>

                                <Typography variant="body2" color="text.secondary">
                                    ID: {selectedRun.id}
                                </Typography>
                                <Typography variant="body2">
                                    Progress stage: {selectedRun.progress_stage ?? "—"}
                                </Typography>
                                <Typography variant="body2">
                                    Seed: {selectedRun.random_seed ?? "—"}
                                </Typography>
                                <Typography variant="body2">
                                    Artifact: {selectedRun.artifact_path ?? "—"}
                                </Typography>
                                {supportsAsyncRerun(selectedRun) ? (
                                    <FormControlLabel
                                        control={
                                            <Checkbox
                                                checked={rerunAsync}
                                                onChange={(event) => setRerunAsync(event.target.checked)}
                                            />
                                        }
                                        label="Run replay asynchronously"
                                    />
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        Replay is inline only.
                                    </Typography>
                                )}
                                {selectedRun.error_message ? (
                                    <Alert severity="error">{selectedRun.error_message}</Alert>
                                ) : null}

                                {selectedRun.replayable === false || selectedRun.rerunnable === false ? (
                                    <Alert severity="warning">
                                        Replay is unavailable
                                        {selectedRun.rerun_block_reason
                                            ? `: ${selectedRun.rerun_block_reason}`
                                            : " for this run type."}
                                    </Alert>
                                ) : (
                                    <Alert severity="info">
                                        Replay re-runs the same settings against the current
                                        corpus and profiles. Exact reproduce uses frozen
                                        preprocessing and checksums from the original experiment.
                                    </Alert>
                                )}

                                <ProvenancePanel
                                    run={selectedRun}
                                    detail={provenanceQuery.data}
                                    showRaw
                                />

                                <ResultsInspector
                                    title="metrics & results"
                                    data={{
                                        metrics: selectedRun.metrics,
                                        results: selectedRun.results,
                                        error: selectedRun.error_message,
                                    }}
                                />
                                <FullResultsActions run={selectedRun} />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            {clonedParams ? (
                <SectionCard
                    title="Cloned parameters"
                    description={`Parameters from ${clonedParams.run_type} ready for reuse or inspection.`}
                >
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        Corpus: {clonedParams.corpus_id ?? "—"}
                    </Typography>
                    <ResultsInspector title="cloned parameters" data={clonedParams.parameters} />
                </SectionCard>
            ) : null}

            {compareReady ? (
                <SectionCard
                    title="Run comparison"
                    description="Changed parameters and metrics between the two selected runs."
                >
                    <QueryBoundary
                        isLoading={compareQuery.isLoading}
                        isError={compareQuery.isError}
                        error={compareQuery.error}
                        onRetry={() => void compareQuery.refetch()}
                    >
                        {compareQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2" color="text.secondary">
                                    A: {compareQuery.data.run_a.run_type} (
                                    {compareQuery.data.run_a.id.slice(0, 8)}…) —{" "}
                                    <RunStatusChip status={compareQuery.data.run_a.status} />
                                    {" · "}
                                    B: {compareQuery.data.run_b.run_type} (
                                    {compareQuery.data.run_b.id.slice(0, 8)}…) —{" "}
                                    <RunStatusChip status={compareQuery.data.run_b.status} />
                                </Typography>

                                <Typography variant="subtitle2">Changed parameters</Typography>
                                {changedParameters.length ? (
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Parameter</TableCell>
                                                <TableCell>Run A</TableCell>
                                                <TableCell>Run B</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {changedParameters.map((row) => (
                                                <TableRow key={String(row.parameter)}>
                                                    <TableCell>{row.parameter}</TableCell>
                                                    <TableCell sx={{ fontFamily: "monospace" }}>
                                                        {formatValue(row.run_a)}
                                                    </TableCell>
                                                    <TableCell sx={{ fontFamily: "monospace" }}>
                                                        {formatValue(row.run_b)}
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        No parameter differences.
                                    </Typography>
                                )}

                                <Typography variant="subtitle2">Changed metrics</Typography>
                                {changedMetrics.length ? (
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Metric</TableCell>
                                                <TableCell>Run A</TableCell>
                                                <TableCell>Run B</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {changedMetrics.map((row) => (
                                                <TableRow key={String(row.metric)}>
                                                    <TableCell>{row.metric}</TableCell>
                                                    <TableCell sx={{ fontFamily: "monospace" }}>
                                                        {formatValue(row.run_a)}
                                                    </TableCell>
                                                    <TableCell sx={{ fontFamily: "monospace" }}>
                                                        {formatValue(row.run_b)}
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        No metric differences.
                                    </Typography>
                                )}

                                <ResultsInspector title="full comparison" data={compareQuery.data} />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
