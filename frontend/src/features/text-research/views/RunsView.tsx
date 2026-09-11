import { useMemo, useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import {
    ContentCopy as CloneIcon,
    Download as DownloadIcon,
    PlayArrow as RerunIcon,
    Stop as CancelIcon,
    Visibility as InspectIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    cancelRun,
    cloneRunParameters,
    compareRuns,
    getRun,
    getRunProvenance,
    listRuns,
    researchExportUrl,
    rerunRun,
    type ClonedRunParameters,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";

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

export default function RunsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [compareIds, setCompareIds] = useState<string[]>([]);
    const [clonedParams, setClonedParams] = useState<ClonedRunParameters | null>(null);
    const sseConnected = useRunEvents(selectedRunId, ctx.projectId);

    const runsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () =>
            listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId || undefined,
                limit: 100,
            }),
        enabled: Boolean(ctx.projectId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(selectedRunId ?? ""),
        queryFn: () => getRun(selectedRunId!),
        enabled: Boolean(selectedRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const provenanceQuery = useQuery({
        queryKey: queryKeys.textResearch.runProvenance(selectedRunId ?? ""),
        queryFn: () => getRunProvenance(selectedRunId!),
        enabled: Boolean(selectedRunId),
    });

    const compareReady = compareIds.length === 2;
    const compareQuery = useQuery({
        queryKey: ["text-research", "run-compare", compareIds[0], compareIds[1]],
        queryFn: () => compareRuns(compareIds[0], compareIds[1]),
        enabled: compareReady,
    });

    const invalidateRuns = async () => {
        await queryClient.invalidateQueries({
            queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId),
        });
    };

    const rerunMutation = useMutation({
        mutationFn: (runId: string) => rerunRun(runId, true),
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

    const cancelMutation = useMutation({
        mutationFn: (runId: string) => cancelRun(runId),
        onSuccess: async (run) => {
            setSelectedRunId(run.id);
            await invalidateRuns();
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.run(run.id),
            });
            showToast({ message: "Run cancelled.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Cancel failed."),
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
    const selectedRun = runQuery.data;

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
                title="Runs & provenance"
                description="Inspect, reproduce, clone, cancel, and compare persisted research operations."
            >
                <QueryBoundary
                    isLoading={runsQuery.isLoading}
                    isError={runsQuery.isError}
                    error={runsQuery.error}
                    onRetry={() => void runsQuery.refetch()}
                >
                    {runs.length ? (
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell padding="checkbox">Compare</TableCell>
                                    <TableCell>Type</TableCell>
                                    <TableCell>Status</TableCell>
                                    <TableCell>Stage</TableCell>
                                    <TableCell>Seed</TableCell>
                                    <TableCell>Created</TableCell>
                                    <TableCell align="right">Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {runs.map((run) => {
                                    const active = isActiveRunStatus(run.status);
                                    const comparing = compareIds.includes(run.id);
                                    return (
                                        <TableRow
                                            key={run.id}
                                            selected={selectedRunId === run.id}
                                            hover
                                        >
                                            <TableCell padding="checkbox">
                                                <Checkbox
                                                    size="small"
                                                    checked={comparing}
                                                    onChange={() => toggleCompare(run.id)}
                                                    inputProps={{
                                                        "aria-label": `Compare run ${run.id}`,
                                                    }}
                                                />
                                            </TableCell>
                                            <TableCell>{run.run_type}</TableCell>
                                            <TableCell>
                                                <RunStatusChip status={run.status} />
                                            </TableCell>
                                            <TableCell>
                                                {run.progress_stage ?? "—"}
                                            </TableCell>
                                            <TableCell>
                                                {run.random_seed ?? "—"}
                                            </TableCell>
                                            <TableCell>
                                                {new Date(run.created_at).toLocaleString()}
                                            </TableCell>
                                            <TableCell align="right">
                                                <Stack
                                                    direction="row"
                                                    spacing={0.5}
                                                    justifyContent="flex-end"
                                                    flexWrap="wrap"
                                                    useFlexGap
                                                >
                                                    <Button
                                                        size="small"
                                                        startIcon={<InspectIcon />}
                                                        onClick={() => setSelectedRunId(run.id)}
                                                    >
                                                        Inspect
                                                    </Button>
                                                    <Button
                                                        size="small"
                                                        startIcon={<RerunIcon />}
                                                        disabled={rerunMutation.isPending}
                                                        onClick={() =>
                                                            rerunMutation.mutate(run.id)
                                                        }
                                                    >
                                                        Reproduce
                                                    </Button>
                                                    <Button
                                                        size="small"
                                                        startIcon={<CloneIcon />}
                                                        disabled={cloneMutation.isPending}
                                                        onClick={() =>
                                                            cloneMutation.mutate(run.id)
                                                        }
                                                    >
                                                        Clone
                                                    </Button>
                                                    {active ? (
                                                        <Button
                                                            size="small"
                                                            color="warning"
                                                            startIcon={<CancelIcon />}
                                                            disabled={cancelMutation.isPending}
                                                            onClick={() =>
                                                                cancelMutation.mutate(run.id)
                                                            }
                                                        >
                                                            Cancel
                                                        </Button>
                                                    ) : null}
                                                </Stack>
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
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
                                        {exportPath ? (
                                            <Button
                                                size="small"
                                                startIcon={<DownloadIcon />}
                                                onClick={() => window.open(exportPath, "_blank")}
                                            >
                                                Export JSON
                                            </Button>
                                        ) : null}
                                        <Button
                                            size="small"
                                            startIcon={<RerunIcon />}
                                            disabled={rerunMutation.isPending}
                                            onClick={() =>
                                                rerunMutation.mutate(selectedRun.id)
                                            }
                                        >
                                            Reproduce
                                        </Button>
                                        {isActiveRunStatus(selectedRun.status) ? (
                                            <Button
                                                size="small"
                                                color="warning"
                                                startIcon={<CancelIcon />}
                                                disabled={cancelMutation.isPending}
                                                onClick={() =>
                                                    cancelMutation.mutate(selectedRun.id)
                                                }
                                            >
                                                Cancel
                                            </Button>
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
                                {selectedRun.error_message ? (
                                    <Alert severity="error">{selectedRun.error_message}</Alert>
                                ) : null}

                                <Alert severity="info">
                                    Use <strong>Reproduce</strong> for one-click re-execution with the
                                    original parameters, seeds, and analysis specification.
                                </Alert>

                                {provenanceQuery.data ? (
                                    <ResultsInspector
                                        title="provenance"
                                        data={{
                                            corpus_checksum:
                                                provenanceQuery.data.provenance?.corpus_checksum,
                                            pipeline_checksum:
                                                provenanceQuery.data.provenance?.pipeline_checksum,
                                            analysis_spec_hash:
                                                provenanceQuery.data.reproduce?.analysis_spec_hash,
                                            evidence_revision_hash:
                                                provenanceQuery.data.provenance
                                                    ?.evidence_revision_hash,
                                            synthesis_provenance:
                                                provenanceQuery.data.provenance
                                                    ?.synthesis_provenance,
                                            git_commit:
                                                provenanceQuery.data.provenance?.git_commit,
                                            container_image_digest:
                                                provenanceQuery.data.provenance
                                                    ?.container_image_digest,
                                            package_versions:
                                                provenanceQuery.data.provenance?.package_versions,
                                            nlp_model: provenanceQuery.data.provenance?.nlp_model,
                                            implementation_version:
                                                provenanceQuery.data.provenance
                                                    ?.implementation_version,
                                            random_seeds:
                                                provenanceQuery.data.provenance?.random_seeds,
                                            parent_artifact_checksums:
                                                provenanceQuery.data.provenance
                                                    ?.parent_artifact_checksums,
                                            analysis_specification:
                                                provenanceQuery.data.reproduce
                                                    ?.analysis_specification,
                                            reproduce: provenanceQuery.data.reproduce,
                                        }}
                                    />
                                ) : null}

                                <ResultsInspector
                                    title="parameters"
                                    data={selectedRun.parameters ?? {}}
                                />
                                <ResultsInspector
                                    title="metrics & results"
                                    data={{
                                        metrics: selectedRun.metrics,
                                        results: selectedRun.results,
                                        error: selectedRun.error_message,
                                    }}
                                />
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
