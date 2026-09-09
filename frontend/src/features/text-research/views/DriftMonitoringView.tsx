import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    MenuItem,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import {
    OpenInNew as OpenIcon,
    Science as DriftIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    compareClassifierDrift,
    getClassifierCoefficients,
    getRun,
    listModelPredictions,
    listModels,
    listRuns,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import {
    DRIFT_DISCLAIMER,
    parseDriftReport,
    topTermsFromCoefficients,
    withPerformanceSection,
    type DriftRowStatus,
} from "../driftDiagnostics";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { aggregatePredictionsForDrift, asRecord, extractMacroF1, modelDisplayName } from "../modelRegistryUtils";
import { activeRunRefetchInterval } from "../runPolling";

function statusColor(status: DriftRowStatus): "success" | "warning" | "error" | "default" {
    switch (status) {
        case "ok":
            return "success";
        case "watch":
            return "warning";
        case "investigate":
            return "error";
        default:
            return "default";
    }
}

export default function DriftMonitoringView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [baselineModelId, setBaselineModelId] = useState("");
    const [currentModelId, setCurrentModelId] = useState("");
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [liveReport, setLiveReport] = useState<Record<string, unknown> | null>(null);

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.models(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () =>
            listModels(ctx.projectId, {
                corpusId: ctx.selectedCorpusId || undefined,
            }),
        enabled: Boolean(ctx.projectId),
    });

    const models = modelsQuery.data ?? [];

    const driftRunsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(
            ctx.projectId,
            ctx.selectedCorpusId,
            "drift_monitoring"
        ),
        queryFn: () =>
            listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId || undefined,
                run_type: "drift_monitoring",
                limit: 30,
            }),
        enabled: Boolean(ctx.projectId),
    });

    const sseConnected = useRunEvents(selectedRunId, ctx.projectId);
    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(selectedRunId ?? ""),
        queryFn: () => getRun(selectedRunId!),
        enabled: Boolean(selectedRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const activeReport = useMemo(() => {
        if (liveReport) return liveReport;
        const results = asRecord(runQuery.data?.results);
        return results;
    }, [liveReport, runQuery.data]);

    const parsed = useMemo(() => parseDriftReport(activeReport), [activeReport]);

    const runDriftMutation = useMutation({
        mutationFn: async () => {
            if (!ctx.selectedCorpusId) {
                throw new Error("Select a corpus in the workspace context bar.");
            }
            if (!baselineModelId || !currentModelId) {
                throw new Error("Select baseline and current models.");
            }
            if (baselineModelId === currentModelId) {
                throw new Error("Baseline and current models must differ.");
            }

            const baselineModel = models.find((m) => m.id === baselineModelId);
            const currentModel = models.find((m) => m.id === currentModelId);
            if (!baselineModel || !currentModel) {
                throw new Error("Models not found in registry.");
            }

            const [baselineRows, currentRows, baselineCoefs, currentCoefs] = await Promise.all([
                listModelPredictions(baselineModelId, { limit: 500 }),
                listModelPredictions(currentModelId, { limit: 500 }),
                getClassifierCoefficients(baselineModelId).catch(() => []),
                getClassifierCoefficients(currentModelId).catch(() => []),
            ]);

            const baselineAgg = aggregatePredictionsForDrift(baselineRows);
            const currentAgg = aggregatePredictionsForDrift(currentRows);
            if (
                !Object.keys(baselineAgg.label_counts).length ||
                !Object.keys(currentAgg.label_counts).length
            ) {
                throw new Error(
                    "Both models need stored predictions. Run Predict from Model Registry first."
                );
            }

            const baselineTerms = topTermsFromCoefficients(baselineCoefs);
            const currentTerms = topTermsFromCoefficients(currentCoefs);

            const report = await compareClassifierDrift(ctx.selectedCorpusId, {
                mode: "MODEL_COMPARISON",
                baseline: {
                    label_counts: baselineAgg.label_counts,
                    scores: baselineAgg.scores,
                    top_terms: baselineTerms.length ? baselineTerms : undefined,
                },
                current: {
                    label_counts: currentAgg.label_counts,
                    scores: currentAgg.scores,
                    top_terms: currentTerms.length ? currentTerms : undefined,
                },
                baseline_run_id: baselineModel.analysis_run_id,
                current_run_id: currentModel.analysis_run_id,
            });

            return withPerformanceSection(report, {
                baseline_macro_f1: extractMacroF1(asRecord(baselineModel.metrics)),
                current_macro_f1: extractMacroF1(asRecord(currentModel.metrics)),
                source: "training/holdout metrics on each model record (when adjudicated/holdout eval exists)",
            });
        },
        onSuccess: async (report) => {
            setLiveReport(report);
            const runId =
                typeof report.analysis_run_id === "string" ? report.analysis_run_id : null;
            if (runId) setSelectedRunId(runId);
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.runs(
                    ctx.projectId,
                    ctx.selectedCorpusId,
                    "drift_monitoring"
                ),
            });
            showToast({
                message: "Drift report saved as analysis run.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Drift check failed."),
                severity: "error",
            }),
    });

    if (!ctx.projectId) {
        return <Alert severity="info">Select a research project to monitor drift.</Alert>;
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Drift monitoring"
                description="Model comparison across stored prediction samples. Use the PredictionSet API with an explicit drift mode for deployment/data drift."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${ctx.projectId}/models`)}
                    >
                        Model Registry
                    </Button>
                }
            >
                <Alert severity="info" sx={{ mb: 2 }}>
                    {DRIFT_DISCLAIMER}
                </Alert>

                {!ctx.selectedCorpusId ? (
                    <Alert severity="warning">
                        Select a corpus in the context bar before running drift checks.
                    </Alert>
                ) : null}

                <QueryBoundary
                    isLoading={modelsQuery.isLoading}
                    isError={modelsQuery.isError}
                    error={modelsQuery.error}
                    onRetry={() => void modelsQuery.refetch()}
                >
                    {!models.length ? (
                        <EmptyState
                            icon={<DriftIcon />}
                            title="No models"
                            description="Train classifiers, run predictions, then compare baseline vs current here."
                            action={
                                <Button
                                    variant="contained"
                                    onClick={() =>
                                        navigate(`/research/${ctx.projectId}/classification`)
                                    }
                                >
                                    Open Classification
                                </Button>
                            }
                        />
                    ) : (
                        <Stack spacing={1.5}>
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                                <TextField
                                    select
                                    size="small"
                                    label="Baseline model"
                                    value={baselineModelId}
                                    onChange={(e) => setBaselineModelId(e.target.value)}
                                    sx={{ minWidth: 240 }}
                                >
                                    <MenuItem value="">Select…</MenuItem>
                                    {models.map((model) => (
                                        <MenuItem key={model.id} value={model.id}>
                                            {modelDisplayName(model)} · {model.lifecycle_status}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    label="Current sample model"
                                    value={currentModelId}
                                    onChange={(e) => setCurrentModelId(e.target.value)}
                                    sx={{ minWidth: 240 }}
                                >
                                    <MenuItem value="">Select…</MenuItem>
                                    {models.map((model) => (
                                        <MenuItem key={model.id} value={model.id}>
                                            {modelDisplayName(model)} · {model.lifecycle_status}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <Button
                                    variant="contained"
                                    startIcon={<DriftIcon />}
                                    disabled={
                                        !ctx.selectedCorpusId ||
                                        !baselineModelId ||
                                        !currentModelId ||
                                        runDriftMutation.isPending
                                    }
                                    onClick={() => runDriftMutation.mutate()}
                                >
                                    Run model comparison
                                </Button>
                            </Stack>
                            <Typography variant="caption" color="text.secondary">
                                This is explicitly a model comparison, not a claim of deployment drift. It persists an auditable{" "}
                                <code>drift_monitoring</code> analysis run.
                            </Typography>
                        </Stack>
                    )}
                </QueryBoundary>
            </SectionCard>

            <SectionCard
                title="Diagnostics"
                description="Baseline vs current sample · metric · advisory threshold · review status"
            >
                {!parsed.rows.length ? (
                    <Typography variant="body2" color="text.secondary">
                        Run a drift check or open a prior drift analysis run to see diagnostics.
                    </Typography>
                ) : (
                    <Stack spacing={1.5}>
                        {parsed.analysisRunId ? (
                            <Typography variant="caption" color="text.secondary">
                                Analysis run: {parsed.analysisRunId}
                            </Typography>
                        ) : null}
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Kind</TableCell>
                                        <TableCell>Baseline</TableCell>
                                        <TableCell>Current sample</TableCell>
                                        <TableCell>Metric</TableCell>
                                        <TableCell align="right">Value</TableCell>
                                        <TableCell>Threshold</TableCell>
                                        <TableCell>Status</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {parsed.rows.map((row) => (
                                        <TableRow key={row.id}>
                                            <TableCell>
                                                <Typography variant="body2" fontWeight={600}>
                                                    {row.kindLabel}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {row.note}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>{row.baseline}</TableCell>
                                            <TableCell>{row.current}</TableCell>
                                            <TableCell>{row.metric}</TableCell>
                                            <TableCell align="right">{row.value}</TableCell>
                                            <TableCell>
                                                <Typography variant="caption">
                                                    {row.threshold}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>
                                                <Chip
                                                    size="small"
                                                    label={row.status}
                                                    color={statusColor(row.status)}
                                                    variant={
                                                        row.status === "ok" ? "filled" : "outlined"
                                                    }
                                                />
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                        {activeReport ? (
                            <ResultsInspector title="full drift report" data={activeReport} />
                        ) : null}
                    </Stack>
                )}
            </SectionCard>

            <SectionCard
                title="Drift analysis runs"
                description="Auditable drift_monitoring runs for this project/corpus."
            >
                <QueryBoundary
                    isLoading={driftRunsQuery.isLoading}
                    isError={driftRunsQuery.isError}
                    error={driftRunsQuery.error}
                    onRetry={() => void driftRunsQuery.refetch()}
                >
                    {!(driftRunsQuery.data?.items.length) ? (
                        <Typography variant="body2" color="text.secondary">
                            No drift runs yet.
                        </Typography>
                    ) : (
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Run</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell align="right">Open</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {driftRunsQuery.data!.items.map((run) => (
                                        <TableRow
                                            key={run.id}
                                            selected={selectedRunId === run.id}
                                        >
                                            <TableCell>{run.id.slice(0, 8)}…</TableCell>
                                            <TableCell>
                                                <RunStatusChip status={run.status} />
                                            </TableCell>
                                            <TableCell>
                                                {new Date(run.created_at).toLocaleString()}
                                            </TableCell>
                                            <TableCell align="right">
                                                <Button
                                                    size="small"
                                                    startIcon={<OpenIcon fontSize="small" />}
                                                    onClick={() => {
                                                        setLiveReport(null);
                                                        setSelectedRunId(run.id);
                                                    }}
                                                >
                                                    Inspect
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                    )}
                </QueryBoundary>
            </SectionCard>
        </Stack>
    );
}
