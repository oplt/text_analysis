import { useMemo, useState } from "react";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
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
    ExpandMore as ExpandIcon,
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
    listPredictionSets,
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
    predictionSetOptionLabel,
    topTermsFromCoefficients,
    withPerformanceSection,
    type DriftRowStatus,
} from "../driftDiagnostics";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { aggregatePredictionsForDrift, asRecord, extractMacroF1, modelDisplayName } from "../modelRegistryUtils";
import { activeRunRefetchInterval } from "../runPolling";

const DRIFT_MODES = [
    { value: "MODEL_COMPARISON", label: "Model comparison" },
    { value: "PREDICTION_DRIFT", label: "Prediction drift (same model)" },
    { value: "DATA_DRIFT", label: "Data drift" },
    { value: "PERFORMANCE_DRIFT", label: "Performance drift (labeled)" },
] as const;

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

    const [baselineSetId, setBaselineSetId] = useState("");
    const [currentSetId, setCurrentSetId] = useState("");
    const [driftMode, setDriftMode] = useState<(typeof DRIFT_MODES)[number]["value"]>(
        "MODEL_COMPARISON"
    );
    const [legacyBaselineModelId, setLegacyBaselineModelId] = useState("");
    const [legacyCurrentModelId, setLegacyCurrentModelId] = useState("");
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [liveReport, setLiveReport] = useState<Record<string, unknown> | null>(null);

    const predictionSetsQuery = useQuery({
        queryKey: queryKeys.textResearch.predictionSets(ctx.selectedCorpusId),
        queryFn: () => listPredictionSets(ctx.selectedCorpusId, { limit: 100 }),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.models(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listModels(ctx.projectId, {
                corpusId: ctx.selectedCorpusId || undefined,
            }, signal),
        enabled: Boolean(ctx.projectId),
    });

    const predictionSets = predictionSetsQuery.data ?? [];
    const models = modelsQuery.data ?? [];

    const driftRunsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(
            ctx.projectId,
            ctx.selectedCorpusId,
            "drift_monitoring"
        ),
        queryFn: ({ signal }) => listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId || undefined,
                run_type: "drift_monitoring",
                limit: 30,
            }, signal),
        enabled: Boolean(ctx.projectId),
    });

    const sseConnected = useRunEvents(selectedRunId, ctx.projectId);
    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(selectedRunId ?? ""),
        queryFn: ({ signal }) => getRun(selectedRunId!, signal),
        enabled: Boolean(selectedRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const activeReport = useMemo(() => {
        if (liveReport) return liveReport;
        const results = asRecord(runQuery.data?.results);
        return results;
    }, [liveReport, runQuery.data]);

    const parsed = useMemo(() => parseDriftReport(activeReport), [activeReport]);

    const runPredictionSetDriftMutation = useMutation({
        mutationFn: async () => {
            if (!ctx.selectedCorpusId) {
                throw new Error("Select a corpus in the workspace context bar.");
            }
            if (!baselineSetId || !currentSetId) {
                throw new Error("Select baseline and current prediction sets.");
            }
            if (baselineSetId === currentSetId) {
                throw new Error("Baseline and current prediction sets must differ.");
            }
            return compareClassifierDrift(ctx.selectedCorpusId, {
                mode: driftMode,
                baseline_prediction_set_id: baselineSetId,
                current_prediction_set_id: currentSetId,
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
                message: "Drift report saved from complete PredictionSets.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Drift check failed."),
                severity: "error",
            }),
    });

    const runLegacyDriftMutation = useMutation({
        mutationFn: async () => {
            if (!ctx.selectedCorpusId) {
                throw new Error("Select a corpus in the workspace context bar.");
            }
            if (!legacyBaselineModelId || !legacyCurrentModelId) {
                throw new Error("Select baseline and current models.");
            }
            if (legacyBaselineModelId === legacyCurrentModelId) {
                throw new Error("Baseline and current models must differ.");
            }

            const baselineModel = models.find((m) => m.id === legacyBaselineModelId);
            const currentModel = models.find((m) => m.id === legacyCurrentModelId);
            if (!baselineModel || !currentModel) {
                throw new Error("Models not found in registry.");
            }

            const [baselineRows, currentRows, baselineCoefs, currentCoefs] = await Promise.all([
                listModelPredictions(legacyBaselineModelId, { limit: 500 }),
                listModelPredictions(legacyCurrentModelId, { limit: 500 }),
                getClassifierCoefficients(legacyBaselineModelId).catch(() => []),
                getClassifierCoefficients(legacyCurrentModelId).catch(() => []),
            ]);

            const baselineAgg = aggregatePredictionsForDrift(baselineRows);
            const currentAgg = aggregatePredictionsForDrift(currentRows);
            if (
                !Object.keys(baselineAgg.label_counts).length ||
                !Object.keys(currentAgg.label_counts).length
            ) {
                throw new Error(
                    "Both models need stored predictions. Prefer PredictionSets above, or run Predict first."
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
                message: "Legacy/manual drift report saved (≤500 predictions per model).",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Legacy drift check failed."),
                severity: "error",
            }),
    });

    if (!ctx.projectId) {
        return <Alert severity="info">Select a research project to monitor drift.</Alert>;
    }

    const { provenance, warningLevel } = parsed;

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Drift monitoring"
                description="Compare complete persisted PredictionSets on the backend. The browser never downloads full prediction rows to aggregate."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${ctx.projectId}/predictions`)}
                    >
                        Prediction Sets
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
                    isLoading={predictionSetsQuery.isLoading}
                    isError={predictionSetsQuery.isError}
                    error={predictionSetsQuery.error}
                    onRetry={() => void predictionSetsQuery.refetch()}
                >
                    {!predictionSets.length ? (
                        <EmptyState
                            icon={<DriftIcon />}
                            title="No prediction sets"
                            description="Run Predict from Model Registry to create PredictionSets, then compare them here."
                            action={
                                <Button
                                    variant="contained"
                                    onClick={() =>
                                        navigate(`/research/${ctx.projectId}/models`)
                                    }
                                >
                                    Open Model Registry
                                </Button>
                            }
                        />
                    ) : (
                        <Stack spacing={1.5}>
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                                <TextField
                                    select
                                    size="small"
                                    label="Baseline prediction set"
                                    value={baselineSetId}
                                    onChange={(e) => setBaselineSetId(e.target.value)}
                                    sx={{ minWidth: 280, flex: 1 }}
                                >
                                    <MenuItem value="">Select…</MenuItem>
                                    {predictionSets.map((ps) => (
                                        <MenuItem key={ps.id} value={ps.id}>
                                            {predictionSetOptionLabel(ps)}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    label="Current prediction set"
                                    value={currentSetId}
                                    onChange={(e) => setCurrentSetId(e.target.value)}
                                    sx={{ minWidth: 280, flex: 1 }}
                                >
                                    <MenuItem value="">Select…</MenuItem>
                                    {predictionSets.map((ps) => (
                                        <MenuItem key={ps.id} value={ps.id}>
                                            {predictionSetOptionLabel(ps)}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    label="Drift mode"
                                    value={driftMode}
                                    onChange={(e) =>
                                        setDriftMode(
                                            e.target.value as (typeof DRIFT_MODES)[number]["value"]
                                        )
                                    }
                                    sx={{ minWidth: 220 }}
                                >
                                    {DRIFT_MODES.map((mode) => (
                                        <MenuItem key={mode.value} value={mode.value}>
                                            {mode.label}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <Button
                                    variant="contained"
                                    startIcon={<DriftIcon />}
                                    disabled={
                                        !ctx.selectedCorpusId ||
                                        !baselineSetId ||
                                        !currentSetId ||
                                        runPredictionSetDriftMutation.isPending
                                    }
                                    onClick={() => runPredictionSetDriftMutation.mutate()}
                                >
                                    Compare prediction sets
                                </Button>
                            </Stack>
                            <Typography variant="caption" color="text.secondary">
                                Backend aggregates <strong>all</strong> predictions in each set
                                (including &gt;500 rows) and persists an auditable{" "}
                                <code>drift_monitoring</code> analysis run.
                            </Typography>
                        </Stack>
                    )}
                </QueryBoundary>

                <Accordion disableGutters elevation={0} sx={{ mt: 2, border: "1px solid", borderColor: "divider" }}>
                    <AccordionSummary expandIcon={<ExpandIcon />}>
                        <Typography variant="subtitle2">
                            Legacy / manual — aggregate up to 500 predictions in the browser
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Alert severity="warning" sx={{ mb: 1.5 }}>
                            This path samples at most 500 predictions per model client-side. Prefer
                            PredictionSet comparison above for scientific completeness.
                        </Alert>
                        <QueryBoundary
                            isLoading={modelsQuery.isLoading}
                            isError={modelsQuery.isError}
                            error={modelsQuery.error}
                            onRetry={() => void modelsQuery.refetch()}
                        >
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                                <TextField
                                    select
                                    size="small"
                                    label="Baseline model"
                                    value={legacyBaselineModelId}
                                    onChange={(e) => setLegacyBaselineModelId(e.target.value)}
                                    sx={{ minWidth: 220 }}
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
                                    label="Current model"
                                    value={legacyCurrentModelId}
                                    onChange={(e) => setLegacyCurrentModelId(e.target.value)}
                                    sx={{ minWidth: 220 }}
                                >
                                    <MenuItem value="">Select…</MenuItem>
                                    {models.map((model) => (
                                        <MenuItem key={model.id} value={model.id}>
                                            {modelDisplayName(model)} · {model.lifecycle_status}
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <Button
                                    variant="outlined"
                                    disabled={
                                        !ctx.selectedCorpusId ||
                                        !legacyBaselineModelId ||
                                        !legacyCurrentModelId ||
                                        runLegacyDriftMutation.isPending
                                    }
                                    onClick={() => runLegacyDriftMutation.mutate()}
                                >
                                    Run legacy/manual drift
                                </Button>
                            </Stack>
                        </QueryBoundary>
                    </AccordionDetails>
                </Accordion>
            </SectionCard>

            <SectionCard
                title="Diagnostics"
                description="Baseline vs current · label-distribution · confidence/uncertainty · warning level · provenance"
            >
                {!parsed.rows.length ? (
                    <Typography variant="body2" color="text.secondary">
                        Run a drift check or open a prior drift analysis run to see diagnostics.
                    </Typography>
                ) : (
                    <Stack spacing={1.5}>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
                            <Chip
                                size="small"
                                label={`Warning: ${warningLevel}`}
                                color={statusColor(warningLevel)}
                            />
                            {provenance.mode ? (
                                <Chip size="small" variant="outlined" label={`Mode: ${provenance.mode}`} />
                            ) : null}
                            {provenance.nObservations != null ? (
                                <Chip
                                    size="small"
                                    variant="outlined"
                                    label={`n observations: ${provenance.nObservations}`}
                                />
                            ) : null}
                            {provenance.aggregation ? (
                                <Chip
                                    size="small"
                                    variant="outlined"
                                    label={provenance.aggregation}
                                />
                            ) : null}
                        </Stack>
                        <Box>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Baseline set: {provenance.baselinePredictionSetId ?? "—"} · n=
                                {provenance.nBaseline ?? "—"} · run{" "}
                                {provenance.baselineAnalysisRunId ?? "—"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Current set: {provenance.currentPredictionSetId ?? "—"} · n=
                                {provenance.nCurrent ?? "—"} · run{" "}
                                {provenance.currentAnalysisRunId ?? "—"}
                            </Typography>
                            {parsed.analysisRunId ? (
                                <Typography variant="caption" color="text.secondary" display="block">
                                    Drift analysis run: {parsed.analysisRunId}
                                </Typography>
                            ) : null}
                        </Box>
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Kind</TableCell>
                                        <TableCell>Baseline</TableCell>
                                        <TableCell>Current</TableCell>
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
