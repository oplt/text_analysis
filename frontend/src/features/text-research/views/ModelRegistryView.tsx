import { lazy, Suspense, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    Chip,
    MenuItem,
    Skeleton,
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
    Archive as ArchiveIcon,
    ContentCopy as CloneIcon,
    CompareArrows as CompareIcon,
    PlayArrow as PredictIcon,
    Science as DriftIcon,
    TrendingUp as PromoteIcon,
    Visibility as OpenIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    cloneClassifierConfig,
    compareClassifierDrift,
    getClassifier,
    getClassifierCoefficients,
    getRunProvenance,
    listModelLifecycleEvents,
    listModels,
    listPredictionSets,
    predictClassifier,
    updateModelLifecycle,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ResultsInspector } from "../components/ResearchCharts";
import { ResearchResultsTable } from "../components/ResearchResults";
import { parseDriftReport, predictionSetOptionLabel } from "../driftDiagnostics";
import { useResearchContext } from "../hooks/useResearchContext";
import {
    asRecord,
    extractConfidenceIntervals,
    extractMacroF1,
    formatMetric,
    lifecycleChipColor,
    lifecycleDisplayLabel,
    modelDisplayName,
    num,
} from "../modelRegistryUtils";
import type { UnitType } from "../types";

const ClassificationCalibrationPanel = lazy(() =>
    import("../components/ClassificationEvalPanels").then((m) => ({
        default: m.ClassificationCalibrationPanel,
    }))
);
const ClassificationModelComparison = lazy(() =>
    import("../components/ClassificationEvalPanels").then((m) => ({
        default: m.ClassificationModelComparison,
    }))
);

function EvalPanelFallback() {
    return <Skeleton variant="rounded" height={160} sx={{ borderRadius: 2 }} />;
}

const LIFECYCLE_FILTERS = [
    { value: "", label: "All statuses" },
    { value: "candidate", label: "Candidate" },
    { value: "staging", label: "Staging" },
    { value: "production", label: "Production" },
    { value: "deprecated", label: "Deprecated" },
    { value: "archived", label: "Archived" },
] as const;

export default function ModelRegistryView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [lifecycleFilter, setLifecycleFilter] = useState("");
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const [compareIds, setCompareIds] = useState<string[]>([]);
    const [baselinePredictionSetId, setBaselinePredictionSetId] = useState("");
    const [currentPredictionSetId, setCurrentPredictionSetId] = useState("");
    const [predictUnitType, setPredictUnitType] = useState<UnitType>(
        ctx.unitType || "paragraph"
    );
    const [clonedConfig, setClonedConfig] = useState<Record<string, unknown> | null>(null);
    const [driftReport, setDriftReport] = useState<Record<string, unknown> | null>(null);

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.models(
            ctx.projectId,
            ctx.selectedCorpusId,
            lifecycleFilter || undefined
        ),
        queryFn: () =>
            listModels(ctx.projectId, {
                corpusId: ctx.selectedCorpusId || undefined,
                lifecycleStatus: lifecycleFilter || undefined,
            }),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchModelLifecycle,
    });

    const models = modelsQuery.data ?? [];
    const selectedModelIdSafe =
        selectedModelId && models.some((m) => m.id === selectedModelId)
            ? selectedModelId
            : (models[0]?.id ?? null);

    const modelQuery = useQuery({
        queryKey: queryKeys.textResearch.model(selectedModelIdSafe ?? ""),
        queryFn: () => getClassifier(selectedModelIdSafe!),
        enabled: Boolean(selectedModelIdSafe),
        staleTime: QUERY_STALE_TIMES.researchModelLifecycle,
    });

    const selectedModel = modelQuery.data ?? models.find((m) => m.id === selectedModelIdSafe) ?? null;

    const coefficientsQuery = useQuery({
        queryKey: queryKeys.textResearch.coefficients(selectedModelIdSafe ?? ""),
        queryFn: () => getClassifierCoefficients(selectedModelIdSafe!),
        enabled: Boolean(selectedModelIdSafe),
        staleTime: QUERY_STALE_TIMES.researchModelMetrics,
    });

    const provenanceQuery = useQuery({
        queryKey: queryKeys.textResearch.runProvenance(selectedModel?.analysis_run_id ?? ""),
        queryFn: () => getRunProvenance(selectedModel!.analysis_run_id),
        enabled: Boolean(selectedModel?.analysis_run_id),
        staleTime: QUERY_STALE_TIMES.researchProvenance,
    });

    const predictionSetsQuery = useQuery({
        queryKey: queryKeys.textResearch.predictionSets(selectedModel?.corpus_id ?? ""),
        queryFn: () => listPredictionSets(selectedModel!.corpus_id, { limit: 50 }),
        enabled: Boolean(selectedModel?.corpus_id),
        staleTime: QUERY_STALE_TIMES.researchPredictionSets,
    });

    const lifecycleEventsQuery = useQuery({
        queryKey: queryKeys.textResearch.modelLifecycleEvents(selectedModelIdSafe ?? ""),
        queryFn: () => listModelLifecycleEvents(selectedModelIdSafe!),
        enabled: Boolean(selectedModelIdSafe),
        staleTime: QUERY_STALE_TIMES.researchModelLifecycle,
    });

    const predictionSets = predictionSetsQuery.data ?? [];
    const modelPredictionSets = selectedModelIdSafe
        ? predictionSets.filter((item) => item.trained_model_id === selectedModelIdSafe)
        : predictionSets;

    const invalidateModels = async () => {
        await queryClient.invalidateQueries({
            queryKey: queryKeys.textResearch.modelsRoot(ctx.projectId),
        });
        await queryClient.invalidateQueries({
            queryKey: queryKeys.textResearch.classifiersRoot(ctx.projectId),
        });
        if (selectedModelIdSafe) {
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.model(selectedModelIdSafe),
            });
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.modelLifecycleEvents(selectedModelIdSafe),
            });
        }
    };

    const lifecycleMutation = useMutation({
        mutationFn: (payload: {
            status: "candidate" | "staging" | "production" | "deprecated" | "archived";
            notes?: string;
            deprecate_others?: boolean;
        }) => updateModelLifecycle(selectedModelIdSafe!, payload),
        onSuccess: async (model) => {
            await invalidateModels();
            setSelectedModelId(model.id);
            showToast({
                message: `Lifecycle → ${lifecycleDisplayLabel(model.lifecycle_status)}.`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Lifecycle update failed."),
                severity: "error",
            }),
    });

    const cloneMutation = useMutation({
        mutationFn: () => cloneClassifierConfig(selectedModelIdSafe!),
        onSuccess: (config) => {
            setClonedConfig(config);
            showToast({ message: "Training config cloned.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Clone failed."),
                severity: "error",
            }),
    });

    const predictMutation = useMutation({
        mutationFn: () =>
            predictClassifier(selectedModelIdSafe!, {
                unit_type: predictUnitType,
                only_unannotated: false,
            }),
        onSuccess: async (run) => {
            await queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.predictionSets(selectedModel?.corpus_id ?? ""),
            });
            showToast({
                message: `Prediction run ${run.id.slice(0, 8)}… started (${run.status}).`,
                severity: "success",
            });
            navigate(`/research/${ctx.projectId}/runs`);
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Predict failed."),
                severity: "error",
            }),
    });

    const driftMutation = useMutation({
        mutationFn: async () => {
            if (!selectedModel) {
                throw new Error("Select a model first.");
            }
            if (!baselinePredictionSetId || !currentPredictionSetId) {
                throw new Error("Select baseline and current prediction sets.");
            }
            if (baselinePredictionSetId === currentPredictionSetId) {
                throw new Error("Baseline and current prediction sets must differ.");
            }
            return compareClassifierDrift(selectedModel.corpus_id, {
                mode: "MODEL_COMPARISON",
                baseline_prediction_set_id: baselinePredictionSetId,
                current_prediction_set_id: currentPredictionSetId,
            });
        },
        onSuccess: (report) => {
            setDriftReport(report);
            showToast({
                message: "Drift report computed from complete PredictionSets.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Drift check failed."),
                severity: "error",
            }),
    });

    const toggleCompare = (modelId: string) => {
        setCompareIds((prev) => {
            if (prev.includes(modelId)) return prev.filter((id) => id !== modelId);
            if (prev.length >= 4) return [...prev.slice(1), modelId];
            return [...prev, modelId];
        });
    };

    const metrics = asRecord(selectedModel?.metrics);
    const featureConfig = asRecord(selectedModel?.feature_config) ?? {};
    const trainingConfig = asRecord(selectedModel?.training_config) ?? {};
    const selectionConfig =
        asRecord(featureConfig.feature_selection) ??
        asRecord(trainingConfig.feature_selection) ??
        asRecord(metrics?.feature_selection);
    const cis = extractConfidenceIntervals(metrics);
    const calibration =
        asRecord(metrics?.calibration) ?? asRecord(metrics?.calibration_summary);

    const parsedDrift = parseDriftReport(driftReport);

    if (!ctx.projectId) {
        return <Alert severity="info">Select a research project to open the model registry.</Alert>;
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Model Registry"
                description="Lifecycle, metrics, prediction sets, and drift entry points for trained classifiers."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${ctx.projectId}/classification?tab=train`)}
                    >
                        Train new
                    </Button>
                }
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} sx={{ mb: 2 }}>
                    <TextField
                        select
                        size="small"
                        label="Lifecycle"
                        value={lifecycleFilter}
                        onChange={(e) => setLifecycleFilter(e.target.value)}
                        sx={{ minWidth: 180 }}
                    >
                        {LIFECYCLE_FILTERS.map((opt) => (
                            <MenuItem key={opt.value || "all"} value={opt.value}>
                                {opt.label}
                            </MenuItem>
                        ))}
                    </TextField>
                    <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center" }}>
                        {ctx.selectedCorpusId
                            ? "Filtered to selected corpus."
                            : "Showing models across all corpora in this project."}
                    </Typography>
                </Stack>

                <QueryBoundary
                    isLoading={modelsQuery.isLoading}
                    isError={modelsQuery.isError}
                    error={modelsQuery.error}
                    onRetry={() => void modelsQuery.refetch()}
                >
                    {!models.length ? (
                        <EmptyState
                            icon={<DriftIcon />}
                            title="No trained models"
                            description="Train a classifier from Classification, then manage lifecycle here."
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
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell padding="checkbox">Cmp</TableCell>
                                        <TableCell>Model</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell align="right">Macro F1</TableCell>
                                        <TableCell>Dataset</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell align="right">Open</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {models.map((model) => {
                                        const macro = extractMacroF1(asRecord(model.metrics));
                                        const selected = model.id === selectedModelIdSafe;
                                        return (
                                            <TableRow key={model.id} selected={selected} hover>
                                                <TableCell padding="checkbox">
                                                    <Checkbox
                                                        size="small"
                                                        checked={compareIds.includes(model.id)}
                                                        onChange={() => toggleCompare(model.id)}
                                                        inputProps={{
                                                            "aria-label": `Compare ${modelDisplayName(model)}`,
                                                        }}
                                                    />
                                                </TableCell>
                                                <TableCell>
                                                    <Typography variant="body2" fontWeight={600}>
                                                        {modelDisplayName(model)}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        {model.model_family} · {model.task_type} · v
                                                        {model.version}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell>
                                                    <Chip
                                                        size="small"
                                                        label={lifecycleDisplayLabel(
                                                            model.lifecycle_status
                                                        )}
                                                        color={lifecycleChipColor(
                                                            model.lifecycle_status
                                                        )}
                                                        variant={
                                                            model.lifecycle_status === "production"
                                                                ? "filled"
                                                                : "outlined"
                                                        }
                                                    />
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(macro)}
                                                </TableCell>
                                                <TableCell>
                                                    <Typography variant="caption" component="span">
                                                        {model.training_dataset_snapshot_id.slice(0, 8)}
                                                        …
                                                    </Typography>
                                                </TableCell>
                                                <TableCell>
                                                    {new Date(model.created_at).toLocaleString()}
                                                </TableCell>
                                                <TableCell align="right">
                                                    <Button
                                                        size="small"
                                                        variant={selected ? "contained" : "outlined"}
                                                        startIcon={<OpenIcon fontSize="small" />}
                                                        onClick={() => {
                                                            setSelectedModelId(model.id);
                                                            setDriftReport(null);
                                                        }}
                                                    >
                                                        Detail
                                                    </Button>
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </Box>
                    )}
                </QueryBoundary>
            </SectionCard>

            {compareIds.length >= 2 ? (
                <SectionCard title="Compare models" description="Holdout metrics side by side.">
                    <Suspense fallback={<EvalPanelFallback />}>
                        <ClassificationModelComparison
                            models={models}
                            selectedIds={compareIds}
                            onToggle={toggleCompare}
                        />
                    </Suspense>
                </SectionCard>
            ) : null}

            {selectedModel ? (
                <SectionCard
                    title={modelDisplayName(selectedModel)}
                    description="Metadata, metrics, coefficients, prediction sets, and lifecycle actions."
                >
                    <QueryBoundary
                        isLoading={modelQuery.isLoading}
                        isError={modelQuery.isError}
                        error={modelQuery.error}
                        onRetry={() => void modelQuery.refetch()}
                    >
                        <Stack spacing={2.5}>
                            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                <Chip
                                    label={lifecycleDisplayLabel(selectedModel.lifecycle_status)}
                                    color={lifecycleChipColor(selectedModel.lifecycle_status)}
                                    size="small"
                                />
                                <Chip label={selectedModel.task_type} size="small" variant="outlined" />
                                <Chip
                                    label={selectedModel.model_family}
                                    size="small"
                                    variant="outlined"
                                />
                                <Chip
                                    label={`v${selectedModel.version}`}
                                    size="small"
                                    variant="outlined"
                                />
                            </Stack>

                            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                <Button
                                    size="small"
                                    variant="contained"
                                    startIcon={<PromoteIcon />}
                                    disabled={
                                        lifecycleMutation.isPending ||
                                        selectedModel.lifecycle_status === "production"
                                    }
                                    onClick={() =>
                                        lifecycleMutation.mutate({
                                            status: "production",
                                            notes: "Promoted to production from Model Registry",
                                            deprecate_others: true,
                                        })
                                    }
                                >
                                    Promote to production
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    disabled={
                                        lifecycleMutation.isPending ||
                                        selectedModel.lifecycle_status === "deprecated"
                                    }
                                    onClick={() =>
                                        lifecycleMutation.mutate({
                                            status: "deprecated",
                                            notes: "Deprecated from Model Registry",
                                        })
                                    }
                                >
                                    Deprecate
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<ArchiveIcon />}
                                    disabled={
                                        lifecycleMutation.isPending ||
                                        selectedModel.lifecycle_status === "archived"
                                    }
                                    onClick={() =>
                                        lifecycleMutation.mutate({
                                            status: "archived",
                                            notes: "Archived from Model Registry",
                                        })
                                    }
                                >
                                    Archive
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() =>
                                        navigate(
                                            `/research/${ctx.projectId}/active-learning${
                                                selectedModelIdSafe
                                                    ? `?modelId=${selectedModelIdSafe}`
                                                    : ""
                                            }`
                                        )
                                    }
                                >
                                    Active learning
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<CloneIcon />}
                                    disabled={cloneMutation.isPending}
                                    onClick={() => cloneMutation.mutate()}
                                >
                                    Clone config
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<CompareIcon />}
                                    onClick={() => toggleCompare(selectedModel.id)}
                                >
                                    {compareIds.includes(selectedModel.id)
                                        ? "In compare set"
                                        : "Add to compare"}
                                </Button>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() =>
                                        cloneClassifierConfig(selectedModel.id)
                                            .then((config) =>
                                                navigate(
                                                    `/research/${ctx.projectId}/classification?tab=train`,
                                                    { state: { retrainConfig: config } }
                                                )
                                            )
                                            .catch((error: unknown) =>
                                                showToast({
                                                    message: getQueryErrorMessage(error, "Retrain setup failed."),
                                                    severity: "error",
                                                })
                                            )
                                    }
                                >
                                    Retrain
                                </Button>
                            </Stack>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Predict
                                </Typography>
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <TextField
                                        select
                                        size="small"
                                        label="Unit type"
                                        value={predictUnitType}
                                        onChange={(e) =>
                                            setPredictUnitType(e.target.value as UnitType)
                                        }
                                        sx={{ minWidth: 160 }}
                                    >
                                        {(["document", "paragraph", "sentence"] as UnitType[]).map(
                                            (unit) => (
                                                <MenuItem key={unit} value={unit}>
                                                    {unit}
                                                </MenuItem>
                                            )
                                        )}
                                    </TextField>
                                    <Button
                                        size="small"
                                        variant="contained"
                                        startIcon={<PredictIcon />}
                                        disabled={predictMutation.isPending}
                                        onClick={() => predictMutation.mutate()}
                                    >
                                        Predict
                                    </Button>
                                </Stack>
                            </Box>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Run drift check
                                </Typography>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    display="block"
                                    sx={{ mb: 1 }}
                                >
                                    Backend compares entire PredictionSets (not a browser sample of
                                    500 rows).
                                </Typography>
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <TextField
                                        select
                                        size="small"
                                        label="Baseline prediction set"
                                        value={baselinePredictionSetId}
                                        onChange={(e) =>
                                            setBaselinePredictionSetId(e.target.value)
                                        }
                                        sx={{ minWidth: 240, flex: 1 }}
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
                                        value={currentPredictionSetId}
                                        onChange={(e) =>
                                            setCurrentPredictionSetId(e.target.value)
                                        }
                                        sx={{ minWidth: 240, flex: 1 }}
                                    >
                                        <MenuItem value="">Select…</MenuItem>
                                        {predictionSets.map((ps) => (
                                            <MenuItem key={ps.id} value={ps.id}>
                                                {predictionSetOptionLabel(ps)}
                                            </MenuItem>
                                        ))}
                                    </TextField>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        startIcon={<DriftIcon />}
                                        disabled={
                                            !baselinePredictionSetId ||
                                            !currentPredictionSetId ||
                                            driftMutation.isPending
                                        }
                                        onClick={() => driftMutation.mutate()}
                                    >
                                        Compare sets
                                    </Button>
                                    <Button
                                        size="small"
                                        variant="text"
                                        onClick={() =>
                                            navigate(`/research/${ctx.projectId}/drift`)
                                        }
                                    >
                                        Open Drift panel
                                    </Button>
                                </Stack>
                                {driftReport ? (
                                    <Box sx={{ mt: 1.5 }}>
                                        <Stack
                                            direction="row"
                                            spacing={1}
                                            flexWrap="wrap"
                                            useFlexGap
                                            sx={{ mb: 1 }}
                                        >
                                            <Chip
                                                size="small"
                                                label={`Warning: ${parsedDrift.warningLevel}`}
                                                color={
                                                    parsedDrift.warningLevel === "investigate"
                                                        ? "error"
                                                        : parsedDrift.warningLevel === "watch"
                                                          ? "warning"
                                                          : "success"
                                                }
                                            />
                                            {parsedDrift.provenance.nObservations != null ? (
                                                <Chip
                                                    size="small"
                                                    variant="outlined"
                                                    label={`n=${parsedDrift.provenance.nObservations}`}
                                                />
                                            ) : null}
                                            {parsedDrift.analysisRunId ? (
                                                <Chip
                                                    size="small"
                                                    variant="outlined"
                                                    label={`run ${parsedDrift.analysisRunId.slice(0, 8)}…`}
                                                />
                                            ) : null}
                                        </Stack>
                                        <ResultsInspector title="drift report" data={driftReport} />
                                    </Box>
                                ) : null}
                            </Box>

                            <Stack spacing={0.5}>
                                <Typography variant="subtitle2">Lifecycle</Typography>
                                <Typography variant="body2">
                                    Status: {lifecycleDisplayLabel(selectedModel.lifecycle_status)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Updated:{" "}
                                    {selectedModel.lifecycle_updated_at
                                        ? new Date(
                                              selectedModel.lifecycle_updated_at
                                          ).toLocaleString()
                                        : "—"}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Notes: {selectedModel.lifecycle_notes || "—"}
                                </Typography>
                                <Stack
                                    direction="row"
                                    spacing={0.75}
                                    flexWrap="wrap"
                                    useFlexGap
                                    sx={{ py: 1 }}
                                    aria-label="Lifecycle path"
                                >
                                    {(["candidate", "staging", "production", "deprecated"] as const).map(
                                        (step, index) => (
                                            <Stack
                                                key={step}
                                                direction="row"
                                                spacing={0.75}
                                                alignItems="center"
                                            >
                                                {index > 0 ? (
                                                    <Typography variant="caption" color="text.secondary">
                                                        →
                                                    </Typography>
                                                ) : null}
                                                <Chip
                                                    size="small"
                                                    label={lifecycleDisplayLabel(step)}
                                                    color={
                                                        selectedModel.lifecycle_status === step
                                                            ? lifecycleChipColor(step)
                                                            : "default"
                                                    }
                                                    variant={
                                                        selectedModel.lifecycle_status === step
                                                            ? "filled"
                                                            : "outlined"
                                                    }
                                                />
                                            </Stack>
                                        )
                                    )}
                                </Stack>
                                <Typography variant="subtitle2" sx={{ pt: 1 }}>
                                    Lifecycle event timeline
                                </Typography>
                                <QueryBoundary
                                    isLoading={lifecycleEventsQuery.isLoading}
                                    isError={lifecycleEventsQuery.isError}
                                    error={lifecycleEventsQuery.error}
                                    onRetry={() => void lifecycleEventsQuery.refetch()}
                                >
                                    {(lifecycleEventsQuery.data ?? []).length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No lifecycle events recorded yet.
                                        </Typography>
                                    ) : (
                                        <Stack spacing={1}>
                                            {(lifecycleEventsQuery.data ?? []).map((event) => (
                                                <Box
                                                    key={event.id}
                                                    sx={{
                                                        p: 1.25,
                                                        borderRadius: 1,
                                                        bgcolor: "action.hover",
                                                    }}
                                                >
                                                    <Typography variant="body2">
                                                        {event.from_status
                                                            ? `${lifecycleDisplayLabel(event.from_status)} → `
                                                            : ""}
                                                        {lifecycleDisplayLabel(event.to_status)}
                                                    </Typography>
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                        display="block"
                                                    >
                                                        {new Date(event.created_at).toLocaleString()}
                                                        {event.actor_id
                                                            ? ` · actor ${event.actor_id.slice(0, 8)}…`
                                                            : ""}
                                                        {event.run_id
                                                            ? ` · run ${event.run_id.slice(0, 8)}…`
                                                            : ""}
                                                    </Typography>
                                                    {event.reason ? (
                                                        <Typography variant="caption" display="block">
                                                            Reason: {event.reason}
                                                        </Typography>
                                                    ) : null}
                                                </Box>
                                            ))}
                                        </Stack>
                                    )}
                                </QueryBoundary>
                            </Stack>

                            <Stack spacing={0.5}>
                                <Typography variant="subtitle2">Training snapshot</Typography>
                                <Typography variant="body2" sx={{ fontFamily: "monospace" }}>
                                    {selectedModel.training_dataset_snapshot_id}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Analysis run: {selectedModel.analysis_run_id}
                                </Typography>
                            </Stack>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Feature / selection config
                                </Typography>
                                <ResultsInspector
                                    title="feature_config"
                                    data={{
                                        feature_config: featureConfig,
                                        feature_selection: selectionConfig,
                                        training_config: trainingConfig,
                                    }}
                                />
                            </Box>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Metrics
                                </Typography>
                                <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
                                    <MetricLine
                                        label="Macro F1"
                                        value={formatMetric(extractMacroF1(metrics))}
                                    />
                                    <MetricLine
                                        label="Micro F1"
                                        value={formatMetric(num(metrics?.f1_micro))}
                                    />
                                    <MetricLine
                                        label="Accuracy"
                                        value={formatMetric(num(metrics?.accuracy))}
                                    />
                                    <MetricLine
                                        label="ROC-AUC"
                                        value={formatMetric(num(metrics?.roc_auc))}
                                    />
                                </Stack>
                                <ResultsInspector title="metrics" data={metrics ?? {}} />
                            </Box>

                            {cis ? (
                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Confidence intervals
                                    </Typography>
                                    <ResultsInspector title="confidence intervals" data={cis} />
                                </Box>
                            ) : null}

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Calibration
                                </Typography>
                                {calibration ? (
                                    <Suspense fallback={<EvalPanelFallback />}>
                                        <ClassificationCalibrationPanel
                                            metrics={metrics}
                                            results={null}
                                        />
                                    </Suspense>
                                ) : (
                                    <Typography variant="body2" color="text.secondary">
                                        No calibration payload on this model.
                                    </Typography>
                                )}
                            </Box>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Coefficients
                                </Typography>
                                <QueryBoundary
                                    isLoading={coefficientsQuery.isLoading}
                                    isError={coefficientsQuery.isError}
                                    error={coefficientsQuery.error}
                                    onRetry={() => void coefficientsQuery.refetch()}
                                >
                                    {(coefficientsQuery.data ?? []).length ? (
                                        <ResearchResultsTable
                                            rows={(coefficientsQuery.data ?? []).slice(0, 40).map(
                                                (row, index) => ({
                                                    id: `${row.feature}-${row.label}-${index}`,
                                                    feature: row.feature,
                                                    label: row.label,
                                                    coefficient: row.coefficient,
                                                    direction: row.direction,
                                                    rank: row.rank,
                                                })
                                            )}
                                            columns={[
                                                {
                                                    id: "feature",
                                                    label: "Feature",
                                                    value: (row) => row.feature,
                                                },
                                                {
                                                    id: "label",
                                                    label: "Label",
                                                    value: (row) => row.label,
                                                },
                                                {
                                                    id: "coefficient",
                                                    label: "Coef",
                                                    value: (row) => row.coefficient,
                                                    align: "right",
                                                },
                                                {
                                                    id: "direction",
                                                    label: "Dir",
                                                    value: (row) => row.direction,
                                                },
                                                {
                                                    id: "rank",
                                                    label: "Rank",
                                                    value: (row) => row.rank,
                                                    align: "right",
                                                },
                                            ]}
                                        />
                                    ) : (
                                        <Typography variant="body2" color="text.secondary">
                                            No coefficients available for this model family.
                                        </Typography>
                                    )}
                                </QueryBoundary>
                            </Box>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Provenance
                                </Typography>
                                <QueryBoundary
                                    isLoading={provenanceQuery.isLoading}
                                    isError={provenanceQuery.isError}
                                    error={provenanceQuery.error}
                                    onRetry={() => void provenanceQuery.refetch()}
                                >
                                    {provenanceQuery.data ? (
                                        <ResultsInspector
                                            title="run provenance"
                                            data={provenanceQuery.data}
                                        />
                                    ) : (
                                        <Typography variant="body2" color="text.secondary">
                                            Provenance unavailable for training run.
                                        </Typography>
                                    )}
                                </QueryBoundary>
                            </Box>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Prediction sets
                                </Typography>
                                <QueryBoundary
                                    isLoading={predictionSetsQuery.isLoading}
                                    isError={predictionSetsQuery.isError}
                                    error={predictionSetsQuery.error}
                                    onRetry={() => void predictionSetsQuery.refetch()}
                                >
                                    {!modelPredictionSets.length ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No prediction sets for this model yet.
                                        </Typography>
                                    ) : (
                                        <Box sx={{ overflowX: "auto" }}>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Set</TableCell>
                                                        <TableCell>Created</TableCell>
                                                        <TableCell>Units</TableCell>
                                                        <TableCell align="right">Open</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {modelPredictionSets.map((ps) => (
                                                        <TableRow key={ps.id}>
                                                            <TableCell>
                                                                {ps.id.slice(0, 8)}…
                                                            </TableCell>
                                                            <TableCell>
                                                                {new Date(
                                                                    ps.created_at
                                                                ).toLocaleString()}
                                                            </TableCell>
                                                            <TableCell>
                                                                {formatMetric(
                                                                    num(ps.metadata?.unit_count),
                                                                    0
                                                                )}
                                                            </TableCell>
                                                            <TableCell align="right">
                                                                <Button
                                                                    size="small"
                                                                    onClick={() =>
                                                                        navigate(
                                                                            `/research/${ctx.projectId}/predictions?set=${encodeURIComponent(ps.id)}`
                                                                        )
                                                                    }
                                                                >
                                                                    Open
                                                                </Button>
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
                                        </Box>
                                    )}
                                </QueryBoundary>
                            </Box>

                            {clonedConfig ? (
                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Cloned config
                                    </Typography>
                                    <ResultsInspector title="cloned classifier config" data={clonedConfig} />
                                    <Button
                                        size="small"
                                        sx={{ mt: 1 }}
                                        onClick={() =>
                                            navigate(
                                                `/research/${ctx.projectId}/classification?tab=train`
                                            )
                                        }
                                    >
                                        Continue in Classification
                                    </Button>
                                </Box>
                            ) : null}
                        </Stack>
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}

function MetricLine({ label, value }: { label: string; value: string }) {
    return (
        <Typography variant="body2">
            <Box component="span" color="text.secondary">
                {label}:{" "}
            </Box>
            {value}
        </Typography>
    );
}
