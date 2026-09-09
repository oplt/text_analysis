import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    Chip,
    FormControlLabel,
    MenuItem,
    Stack,
    Step,
    StepLabel,
    Stepper,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { PageTabs } from "../../../components/ui/PageTabs";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import {
    ModelTraining as ModelIcon,
    PlayArrow as TrainIcon,
    Preview as PreviewIcon,
    RateReview as AnnotateIcon,
    Science as PredictIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import { useAuth } from "../../../hooks/useAuth";
import {
    assignUncertainPredictions,
    freezeDataset,
    getClassifierCoefficients,
    getRun,
    listClassifiers,
    listDatasetSnapshots,
    listPreprocessingProfiles,
    listUncertainPredictions,
    predictClassifier,
    previewDataset,
    trainClassifier,
    type ClassifierCoefficient,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import {
    DivergingBarChart,
    MatrixHeatmap,
    MetricCards,
    RankedBarChart,
    ResultsInspector,
} from "../components/ResearchCharts";
import { ResearchResultsTable } from "../components/ResearchResults";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval } from "../runPolling";
import type { AnalysisRun, TrainedModel } from "../types";

type Algorithm = "logistic_regression" | "linear_svm";
type ClassWeight = "none" | "balanced";
type AnnotationSource = "adjudicated_only" | "majority_vote" | "selected_annotator";

const ACTIVE_LEARNING_STEPS = [
    "Train",
    "Predict",
    "Review uncertain units",
    "Send to annotation",
    "Annotate",
    "Freeze dataset v2",
    "Retrain",
] as const;

const CLASSIFICATION_TABS = ["dataset", "train", "evaluate", "models", "active"] as const;
type ClassificationTab = (typeof CLASSIFICATION_TABS)[number];

const CLASSIFICATION_TAB_ITEMS: Array<{ value: ClassificationTab; label: string }> = [
    { value: "dataset", label: "Dataset" },
    { value: "train", label: "Train" },
    { value: "evaluate", label: "Evaluate" },
    { value: "models", label: "Models" },
    { value: "active", label: "Active learning" },
];

const ANNOTATION_SOURCES: Array<{ value: AnnotationSource; label: string }> = [
    { value: "adjudicated_only", label: "Adjudicated only" },
    { value: "majority_vote", label: "Majority vote" },
    { value: "selected_annotator", label: "Selected annotator" },
];

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function formatMetric(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function num(value: unknown): number | null {
    if (typeof value === "number" && !Number.isNaN(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
}

function pickMetrics(...sources: Array<Record<string, unknown> | null | undefined>) {
    for (const source of sources) {
        if (source && Object.keys(source).length > 0) return source;
    }
    return null;
}

function evaluationCards(metrics: Record<string, unknown> | null, results: Record<string, unknown> | null) {
    const cards: Array<{ label: string; value: string | number }> = [];
    if (!metrics && !results) return cards;

    const metricKeys: Array<[string, string]> = [
        ["f1_macro", "Macro F1"],
        ["f1_micro", "Micro F1"],
        ["f1_weighted", "Weighted F1"],
        ["precision_macro", "Precision (macro)"],
        ["recall_macro", "Recall (macro)"],
        ["precision_micro", "Precision (micro)"],
        ["recall_micro", "Recall (micro)"],
        ["precision_weighted", "Precision (weighted)"],
        ["recall_weighted", "Recall (weighted)"],
        ["accuracy", "Accuracy"],
    ];
    for (const [key, label] of metricKeys) {
        const value = num(metrics?.[key]);
        if (value != null) cards.push({ label, value: formatMetric(value) });
    }

    const nTrain = num(results?.n_train) ?? num(metrics?.n_train) ?? num(metrics?.train_size);
    const nTest = num(results?.n_test) ?? num(metrics?.n_test) ?? num(metrics?.test_size);
    const vocab =
        num(results?.vocabulary_size) ?? num(metrics?.vocabulary_size) ?? num(metrics?.vocab_size);
    if (nTrain != null) cards.push({ label: "Train size", value: nTrain });
    if (nTest != null) cards.push({ label: "Test size", value: nTest });
    if (vocab != null) cards.push({ label: "Vocabulary size", value: vocab });
    return cards;
}

function perLabelF1Items(metrics: Record<string, unknown> | null): Array<{ label: string; value: number }> {
    const perClass = asRecord(metrics?.per_class) ?? asRecord(metrics?.per_label);
    if (!perClass) return [];
    return Object.entries(perClass)
        .map(([label, raw]) => {
            const row = asRecord(raw);
            const f1 = num(row?.f1) ?? num(row?.["f1-score"]);
            return f1 == null ? null : { label, value: f1 };
        })
        .filter((item): item is { label: string; value: number } => item != null)
        .sort((a, b) => b.value - a.value);
}

function classDistributionItems(
    distribution: Record<string, Record<string, number>>,
    labelNames?: Map<string, string>
): Array<{ label: string; value: number }> {
    const items: Array<{ label: string; value: number }> = [];
    for (const [labelKey, counts] of Object.entries(distribution)) {
        const display = labelNames?.get(labelKey) ?? labelKey;
        const yes = counts.yes ?? counts["1"] ?? counts.positive;
        if (typeof yes === "number") {
            items.push({ label: `${display} (yes)`, value: yes });
            continue;
        }
        for (const [value, count] of Object.entries(counts)) {
            items.push({ label: `${display}: ${value}`, value: Number(count) || 0 });
        }
    }
    return items.sort((a, b) => b.value - a.value);
}

function perLabelMetrics(
    metrics: Record<string, unknown> | null
): Array<{ label: string; precision: number | null; recall: number | null; f1: number | null; support: number | null }> {
    const perClass = asRecord(metrics?.per_class) ?? asRecord(metrics?.per_label);
    if (!perClass) return [];
    return Object.entries(perClass).map(([label, raw]) => {
        const row = asRecord(raw);
        return {
            label,
            precision: num(row?.precision),
            recall: num(row?.recall),
            f1: num(row?.f1) ?? num(row?.["f1-score"]),
            support: num(row?.support),
        };
    });
}

function coefficientItemsForLabel(
    coefficients: ClassifierCoefficient[],
    label: string,
    topN = 12
): Array<{ label: string; value: number }> {
    const forLabel = coefficients.filter((c) => c.label === label);
    const positives = [...forLabel]
        .filter((c) => c.coefficient > 0)
        .sort((a, b) => b.coefficient - a.coefficient)
        .slice(0, topN);
    const negatives = [...forLabel]
        .filter((c) => c.coefficient < 0)
        .sort((a, b) => a.coefficient - b.coefficient)
        .slice(0, topN)
        .reverse();
    return [...positives, ...negatives].map((c) => ({
        label: `${c.direction === "negative" ? "−" : "+"}${c.feature}`,
        value: c.coefficient,
    }));
}

function activeLearningStepIndex(opts: {
    hasModel: boolean;
    hasPredictRun: boolean;
    uncertainCount: number;
    assignedOnce: boolean;
}): number {
    if (!opts.hasModel) return 0;
    if (!opts.hasPredictRun) return 1;
    if (opts.uncertainCount === 0 && !opts.assignedOnce) return 2;
    if (!opts.assignedOnce) return 3;
    return 4;
}

export default function ClassificationView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const { currentUser } = useAuth();

    const [annotationSource, setAnnotationSource] = useState<AnnotationSource>("adjudicated_only");
    const [minimumAgreement, setMinimumAgreement] = useState(0.66);
    const [selectedAnnotatorId, setSelectedAnnotatorId] = useState("");
    const [snapshotName, setSnapshotName] = useState("Training snapshot");
    const [snapshotId, setSnapshotId] = useState<string | null>(null);

    const [algorithm, setAlgorithm] = useState<Algorithm>("logistic_regression");
    const [profileId, setProfileId] = useState("");
    const [ngramMax, setNgramMax] = useState(2);
    const [minDf, setMinDf] = useState(1);
    const [maxDf, setMaxDf] = useState(1);
    const [maxFeatures, setMaxFeatures] = useState<string>("");
    const [classWeight, setClassWeight] = useState<ClassWeight>("balanced");
    const [regularizationC, setRegularizationC] = useState(1);
    const [testSize, setTestSize] = useState(0.25);
    const [randomSeed, setRandomSeed] = useState(42);
    const [modelName, setModelName] = useState("");

    const [trainRunId, setTrainRunId] = useState<string | null>(null);
    const [predictRunId, setPredictRunId] = useState<string | null>(null);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const [selectedUncertainIds, setSelectedUncertainIds] = useState<string[]>([]);
    const [assignedOnce, setAssignedOnce] = useState(false);
    const [coefficientLabel, setCoefficientLabel] = useState("");
    const [confusionNormalized, setConfusionNormalized] = useState(false);
    const [confusionLabel, setConfusionLabel] = useState("");
    const [tab, setTab] = useTabQueryParam(CLASSIFICATION_TABS, "dataset");

    const labelIds = ctx.labels.map((l) => l.id);
    const labelNameById = useMemo(
        () => new Map(ctx.labels.map((l) => [l.id, l.name])),
        [ctx.labels]
    );

    const canPreview = Boolean(ctx.selectedCorpusId && ctx.selectedCodebookId && labelIds.length > 0);

    const previewQuery = useQuery({
        queryKey: [
            "text-research",
            "dataset-preview",
            ctx.selectedCorpusId,
            ctx.selectedCodebookId,
            ctx.unitType,
            labelIds.join(","),
            annotationSource,
            minimumAgreement,
            selectedAnnotatorId,
        ],
        queryFn: () =>
            previewDataset({
                corpus_id: ctx.selectedCorpusId,
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: labelIds,
                annotation_source: annotationSource,
                minimum_agreement:
                    annotationSource === "majority_vote" ? minimumAgreement : undefined,
                selected_annotator_id:
                    annotationSource === "selected_annotator"
                        ? selectedAnnotatorId || undefined
                        : undefined,
            }),
        enabled:
            canPreview &&
            (annotationSource !== "selected_annotator" || Boolean(selectedAnnotatorId)),
    });

    const snapshotsQuery = useQuery({
        queryKey: queryKeys.textResearch.datasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listDatasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId),
    });

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listClassifiers(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId),
    });

    const trainRunQuery = useQuery({
        queryKey: queryKeys.textResearch.run(trainRunId ?? ""),
        queryFn: () => getRun(trainRunId!),
        enabled: Boolean(trainRunId),
        refetchInterval: activeRunRefetchInterval,
    });

    const predictRunQuery = useQuery({
        queryKey: queryKeys.textResearch.run(predictRunId ?? ""),
        queryFn: () => getRun(predictRunId!),
        enabled: Boolean(predictRunId),
        refetchInterval: activeRunRefetchInterval,
    });

    const uncertainQuery = useQuery({
        queryKey: ["text-research", "uncertain-predictions", selectedModelId],
        queryFn: () => listUncertainPredictions(selectedModelId!),
        enabled: Boolean(selectedModelId),
    });

    const coefficientsQuery = useQuery({
        queryKey: queryKeys.textResearch.coefficients(selectedModelId ?? ""),
        queryFn: () => getClassifierCoefficients(selectedModelId!),
        enabled: Boolean(selectedModelId),
    });

    const selectedModel: TrainedModel | undefined = classifiersQuery.data?.find(
        (m) => m.id === selectedModelId
    );

    const trainMetrics = pickMetrics(
        asRecord(trainRunQuery.data?.metrics),
        asRecord(selectedModel?.metrics)
    );
    const trainResults = asRecord(trainRunQuery.data?.results);
    const evalCards = evaluationCards(trainMetrics, trainResults);
    const perLabelItems = perLabelF1Items(trainMetrics);
    const perLabelMetricRows = perLabelMetrics(trainMetrics);
    const trainGroups = Array.isArray(trainResults?.train_groups)
        ? trainResults.train_groups
        : null;
    const testGroups = Array.isArray(trainResults?.test_groups) ? trainResults.test_groups : null;
    const multiclassConfusion = Array.isArray(trainMetrics?.confusion_matrix)
        ? (trainMetrics?.confusion_matrix as number[][])
        : null;
    const multiclassConfusionLabels = Array.isArray(trainMetrics?.confusion_matrix_labels)
        ? trainMetrics.confusion_matrix_labels.map(String)
        : [];
    const multilabelConfusions = asRecord(trainMetrics?.multilabel_confusion_matrices);
    const availableConfusionLabels = Object.keys(multilabelConfusions ?? {});
    const activeConfusionLabel = confusionLabel || availableConfusionLabels[0] || "";
    const multilabelConfusion = Array.isArray(multilabelConfusions?.[activeConfusionLabel])
        ? (multilabelConfusions?.[activeConfusionLabel] as number[][])
        : null;

    const distributionItems = previewQuery.data
        ? classDistributionItems(previewQuery.data.class_distribution)
        : [];

    const coefficientLabels = useMemo(() => {
        const labels = new Set((coefficientsQuery.data ?? []).map((c) => c.label));
        return Array.from(labels).sort();
    }, [coefficientsQuery.data]);

    const activeCoefficientLabel =
        coefficientLabel && coefficientLabels.includes(coefficientLabel)
            ? coefficientLabel
            : coefficientLabels[0] ?? "";

    const divergingItems = coefficientItemsForLabel(
        coefficientsQuery.data ?? [],
        activeCoefficientLabel
    );

    const freezeMutation = useMutation({
        mutationFn: () =>
            freezeDataset({
                name: snapshotName.trim() || "Training snapshot",
                corpus_id: ctx.selectedCorpusId,
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: labelIds,
                annotation_source: annotationSource,
                minimum_agreement:
                    annotationSource === "majority_vote" ? minimumAgreement : undefined,
                selected_annotator_id:
                    annotationSource === "selected_annotator"
                        ? selectedAnnotatorId || undefined
                        : undefined,
            }),
        onSuccess: (snapshot) => {
            setSnapshotId(snapshot.id);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.datasetSnapshots(
                    ctx.projectId,
                    ctx.selectedCorpusId
                ),
            });
            showToast({ message: "Dataset snapshot frozen.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to freeze dataset."),
                severity: "error",
            }),
    });

    const trainMutation = useMutation({
        mutationFn: () => {
            const parsedMaxFeatures = maxFeatures.trim()
                ? Number.parseInt(maxFeatures.trim(), 10)
                : null;
            return trainClassifier({
                snapshot_id: snapshotId!,
                name:
                    modelName.trim() ||
                    `Classifier ${new Date().toLocaleDateString()} (${algorithm})`,
                algorithm,
                preprocessing_profile_id: profileId || undefined,
                ngram_max: ngramMax,
                min_df: minDf,
                max_df: maxDf,
                max_features:
                    parsedMaxFeatures != null && !Number.isNaN(parsedMaxFeatures)
                        ? parsedMaxFeatures
                        : null,
                class_weight: classWeight === "none" ? null : classWeight,
                regularization_c: regularizationC,
                test_size: testSize,
                random_seed: randomSeed,
                run_async: true,
            });
        },
        onSuccess: (run: AnalysisRun) => {
            setTrainRunId(run.id);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
            });
            showToast({ message: "Classifier training started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to train classifier."),
                severity: "error",
            }),
    });

    const predictMutation = useMutation({
        mutationFn: () =>
            predictClassifier(selectedModelId!, {
                unit_type: ctx.unitType,
                only_unannotated: true,
            }),
        onSuccess: (run) => {
            setPredictRunId(run.id);
            void uncertainQuery.refetch();
            showToast({
                message: "Model predictions saved. Review uncertain cases below.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Prediction failed."),
                severity: "error",
            }),
    });

    const assignMutation = useMutation({
        mutationFn: () =>
            assignUncertainPredictions(selectedModelId!, selectedUncertainIds, [currentUser!.id]),
        onSuccess: () => {
            setSelectedUncertainIds([]);
            setAssignedOnce(true);
            showToast({
                message: "Selected cases were added to your annotation queue.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Could not assign cases."),
                severity: "error",
            }),
    });

    useEffect(() => {
        if (trainRunQuery.data?.status !== "completed") return;
        void client.invalidateQueries({
            queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        });
        const trainedModelIdFromRun =
            typeof trainResults?.trained_model_id === "string"
                ? trainResults.trained_model_id
                : null;
        if (
            trainedModelIdFromRun &&
            classifiersQuery.data?.some((m) => m.id === trainedModelIdFromRun)
        ) {
            const selectModel = window.setTimeout(
                () => setSelectedModelId(trainedModelIdFromRun),
                0
            );
            return () => window.clearTimeout(selectModel);
        }
    }, [
        client,
        ctx.projectId,
        ctx.selectedCorpusId,
        trainRunQuery.data?.status,
        trainResults?.trained_model_id,
        classifiersQuery.data,
    ]);

    useEffect(() => {
        if (predictRunQuery.data?.status !== "completed" || !selectedModelId) return;
        void client.invalidateQueries({
            queryKey: ["text-research", "uncertain-predictions", selectedModelId],
        });
    }, [client, predictRunQuery.data?.status, selectedModelId]);

    const workflowStep = activeLearningStepIndex({
        hasModel: Boolean(selectedModelId || classifiersQuery.data?.length),
        hasPredictRun: Boolean(predictRunId || (uncertainQuery.data?.length ?? 0) > 0),
        uncertainCount: uncertainQuery.data?.length ?? 0,
        assignedOnce,
    });

    const previewCoverage = previewQuery.data?.annotator_coverage ?? [];

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={CLASSIFICATION_TAB_ITEMS}
                ariaLabel="Classification workflow"
            />

            {tab === "dataset" ? (
            <>
            <SectionCard
                title="Dataset preview"
                description="Inspect labeled units before freezing a training snapshot."
                action={
                    <Button
                        variant="outlined"
                        startIcon={<PreviewIcon />}
                        onClick={() => void previewQuery.refetch()}
                        disabled={!canPreview}
                    >
                        Refresh preview
                    </Button>
                }
            >
                {!canPreview ? (
                    <EmptyState
                        icon={<PreviewIcon fontSize="large" />}
                        title="Training dataset not ready"
                        description={
                            !ctx.selectedCorpusId
                                ? "Select a corpus, then ensure a codebook with labels exists so labeled units can be previewed."
                                : !ctx.selectedCodebookId
                                  ? "Create or select a codebook with labels before preparing a training dataset."
                                  : "Add at least one codebook label, then annotate units so the dataset preview has classes to show."
                        }
                        action={
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                <Button
                                    variant="contained"
                                    onClick={() =>
                                        navigate(
                                            !ctx.selectedCodebookId
                                                ? `/research/${ctx.projectId}/codebook`
                                                : `/research/${ctx.projectId}/annotation`
                                        )
                                    }
                                >
                                    {!ctx.selectedCodebookId ? "Create codebook" : "Go to annotation"}
                                </Button>
                                <Button
                                    variant="outlined"
                                    onClick={() => navigate(`/research/${ctx.projectId}/corpus`)}
                                >
                                    Check corpus
                                </Button>
                            </Stack>
                        }
                    />
                ) : (
                    <Stack spacing={2}>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                            <TextField
                                select
                                size="small"
                                label="Annotation source"
                                value={annotationSource}
                                onChange={(e) =>
                                    setAnnotationSource(e.target.value as AnnotationSource)
                                }
                                sx={{ minWidth: 220 }}
                            >
                                {ANNOTATION_SOURCES.map((opt) => (
                                    <MenuItem key={opt.value} value={opt.value}>
                                        {opt.label}
                                    </MenuItem>
                                ))}
                            </TextField>
                            {annotationSource === "majority_vote" ? (
                                <TextField
                                    size="small"
                                    type="number"
                                    label="Minimum agreement"
                                    value={minimumAgreement}
                                    onChange={(e) =>
                                        setMinimumAgreement(Number(e.target.value) || 0)
                                    }
                                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                                    sx={{ width: 160 }}
                                />
                            ) : null}
                            {annotationSource === "selected_annotator" ? (
                                <TextField
                                    size="small"
                                    label="Annotator ID"
                                    value={selectedAnnotatorId}
                                    onChange={(e) => setSelectedAnnotatorId(e.target.value)}
                                    sx={{ minWidth: 220 }}
                                    helperText="Required when using selected annotator"
                                />
                            ) : null}
                        </Stack>

                        <QueryBoundary
                            isLoading={previewQuery.isLoading}
                            isError={previewQuery.isError}
                            error={previewQuery.error}
                            onRetry={() => void previewQuery.refetch()}
                        >
                            {previewQuery.data ? (
                                <Stack spacing={2}>
                                    <Typography variant="body2" color="text.secondary">
                                        Codebook {ctx.selectedCodebook?.name} · v
                                        {ctx.selectedCodebook?.version}
                                        {ctx.selectedCodebook?.is_frozen ? " · frozen" : ""}
                                    </Typography>

                                    <MetricCards
                                        items={[
                                            {
                                                label: "Fully labeled units",
                                                value: previewQuery.data.unit_count,
                                            },
                                            {
                                                label: "Documents",
                                                value: previewQuery.data.document_count,
                                            },
                                            {
                                                label: "Missing labels",
                                                value: previewQuery.data.missing_labels.length,
                                            },
                                            {
                                                label: "Excluded disagreements",
                                                value: previewQuery.data.excluded_disagreements
                                                    .length,
                                            },
                                        ]}
                                    />

                                    <Box>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Class distribution
                                        </Typography>
                                        {distributionItems.length ? (
                                            <RankedBarChart items={distributionItems} height={280} />
                                        ) : (
                                            <Typography variant="body2" color="text.secondary">
                                                No class counts in this preview.
                                            </Typography>
                                        )}
                                    </Box>

                                    <Typography variant="body2" color="text.secondary">
                                        Annotator coverage:{" "}
                                        {previewCoverage.length
                                            ? previewCoverage.join(", ")
                                            : "none yet"}
                                    </Typography>

                                    {previewQuery.data.warnings.map((warning) => (
                                        <Alert key={warning} severity="warning">
                                            {warning}
                                        </Alert>
                                    ))}

                                    <ResultsInspector
                                        title="dataset preview"
                                        data={previewQuery.data}
                                    />
                                </Stack>
                            ) : null}
                        </QueryBoundary>
                    </Stack>
                )}
            </SectionCard>

            <SectionCard
                title="Freeze training snapshot"
                description="A frozen TrainingDatasetSnapshot is required before training. Retrain later by freezing a v2 snapshot after new annotations."
            >
                <Stack spacing={2}>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems="flex-start">
                        <TextField
                            size="small"
                            label="Snapshot name"
                            value={snapshotName}
                            onChange={(e) => setSnapshotName(e.target.value)}
                            sx={{ minWidth: 220 }}
                        />
                        <Button
                            variant="contained"
                            onClick={() => freezeMutation.mutate()}
                            disabled={!canPreview || freezeMutation.isPending}
                        >
                            Freeze snapshot
                        </Button>
                    </Stack>

                    <QueryBoundary
                        isLoading={snapshotsQuery.isLoading}
                        isError={snapshotsQuery.isError}
                        error={snapshotsQuery.error}
                        onRetry={() => void snapshotsQuery.refetch()}
                    >
                        {snapshotsQuery.data?.length ? (
                            <Box sx={{ overflowX: "auto" }}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Name</TableCell>
                                            <TableCell>Codebook version</TableCell>
                                            <TableCell>Source</TableCell>
                                            <TableCell>Created</TableCell>
                                            <TableCell align="right">Use</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {snapshotsQuery.data.map((snap) => (
                                            <TableRow
                                                key={snap.id}
                                                selected={snapshotId === snap.id}
                                            >
                                                <TableCell>{snap.name}</TableCell>
                                                <TableCell>{snap.codebook_version}</TableCell>
                                                <TableCell>{snap.annotation_source}</TableCell>
                                                <TableCell>
                                                    {new Date(snap.created_at).toLocaleString()}
                                                </TableCell>
                                                <TableCell align="right">
                                                    <Button
                                                        size="small"
                                                        variant={
                                                            snapshotId === snap.id
                                                                ? "contained"
                                                                : "outlined"
                                                        }
                                                        onClick={() => setSnapshotId(snap.id)}
                                                    >
                                                        {snapshotId === snap.id
                                                            ? "Selected"
                                                            : "Select"}
                                                    </Button>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No snapshots yet. Freeze the previewed dataset to unlock training.
                            </Typography>
                        )}
                    </QueryBoundary>

                    {snapshotId ? (
                        <Alert severity="success">
                            Active snapshot selected. Training will use this exact frozen unit set.
                        </Alert>
                    ) : (
                        <Alert severity="info">
                            Freeze or select a snapshot before configuring and training a model.
                        </Alert>
                    )}
                </Stack>
            </SectionCard>
            </>
            ) : null}

            {tab === "train" ? (
            <SectionCard
                title="Model configuration"
                description="Configure a linear classifier. Train/test splitting is grouped by source document to prevent leakage."
            >
                <Stack spacing={2}>
                    <Alert severity="warning">
                        Train/test splitting is grouped by source document to prevent leakage.
                    </Alert>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                        <TextField
                            select
                            size="small"
                            label="Algorithm"
                            value={algorithm}
                            onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
                            sx={{ minWidth: 200 }}
                        >
                            <MenuItem value="logistic_regression">Logistic regression</MenuItem>
                            <MenuItem value="linear_svm">Linear SVM</MenuItem>
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Preprocessing profile"
                            value={profileId}
                            onChange={(e) => setProfileId(e.target.value)}
                            sx={{ minWidth: 220 }}
                        >
                            <MenuItem value="">Default (none)</MenuItem>
                            {(profilesQuery.data ?? []).map((profile) => (
                                <MenuItem key={profile.id} value={profile.id}>
                                    {profile.name}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            size="small"
                            label="Model name"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            sx={{ minWidth: 200 }}
                            placeholder="Optional"
                        />
                    </Stack>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                        <TextField
                            size="small"
                            type="number"
                            label="ngram_max"
                            value={ngramMax}
                            onChange={(e) => setNgramMax(Number(e.target.value) || 1)}
                            inputProps={{ min: 1, max: 5, step: 1 }}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="min_df"
                            value={minDf}
                            onChange={(e) => setMinDf(Number(e.target.value) || 1)}
                            inputProps={{ min: 1, step: 1 }}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="max_df"
                            value={maxDf}
                            onChange={(e) => setMaxDf(Number(e.target.value) || 1)}
                            inputProps={{ min: 0.01, max: 1, step: 0.01 }}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            label="max_features"
                            value={maxFeatures}
                            onChange={(e) => setMaxFeatures(e.target.value)}
                            placeholder="unlimited"
                            sx={{ width: 140 }}
                            helperText="Blank = no cap"
                        />
                        <TextField
                            select
                            size="small"
                            label="class_weight"
                            value={classWeight}
                            onChange={(e) => setClassWeight(e.target.value as ClassWeight)}
                            sx={{ width: 150 }}
                        >
                            <MenuItem value="none">none</MenuItem>
                            <MenuItem value="balanced">balanced</MenuItem>
                        </TextField>
                        <TextField
                            size="small"
                            type="number"
                            label="C (regularization)"
                            value={regularizationC}
                            onChange={(e) => setRegularizationC(Number(e.target.value) || 1)}
                            inputProps={{ min: 0.001, step: 0.1 }}
                            sx={{ width: 150 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="test_size"
                            value={testSize}
                            onChange={(e) => setTestSize(Number(e.target.value) || 0.25)}
                            inputProps={{ min: 0.05, max: 0.5, step: 0.05 }}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="random_seed"
                            value={randomSeed}
                            onChange={(e) => setRandomSeed(Number(e.target.value) || 0)}
                            inputProps={{ step: 1 }}
                            sx={{ width: 130 }}
                        />
                    </Stack>

                    <Button
                        variant="contained"
                        startIcon={<TrainIcon />}
                        onClick={() => trainMutation.mutate()}
                        disabled={!snapshotId || trainMutation.isPending}
                        sx={{ alignSelf: "flex-start" }}
                    >
                        Train classifier
                    </Button>

                    {trainRunId && trainRunQuery.data ? (
                        <Box>
                            <Typography variant="body2" sx={{ mb: 1 }}>
                                Training run —{" "}
                                <RunStatusChip status={trainRunQuery.data.status} />
                            </Typography>
                            {trainRunQuery.data.error_message ? (
                                <Alert severity="error">{trainRunQuery.data.error_message}</Alert>
                            ) : null}
                        </Box>
                    ) : null}
                </Stack>
            </SectionCard>
            ) : null}

            {tab === "evaluate" ? (
            <SectionCard
                title="Evaluation"
                description="Metrics from the latest training run or the selected model. Splits are document-grouped."
            >
                {trainRunQuery.data?.status === "completed" || selectedModel ? (
                    <Stack spacing={2}>
                        {evalCards.length ? <MetricCards items={evalCards} /> : null}

                        {perLabelItems.length ? (
                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Per-label F1
                                </Typography>
                                <RankedBarChart
                                    items={perLabelItems}
                                    height={280}
                                    valueFormatter={(v) => formatMetric(v)}
                                />
                            </Box>
                        ) : null}

                        {multiclassConfusion || multilabelConfusion ? (
                            <Stack spacing={1}>
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                                    <Typography variant="subtitle2">Confusion matrix</Typography>
                                    <FormControlLabel
                                        control={<Checkbox checked={confusionNormalized} onChange={(event) => setConfusionNormalized(event.target.checked)} />}
                                        label="Row-normalized percentages"
                                    />
                                    {availableConfusionLabels.length ? (
                                        <TextField select size="small" label="Multilabel class" value={activeConfusionLabel} onChange={(event) => setConfusionLabel(event.target.value)} sx={{ minWidth: 180 }}>
                                            {availableConfusionLabels.map((label) => <MenuItem key={label} value={label}>{labelNameById.get(label) ?? label}</MenuItem>)}
                                        </TextField>
                                    ) : null}
                                </Stack>
                                {(() => {
                                    const matrix = multiclassConfusion ?? multilabelConfusion ?? [];
                                    const labels = multiclassConfusion ? multiclassConfusionLabels : ["Actual no", "Actual yes"];
                                    const columns = multiclassConfusion ? multiclassConfusionLabels : ["Predicted no", "Predicted yes"];
                                    const values = matrix.map((row) => {
                                        const total = row.reduce((sum, value) => sum + value, 0);
                                        return row.map((value) => confusionNormalized && total ? value / total : value);
                                    });
                                    return <MatrixHeatmap rowLabels={labels} colLabels={columns} values={values} formatCell={(value) => value == null ? "—" : confusionNormalized ? `${(value * 100).toFixed(1)}%` : String(value)} />;
                                })()}
                            </Stack>
                        ) : null}

                        {(trainGroups || testGroups) && (
                            <Typography variant="body2" color="text.secondary">
                                Grouped split:{" "}
                                {trainGroups?.length ?? "—"}{" "}
                                train documents ·{" "}
                                {testGroups?.length ?? "—"}{" "}
                                test documents
                            </Typography>
                        )}

                        {perLabelMetricRows.length ? (
                            <Box sx={{ overflowX: "auto" }}>
                                <Typography variant="subtitle2" gutterBottom>
                                    Per-label metrics
                                </Typography>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Label</TableCell>
                                            <TableCell align="right">Precision</TableCell>
                                            <TableCell align="right">Recall</TableCell>
                                            <TableCell align="right">F1</TableCell>
                                            <TableCell align="right">Support</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {perLabelMetricRows.map((row) => (
                                            <TableRow key={row.label}>
                                                <TableCell>
                                                    {labelNameById.get(row.label) ?? row.label}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(row.precision)}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(row.recall)}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(row.f1)}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(row.support, 0)}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>
                        ) : null}

                        <ResultsInspector
                            title="run metrics / results"
                            data={{
                                metrics: trainMetrics,
                                results: trainResults,
                                model_metrics: selectedModel?.metrics,
                            }}
                        />
                    </Stack>
                ) : (
                    <Typography variant="body2" color="text.secondary">
                        Train a classifier (or select a saved model) to see evaluation metrics.
                    </Typography>
                )}
            </SectionCard>
            ) : null}

            {tab === "models" ? (
            <>
            <SectionCard title="Trained models" description="Classifiers saved for this project/corpus.">
                <QueryBoundary
                    isLoading={classifiersQuery.isLoading}
                    isError={classifiersQuery.isError}
                    error={classifiersQuery.error}
                    onRetry={() => void classifiersQuery.refetch()}
                >
                    {classifiersQuery.data?.length ? (
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Algorithm</TableCell>
                                        <TableCell>Version</TableCell>
                                        <TableCell>Macro F1</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell align="right">Select</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {classifiersQuery.data.map((model) => {
                                        const macro = num(asRecord(model.metrics)?.f1_macro);
                                        return (
                                            <TableRow
                                                key={model.id}
                                                selected={selectedModelId === model.id}
                                            >
                                                <TableCell>
                                                    {model.name ?? model.id.slice(0, 8)}
                                                </TableCell>
                                                <TableCell>{model.model_family}</TableCell>
                                                <TableCell>{model.version}</TableCell>
                                                <TableCell>{formatMetric(macro)}</TableCell>
                                                <TableCell>
                                                    {new Date(model.created_at).toLocaleDateString()}
                                                </TableCell>
                                                <TableCell align="right">
                                                    <Button
                                                        size="small"
                                                        variant={
                                                            selectedModelId === model.id
                                                                ? "contained"
                                                                : "outlined"
                                                        }
                                                        onClick={() => {
                                                            setSelectedModelId(model.id);
                                                            setCoefficientLabel("");
                                                        }}
                                                    >
                                                        Select
                                                    </Button>
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </Box>
                    ) : (
                        <EmptyState
                            icon={<ModelIcon fontSize="large" />}
                            title="No trained models yet"
                            description="Prerequisites: a corpus with labeled units, a codebook with labels, and a frozen training snapshot. Prepare the dataset above, then train."
                            action={
                                <Button
                                    variant="contained"
                                    startIcon={<TrainIcon />}
                                    onClick={() => {
                                        if (!canPreview) {
                                            navigate(`/research/${ctx.projectId}/annotation`);
                                            return;
                                        }
                                        if (!snapshotId) {
                                            freezeMutation.mutate();
                                            return;
                                        }
                                        trainMutation.mutate();
                                    }}
                                    disabled={freezeMutation.isPending || trainMutation.isPending}
                                >
                                    Prepare training dataset
                                </Button>
                            }
                        />
                    )}
                </QueryBoundary>
            </SectionCard>

            <SectionCard
                title="Coefficients"
                description="Strongest positive and negative features per label for the selected linear model."
            >
                {!selectedModelId ? (
                    <Typography color="text.secondary">Select a trained model above.</Typography>
                ) : (
                    <QueryBoundary
                        isLoading={coefficientsQuery.isLoading}
                        isError={coefficientsQuery.isError}
                        error={coefficientsQuery.error}
                        onRetry={() => void coefficientsQuery.refetch()}
                    >
                        <Stack spacing={2}>
                            {coefficientLabels.length ? (
                                <TextField
                                    select
                                    size="small"
                                    label="Label"
                                    value={activeCoefficientLabel}
                                    onChange={(e) => setCoefficientLabel(e.target.value)}
                                    sx={{ maxWidth: 280 }}
                                >
                                    {coefficientLabels.map((label) => (
                                        <MenuItem key={label} value={label}>
                                            {labelNameById.get(label) ?? label}
                                        </MenuItem>
                                    ))}
                                </TextField>
                            ) : null}
                            <DivergingBarChart items={divergingItems} height={360} />
                            <ResultsInspector
                                title="coefficients"
                                data={coefficientsQuery.data}
                            />
                        </Stack>
                    </QueryBoundary>
                )}
            </SectionCard>
            </>
            ) : null}

            {tab === "active" ? (
            <>
            <SectionCard
                title="Active learning loop"
                description="Train a model, score unannotated units, review uncertain cases, send them for human coding, then freeze and retrain."
                compact
            >
                <Stepper activeStep={workflowStep} alternativeLabel sx={{ mb: 2 }}>
                    {ACTIVE_LEARNING_STEPS.map((label) => (
                        <Step key={label}>
                            <StepLabel>{label}</StepLabel>
                        </Step>
                    ))}
                </Stepper>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} flexWrap="wrap">
                    <Button
                        variant="outlined"
                        startIcon={<AnnotateIcon />}
                        onClick={() => navigate(`/research/${ctx.projectId}/annotation`)}
                    >
                        Open annotation
                    </Button>
                    <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center" }}>
                        Model scores stay separate from human coding. Predictions are labeled as
                        model output until a coder confirms them.
                    </Typography>
                </Stack>
            </SectionCard>

            <SectionCard
                title="Apply model & uncertain cases"
                description="Predictions stay separate from human coding. Send difficult cases to annotation deliberately."
                action={
                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="contained"
                            startIcon={<PredictIcon />}
                            onClick={() => predictMutation.mutate()}
                            disabled={!selectedModelId || predictMutation.isPending}
                        >
                            Predict unannotated units
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={() => navigate(`/research/${ctx.projectId}/annotation`)}
                        >
                            Go to annotation
                        </Button>
                    </Stack>
                }
            >
                {!selectedModelId ? (
                    <Typography color="text.secondary">Select a trained model above.</Typography>
                ) : (
                    <Stack spacing={2}>
                        {predictRunId && predictRunQuery.data ? (
                            <Typography variant="body2">
                                Predict run —{" "}
                                <RunStatusChip status={predictRunQuery.data.status} />
                                {predictRunQuery.data.metrics
                                    ? ` · units predicted: ${String(
                                          asRecord(predictRunQuery.data.metrics)?.units_predicted ??
                                              "—"
                                      )}`
                                    : null}
                            </Typography>
                        ) : null}

                        <QueryBoundary
                            isLoading={uncertainQuery.isLoading}
                            isError={uncertainQuery.isError}
                            error={uncertainQuery.error}
                            onRetry={() => void uncertainQuery.refetch()}
                        >
                            <Stack spacing={1}>
                                <Typography variant="body2" color="text.secondary">
                                    Highest uncertainty first. These are model candidates
                                    for human review — not automatic labels.
                                </Typography>
                                {(uncertainQuery.data ?? []).length === 0 ? (
                                    <Typography variant="body2" color="text.secondary">
                                        No uncertain predictions yet. Run predict on unannotated
                                        units after training.
                                    </Typography>
                                ) : null}
                                {uncertainQuery.data?.map((item) => {
                                    const checked = selectedUncertainIds.includes(
                                        item.text_unit.id
                                    );
                                    const scoreEntries = Object.entries(item.prediction.scores ?? {})
                                        .sort((a, b) => b[1] - a[1])
                                        .slice(0, 4);
                                    return (
                                        <Box
                                            key={item.prediction.id}
                                            sx={{
                                                display: "flex",
                                                gap: 1,
                                                alignItems: "flex-start",
                                                py: 0.5,
                                            }}
                                        >
                                            <Checkbox
                                                checked={checked}
                                                onChange={(_, next) =>
                                                    setSelectedUncertainIds((ids) =>
                                                        next
                                                            ? [...ids, item.text_unit.id]
                                                            : ids.filter(
                                                                  (id) => id !== item.text_unit.id
                                                              )
                                                    )
                                                }
                                            />
                                            <Box sx={{ flex: 1 }}>
                                                <Typography variant="body2">
                                                    {item.text_unit.text}
                                                </Typography>
                                                <Stack
                                                    direction="row"
                                                    spacing={0.75}
                                                    flexWrap="wrap"
                                                    useFlexGap
                                                    sx={{ mt: 0.5 }}
                                                >
                                                    <Chip
                                                        size="small"
                                                        color="info"
                                                        label={`model: ${
                                                            item.prediction.predicted_labels
                                                                .join(", ") || "none"
                                                        }`}
                                                    />
                                                    <Chip
                                                        size="small"
                                                        variant="outlined"
                                                        label={`uncertainty ${
                                                            item.prediction.uncertainty?.toFixed(
                                                                3
                                                            ) ?? "n/a"
                                                        }`}
                                                    />
                                                    {scoreEntries.map(([label, score]) => (
                                                        <Chip
                                                            key={label}
                                                            size="small"
                                                            variant="outlined"
                                                            label={`${labelNameById.get(label) ?? label}: ${formatMetric(score)}`}
                                                        />
                                                    ))}
                                                </Stack>
                                            </Box>
                                        </Box>
                                    );
                                })}
                                <ResearchResultsTable
                                    rows={(uncertainQuery.data ?? []).map((item) => ({
                                        id: item.prediction.id,
                                        text: item.text_unit.text,
                                        labels: item.prediction.predicted_labels.join(", "),
                                        scores: Object.entries(item.prediction.scores ?? {})
                                            .map(([label, score]) => `${labelNameById.get(label) ?? label}: ${formatMetric(score)}`)
                                            .join(", "),
                                        uncertainty: item.prediction.uncertainty,
                                        model: selectedModel?.name ?? selectedModel?.model_family ?? selectedModelId,
                                        provenance: "Model prediction",
                                    }))}
                                    columns={[
                                        { id: "text", label: "Text unit", value: (row) => row.text },
                                        { id: "labels", label: "Predicted label(s)", value: (row) => row.labels },
                                        { id: "scores", label: "Scores", value: (row) => row.scores },
                                        { id: "uncertainty", label: "Uncertainty", value: (row) => row.uncertainty, align: "right" },
                                        { id: "model", label: "Model", value: (row) => row.model },
                                        { id: "provenance", label: "Provenance", value: (row) => row.provenance },
                                    ]}
                                />
                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <Button
                                        variant="contained"
                                        onClick={() => assignMutation.mutate()}
                                        disabled={
                                            !selectedUncertainIds.length ||
                                            !currentUser ||
                                            assignMutation.isPending
                                        }
                                    >
                                        Send selected to my annotation queue
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        onClick={() => {
                                            setSnapshotName(
                                                `Training snapshot v${(snapshotsQuery.data?.length ?? 0) + 2}`
                                            );
                                            setTab("dataset");
                                        }}
                                    >
                                        After annotating: freeze dataset v2 → retrain
                                    </Button>
                                </Stack>
                            </Stack>
                        </QueryBoundary>
                    </Stack>
                )}
            </SectionCard>
            </>
            ) : null}
        </Stack>
    );
}
