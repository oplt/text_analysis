import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    Chip,
    FormControlLabel,
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
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
    getClassifier,
    getPredictionSet,
    listAdjudications,
    listAnnotationsForUnits,
    listModels,
    listPredictionSets,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { ResultsInspector } from "../components/ResearchCharts";
import { useResearchContext } from "../hooks/useResearchContext";
import { formatMetric, modelDisplayName, num } from "../modelRegistryUtils";
import {
    DEFAULT_PREDICTION_SET_FILTERS,
    PREDICTION_LAYER_LABELS,
    buildPredictionRows,
    filterPredictionRows,
    type PredictionSetFilters,
    type ReviewStatus,
} from "../predictionSetFilters";

function scoresSummary(scores: Record<string, number>): string {
    const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return "—";
    return entries
        .slice(0, 3)
        .map(([label, value]) => `${label}:${value.toFixed(2)}`)
        .join(" · ");
}

export default function PredictionSetsView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const selectedSetId = searchParams.get("set") ?? "";

    const [modelFilter, setModelFilter] = useState("");
    const [filters, setFilters] = useState<PredictionSetFilters>(DEFAULT_PREDICTION_SET_FILTERS);

    const setsQuery = useQuery({
        queryKey: queryKeys.textResearch.predictionSets(ctx.selectedCorpusId),
        queryFn: () => listPredictionSets(ctx.selectedCorpusId, { limit: 100 }),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.models(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () =>
            listModels(ctx.projectId, { corpusId: ctx.selectedCorpusId || undefined }),
        enabled: Boolean(ctx.projectId),
    });

    const modelsById = useMemo(() => {
        const map = new Map<string, (typeof modelsQuery.data)[number]>();
        for (const model of modelsQuery.data ?? []) {
            map.set(model.id, model);
        }
        return map;
    }, [modelsQuery.data]);

    const sets = useMemo(() => {
        const all = setsQuery.data ?? [];
        if (!modelFilter) return all;
        return all.filter((item) => item.trained_model_id === modelFilter);
    }, [setsQuery.data, modelFilter]);

    const detailQuery = useQuery({
        queryKey: queryKeys.textResearch.predictionSet(selectedSetId),
        queryFn: () => getPredictionSet(selectedSetId),
        enabled: Boolean(selectedSetId),
    });

    const detail = detailQuery.data ?? null;
    const unitIds = useMemo(
        () => [...new Set((detail?.predictions ?? []).map((row) => row.text_unit_id))],
        [detail?.predictions]
    );

    const annotationsQuery = useQuery({
        queryKey: [
            "text-research",
            "prediction-set-annotations",
            detail?.corpus_id,
            selectedSetId,
            unitIds.length,
        ],
        queryFn: () => listAnnotationsForUnits(detail!.corpus_id, unitIds),
        enabled: Boolean(detail?.corpus_id && unitIds.length),
    });

    const adjudicationsQuery = useQuery({
        queryKey: queryKeys.textResearch.adjudications(detail?.corpus_id ?? ""),
        queryFn: () => listAdjudications(detail!.corpus_id),
        enabled: Boolean(detail?.corpus_id),
    });

    const sourceModelQuery = useQuery({
        queryKey: queryKeys.textResearch.model(detail?.trained_model_id ?? ""),
        queryFn: () => getClassifier(detail!.trained_model_id),
        enabled: Boolean(detail?.trained_model_id),
    });

    const rows = useMemo(
        () =>
            buildPredictionRows({
                predictions: detail?.predictions ?? [],
                annotations: annotationsQuery.data ?? [],
                adjudications: (adjudicationsQuery.data ?? []).filter((row) =>
                    unitIds.includes(row.text_unit_id)
                ),
            }),
        [detail?.predictions, annotationsQuery.data, adjudicationsQuery.data, unitIds]
    );

    const filteredRows = useMemo(
        () => filterPredictionRows(rows, filters),
        [rows, filters]
    );

    const openSet = (id: string) => {
        setSearchParams(id ? { set: id } : {});
        setFilters(DEFAULT_PREDICTION_SET_FILTERS);
    };

    if (!ctx.projectId) {
        return <Alert severity="info">Select a research project.</Alert>;
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Prediction sets"
                description="Model outputs stay separate from human codes and adjudicated gold. Predictions are never converted to gold."
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
                    Layers: <strong>{PREDICTION_LAYER_LABELS.model}</strong> ·{" "}
                    <strong>{PREDICTION_LAYER_LABELS.human}</strong> ·{" "}
                    <strong>{PREDICTION_LAYER_LABELS.gold}</strong>. Filtering never mutates gold.
                </Alert>

                {!ctx.selectedCorpusId ? (
                    <Alert severity="warning">Select a corpus to list prediction sets.</Alert>
                ) : (
                    <>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} sx={{ mb: 2 }}>
                            <TextField
                                select
                                size="small"
                                label="Source model"
                                value={modelFilter}
                                onChange={(e) => setModelFilter(e.target.value)}
                                sx={{ minWidth: 220 }}
                            >
                                <MenuItem value="">All models</MenuItem>
                                {(modelsQuery.data ?? []).map((model) => (
                                    <MenuItem key={model.id} value={model.id}>
                                        {modelDisplayName(model)}
                                    </MenuItem>
                                ))}
                            </TextField>
                        </Stack>

                        <QueryBoundary
                            isLoading={setsQuery.isLoading}
                            isError={setsQuery.isError}
                            error={setsQuery.error}
                            onRetry={() => void setsQuery.refetch()}
                        >
                            {!sets.length ? (
                                <EmptyState
                                    title="No prediction sets"
                                    description="Run Predict on an approved or candidate model to create a prediction set."
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
                                <Box sx={{ overflowX: "auto" }}>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Set</TableCell>
                                                <TableCell>Source model</TableCell>
                                                <TableCell>Snapshot</TableCell>
                                                <TableCell>Units</TableCell>
                                                <TableCell>Created</TableCell>
                                                <TableCell align="right">Open</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {sets.map((ps) => {
                                                const model = modelsById.get(ps.trained_model_id);
                                                return (
                                                    <TableRow
                                                        key={ps.id}
                                                        selected={ps.id === selectedSetId}
                                                        hover
                                                    >
                                                        <TableCell>
                                                            <Typography
                                                                variant="body2"
                                                                sx={{ fontFamily: "monospace" }}
                                                            >
                                                                {ps.id.slice(0, 10)}…
                                                            </Typography>
                                                        </TableCell>
                                                        <TableCell>
                                                            {model
                                                                ? modelDisplayName(model)
                                                                : `${ps.trained_model_id.slice(0, 8)}… v${ps.model_version}`}
                                                        </TableCell>
                                                        <TableCell>
                                                            {ps.dataset_snapshot_id
                                                                ? `${ps.dataset_snapshot_id.slice(0, 8)}…`
                                                                : "—"}
                                                        </TableCell>
                                                        <TableCell>
                                                            {formatMetric(
                                                                num(ps.metadata?.unit_count),
                                                                0
                                                            )}
                                                        </TableCell>
                                                        <TableCell>
                                                            {new Date(
                                                                ps.created_at
                                                            ).toLocaleString()}
                                                        </TableCell>
                                                        <TableCell align="right">
                                                            <Button
                                                                size="small"
                                                                variant={
                                                                    ps.id === selectedSetId
                                                                        ? "contained"
                                                                        : "outlined"
                                                                }
                                                                onClick={() => openSet(ps.id)}
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
                    </>
                )}
            </SectionCard>

            {selectedSetId ? (
                <SectionCard
                    title="Prediction set detail"
                    description="Individual predictions with scores, uncertainty, and separate human/gold columns."
                    action={
                        <Button size="small" onClick={() => openSet("")}>
                            Close
                        </Button>
                    }
                >
                    <QueryBoundary
                        isLoading={detailQuery.isLoading}
                        isError={detailQuery.isError}
                        error={detailQuery.error}
                        onRetry={() => void detailQuery.refetch()}
                    >
                        {detail ? (
                            <Stack spacing={2}>
                                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                    <Chip
                                        size="small"
                                        color="primary"
                                        label={PREDICTION_LAYER_LABELS.model}
                                    />
                                    <Chip
                                        size="small"
                                        variant="outlined"
                                        label={PREDICTION_LAYER_LABELS.human}
                                    />
                                    <Chip
                                        size="small"
                                        variant="outlined"
                                        color="success"
                                        label={PREDICTION_LAYER_LABELS.gold}
                                    />
                                </Stack>

                                <Box>
                                    <Typography variant="body2">
                                        Source model:{" "}
                                        {sourceModelQuery.data
                                            ? modelDisplayName(sourceModelQuery.data)
                                            : detail.trained_model_id}{" "}
                                        (v{detail.model_version})
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Corpus snapshot:{" "}
                                        {detail.dataset_snapshot_id ?? "—"} · Created{" "}
                                        {new Date(detail.created_at).toLocaleString()} · Run{" "}
                                        {detail.analysis_run_id.slice(0, 8)}…
                                    </Typography>
                                </Box>

                                <Stack
                                    direction={{ xs: "column", md: "row" }}
                                    spacing={1.5}
                                    flexWrap="wrap"
                                    useFlexGap
                                >
                                    <TextField
                                        size="small"
                                        label="Predicted label"
                                        value={filters.predictedLabel}
                                        onChange={(e) =>
                                            setFilters((prev) => ({
                                                ...prev,
                                                predictedLabel: e.target.value,
                                            }))
                                        }
                                    />
                                    <TextField
                                        size="small"
                                        label="Min confidence"
                                        type="number"
                                        inputProps={{ step: 0.01, min: 0, max: 1 }}
                                        value={filters.minConfidence}
                                        onChange={(e) =>
                                            setFilters((prev) => ({
                                                ...prev,
                                                minConfidence: e.target.value,
                                            }))
                                        }
                                        sx={{ width: 140 }}
                                    />
                                    <TextField
                                        size="small"
                                        label="Max uncertainty"
                                        type="number"
                                        inputProps={{ step: 0.01, min: 0, max: 1 }}
                                        value={filters.maxUncertainty}
                                        onChange={(e) =>
                                            setFilters((prev) => ({
                                                ...prev,
                                                maxUncertainty: e.target.value,
                                            }))
                                        }
                                        sx={{ width: 150 }}
                                    />
                                    <TextField
                                        select
                                        size="small"
                                        label="Review status"
                                        value={filters.reviewStatus}
                                        onChange={(e) =>
                                            setFilters((prev) => ({
                                                ...prev,
                                                reviewStatus: e.target
                                                    .value as PredictionSetFilters["reviewStatus"],
                                            }))
                                        }
                                        sx={{ minWidth: 160 }}
                                    >
                                        <MenuItem value="">Any</MenuItem>
                                        {(
                                            [
                                                "unreviewed",
                                                "annotated",
                                                "adjudicated",
                                            ] as ReviewStatus[]
                                        ).map((status) => (
                                            <MenuItem key={status} value={status}>
                                                {status}
                                            </MenuItem>
                                        ))}
                                    </TextField>
                                    <FormControlLabel
                                        control={
                                            <Checkbox
                                                size="small"
                                                checked={filters.humanDisagreementOnly}
                                                onChange={(e) =>
                                                    setFilters((prev) => ({
                                                        ...prev,
                                                        humanDisagreementOnly: e.target.checked,
                                                    }))
                                                }
                                            />
                                        }
                                        label="Human disagreement only"
                                    />
                                </Stack>

                                <Typography variant="caption" color="text.secondary">
                                    Showing {filteredRows.length} of {rows.length} predictions
                                    {annotationsQuery.isLoading || adjudicationsQuery.isLoading
                                        ? " (loading human/gold layers…)"
                                        : ""}
                                </Typography>

                                <Box sx={{ overflowX: "auto" }}>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Unit</TableCell>
                                                <TableCell>
                                                    {PREDICTION_LAYER_LABELS.model}
                                                </TableCell>
                                                <TableCell>Scores / probs</TableCell>
                                                <TableCell align="right">Uncertainty</TableCell>
                                                <TableCell>
                                                    {PREDICTION_LAYER_LABELS.human}
                                                </TableCell>
                                                <TableCell>
                                                    {PREDICTION_LAYER_LABELS.gold}
                                                </TableCell>
                                                <TableCell>Review</TableCell>
                                                <TableCell>Flags</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {filteredRows.slice(0, 200).map((row) => (
                                                <TableRow key={row.id}>
                                                    <TableCell>
                                                        <Typography
                                                            variant="caption"
                                                            sx={{ fontFamily: "monospace" }}
                                                        >
                                                            {row.text_unit_id.slice(0, 10)}…
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>
                                                        {row.predicted_labels.join(", ") || "—"}
                                                    </TableCell>
                                                    <TableCell>
                                                        <Typography variant="caption">
                                                            {scoresSummary(row.scores)}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {formatMetric(row.uncertainty)}
                                                    </TableCell>
                                                    <TableCell>
                                                        {row.human_values.join(", ") || "—"}
                                                    </TableCell>
                                                    <TableCell>
                                                        {row.gold_values.join(", ") || "—"}
                                                    </TableCell>
                                                    <TableCell>
                                                        <Chip
                                                            size="small"
                                                            label={row.review_status}
                                                            variant="outlined"
                                                        />
                                                    </TableCell>
                                                    <TableCell>
                                                        <Stack
                                                            direction="row"
                                                            spacing={0.5}
                                                            flexWrap="wrap"
                                                            useFlexGap
                                                        >
                                                            {row.human_disagreement ? (
                                                                <Chip
                                                                    size="small"
                                                                    color="warning"
                                                                    label="human≠"
                                                                />
                                                            ) : null}
                                                            {row.model_vs_gold_disagreement ? (
                                                                <Chip
                                                                    size="small"
                                                                    color="error"
                                                                    label="model≠gold"
                                                                />
                                                            ) : null}
                                                        </Stack>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </Box>

                                <ResultsInspector
                                    title="prediction set metadata"
                                    data={{
                                        id: detail.id,
                                        trained_model_id: detail.trained_model_id,
                                        dataset_snapshot_id: detail.dataset_snapshot_id,
                                        analysis_run_id: detail.analysis_run_id,
                                        metadata: detail.metadata,
                                        note: "Predictions listed above are model outputs only; human/gold columns are joined for review and never written back as gold.",
                                    }}
                                />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
