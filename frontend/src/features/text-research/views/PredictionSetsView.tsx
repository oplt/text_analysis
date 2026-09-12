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
import { Science as PredictionIcon } from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
    getClassifier,
    listModels,
    listPredictionSets,
    listPredictionSetPredictions,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { ResultsInspector } from "../components/ResearchCharts";
import { useResearchContext } from "../hooks/useResearchContext";
import { formatMetric, modelDisplayName, num } from "../modelRegistryUtils";
import {
    DEFAULT_PREDICTION_SET_FILTERS,
    PREDICTION_LAYER_LABELS,
    type PredictionSetFilters,
    type ReviewStatus,
} from "../predictionSetFilters";
import type { TrainedModel } from "../types";

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
    const [pageOffset, setPageOffset] = useState(0);

    const setsQuery = useQuery({
        queryKey: queryKeys.textResearch.predictionSets(ctx.selectedCorpusId),
        queryFn: () => listPredictionSets(ctx.selectedCorpusId, { limit: 100 }),
        enabled: Boolean(ctx.selectedCorpusId),
        staleTime: QUERY_STALE_TIMES.researchPredictionSets,
    });

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.models(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listModels(ctx.projectId, { corpusId: ctx.selectedCorpusId || undefined }, signal),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchModelLifecycle,
    });

    const modelsById = (() => {
        const map = new Map<string, TrainedModel>();
        for (const model of modelsQuery.data ?? []) {
            map.set(model.id, model);
        }
        return map;
    })();

    const sets = useMemo(() => {
        const all = setsQuery.data ?? [];
        if (!modelFilter) return all;
        return all.filter((item) => item.trained_model_id === modelFilter);
    }, [setsQuery.data, modelFilter]);

    const selectedSet = (setsQuery.data ?? []).find((item) => item.id === selectedSetId) ?? null;
    const pageQuery = useQuery({
        queryKey: [
            ...queryKeys.textResearch.predictionSetPageRoot(selectedSetId),
            pageOffset,
            filters,
        ],
        queryFn: () =>
            listPredictionSetPredictions(selectedSetId, {
                limit: 100,
                offset: pageOffset,
                predictedLabel: filters.predictedLabel || undefined,
                minConfidence: filters.minConfidence ? Number(filters.minConfidence) : undefined,
                maxUncertainty: filters.maxUncertainty ? Number(filters.maxUncertainty) : undefined,
                reviewStatus: filters.reviewStatus || undefined,
                humanDisagreement: filters.humanDisagreementOnly || undefined,
            }),
        enabled: Boolean(selectedSetId),
        staleTime: QUERY_STALE_TIMES.researchPredictionSets,
    });

    const sourceModelQuery = useQuery({
        queryKey: queryKeys.textResearch.model(selectedSet?.trained_model_id ?? ""),
        queryFn: () => getClassifier(selectedSet!.trained_model_id),
        enabled: Boolean(selectedSet?.trained_model_id),
    });

    const rows = useMemo(() => (pageQuery.data?.items ?? []).map((item) => {
        const humanValues = [...new Set(item.human_annotations.map((row) => row.value))].sort();
        const goldValues = [...new Set(item.adjudications.map((row) => row.final_value))].sort();
        const predicted = new Set(item.prediction.predicted_labels);
        const gold = new Set(goldValues);
        return {
            ...item.prediction,
            human_values: humanValues,
            gold_values: goldValues,
            review_status: item.review_status,
            human_disagreement: item.human_disagreement,
            model_vs_gold_disagreement: gold.size > 0 &&
                (predicted.size !== gold.size || [...predicted].some((label) => !gold.has(label))),
        };
    }), [pageQuery.data]);

    const openSet = (id: string) => {
        setSearchParams(id ? { set: id } : {});
        setFilters(DEFAULT_PREDICTION_SET_FILTERS);
        setPageOffset(0);
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
                                    icon={<PredictionIcon />}
                                    title="No prediction sets"
                                    description="Run Predict on a candidate, staging, or production model to create a prediction set."
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
                        isLoading={pageQuery.isLoading}
                        isError={pageQuery.isError}
                        error={pageQuery.error}
                        onRetry={() => void pageQuery.refetch()}
                    >
                        {selectedSet ? (
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
                                            : selectedSet.trained_model_id}{" "}
                                        (v{selectedSet.model_version})
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Corpus snapshot:{" "}
                                        {selectedSet.dataset_snapshot_id ?? "—"} · Created{" "}
                                        {new Date(selectedSet.created_at).toLocaleString()} · Run{" "}
                                        {selectedSet.analysis_run_id.slice(0, 8)}…
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
                                    Showing {pageOffset + 1}–{pageOffset + rows.length} of {pageQuery.data?.total ?? 0} predictions
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
                                            {rows.map((row) => (
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

                                <Stack direction="row" spacing={1}>
                                    <Button size="small" disabled={pageOffset === 0} onClick={() => setPageOffset((offset) => Math.max(0, offset - 100))}>Previous</Button>
                                    <Button size="small" disabled={pageOffset + rows.length >= (pageQuery.data?.total ?? 0)} onClick={() => setPageOffset((offset) => offset + 100)}>Next</Button>
                                </Stack>

                                <ResultsInspector
                                    title="prediction set metadata"
                                    data={{
                                        id: selectedSet.id,
                                        trained_model_id: selectedSet.trained_model_id,
                                        dataset_snapshot_id: selectedSet.dataset_snapshot_id,
                                        analysis_run_id: selectedSet.analysis_run_id,
                                        metadata: selectedSet.metadata,
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
