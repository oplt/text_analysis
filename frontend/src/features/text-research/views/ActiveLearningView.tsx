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
    TablePagination,
    TextField,
    Typography,
} from "@mui/material";
import {
    ModelTraining as ActiveIcon,
    PlaylistAddCheck as AssignIcon,
    Science as PredictIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    assignUncertainPredictions,
    getRun,
    listAnnotationCampaigns,
    listClassifiers,
    listUncertainPredictions,
    predictClassifier,
} from "../../../api/textResearch";
import { listUserDirectory } from "../../../api/users";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES, researchRunStaleTime } from "../../../config/queryTiming";
import { useAuth } from "../../../hooks/useAuth";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval } from "../runPolling";
import type { UnitType } from "../types";

const PAGE_SIZE = 20;

function formatScore(value: number | null | undefined): string {
    if (value == null || Number.isNaN(value)) return "n/a";
    return value.toFixed(3);
}

export default function ActiveLearningView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const { currentUser } = useAuth();

    const modelFromUrl = searchParams.get("modelId") ?? "";
    const [selectedModelId, setSelectedModelId] = useState(modelFromUrl);
    const [page, setPage] = useState(0);
    const [selectedUnitIds, setSelectedUnitIds] = useState<string[]>([]);
    const [annotatorIds, setAnnotatorIds] = useState<string[]>(
        currentUser?.id ? [currentUser.id] : []
    );
    const [campaignId, setCampaignId] = useState("");
    const [predictUnitType, setPredictUnitType] = useState<UnitType>(
        ctx.unitType || "paragraph"
    );
    const [predictRunId, setPredictRunId] = useState<string | null>(null);
    const [hideScores, setHideScores] = useState(false);

    const offset = page * PAGE_SIZE;
    const predictSseConnected = useRunEvents(predictRunId, ctx.projectId);

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listClassifiers(ctx.projectId, ctx.selectedCorpusId, undefined, signal),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchModelLifecycle,
    });

    const campaignsQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationCampaigns(
            ctx.projectId,
            ctx.selectedCorpusId
        ),
        queryFn: ({ signal }) =>
            listAnnotationCampaigns(ctx.projectId, ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchAnnotationQueue,
    });

    const directoryQuery = useQuery({
        queryKey: queryKeys.users.directory,
        queryFn: listUserDirectory,
        staleTime: QUERY_STALE_TIMES.userDirectory,
    });

    const selectedCampaign = useMemo(
        () => (campaignsQuery.data ?? []).find((c) => c.id === campaignId) ?? null,
        [campaignsQuery.data, campaignId]
    );
    const blindBlocksPredictions = Boolean(selectedCampaign?.blind_mode);

    const queueQuery = useQuery({
        queryKey: queryKeys.textResearch.activeLearningQueue(selectedModelId, {
            offset,
            limit: PAGE_SIZE,
            campaignId: campaignId || undefined,
        }),
        queryFn: () =>
            listUncertainPredictions(selectedModelId, {
                limit: PAGE_SIZE,
                offset,
                campaignId: campaignId || undefined,
                contentMode: "snippet",
            }),
        enabled: Boolean(selectedModelId) && !blindBlocksPredictions,
        staleTime: QUERY_STALE_TIMES.researchAnnotationQueue,
    });

    const predictRunQuery = useQuery({
        queryKey: queryKeys.textResearch.run(predictRunId ?? ""),
        queryFn: ({ signal }) => getRun(predictRunId!, signal),
        enabled: Boolean(predictRunId),
        staleTime: (query) => researchRunStaleTime(query.state.data?.status),
        refetchInterval: (query) => activeRunRefetchInterval(query, predictSseConnected),
    });

    const labelNameById = useMemo(
        () => new Map(ctx.labels.map((l) => [l.id, l.name])),
        [ctx.labels]
    );

    const predictMutation = useMutation({
        mutationFn: () =>
            predictClassifier(selectedModelId, {
                unit_type: predictUnitType,
                only_unannotated: true,
            }),
        onSuccess: (run) => {
            setPredictRunId(run.id);
            showToast({
                message: "Prediction run started. Uncertain units will refresh when complete.",
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
            assignUncertainPredictions(selectedModelId, selectedUnitIds, annotatorIds),
        onSuccess: async (tasks) => {
            setSelectedUnitIds([]);
            await client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationQueueRoot,
            });
            showToast({
                message: `Assigned ${tasks.length} task(s) to annotation queue.`,
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
        if (predictRunQuery.data?.status !== "completed" || !selectedModelId) return;
        void client.invalidateQueries({
            queryKey: queryKeys.textResearch.activeLearningRoot(selectedModelId),
        });
    }, [client, predictRunQuery.data?.status, selectedModelId]);

    const items = queueQuery.data?.items ?? [];
    const total = queueQuery.data?.total ?? 0;
    const showPredictions = !blindBlocksPredictions && !hideScores;

    function selectModel(id: string) {
        setSelectedModelId(id);
        setPage(0);
        setSelectedUnitIds([]);
        const next = new URLSearchParams(searchParams);
        if (id) next.set("modelId", id);
        else next.delete("modelId");
        setSearchParams(next, { replace: true });
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Active learning"
                description="Rank uncertain model predictions, inspect snippets, assign to annotators or a campaign, then freeze and retrain."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${ctx.projectId}/classification`)}
                    >
                        Classification
                    </Button>
                }
            >
                <Stack spacing={2}>
                    <Alert severity="info">
                        Model scores stay separate from human coding. Blind campaigns never
                        fetch or display predictions for annotators under that policy.
                    </Alert>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                        <TextField
                            select
                            size="small"
                            label="Trained model"
                            value={selectedModelId}
                            onChange={(e) => selectModel(e.target.value)}
                            sx={{ minWidth: 260 }}
                        >
                            <MenuItem value="">Select model</MenuItem>
                            {(modelsQuery.data ?? []).map((model) => (
                                <MenuItem key={model.id} value={model.id}>
                                    {model.name ?? model.id.slice(0, 8)} · v{model.version}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Campaign (optional)"
                            value={campaignId}
                            onChange={(e) => {
                                setCampaignId(e.target.value);
                                setPage(0);
                            }}
                            sx={{ minWidth: 220 }}
                            helperText={
                                selectedCampaign?.blind_mode
                                    ? "Blind campaign — predictions hidden"
                                    : "Used to enforce blind policy"
                            }
                        >
                            <MenuItem value="">None</MenuItem>
                            {(campaignsQuery.data ?? []).map((campaign) => (
                                <MenuItem key={campaign.id} value={campaign.id}>
                                    {campaign.name}
                                    {campaign.blind_mode ? " · blind" : ""}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Predict unit type"
                            value={predictUnitType}
                            onChange={(e) => setPredictUnitType(e.target.value as UnitType)}
                            sx={{ minWidth: 160 }}
                        >
                            {(["document", "paragraph", "sentence"] as UnitType[]).map((unit) => (
                                <MenuItem key={unit} value={unit}>
                                    {unit}
                                </MenuItem>
                            ))}
                        </TextField>
                        <Button
                            variant="contained"
                            startIcon={<PredictIcon />}
                            disabled={!selectedModelId || predictMutation.isPending}
                            onClick={() => predictMutation.mutate()}
                        >
                            Predict unannotated
                        </Button>
                    </Stack>

                    {predictRunId && predictRunQuery.data ? (
                        <Typography variant="body2">
                            Predict run — <RunStatusChip status={predictRunQuery.data.status} />
                        </Typography>
                    ) : null}
                </Stack>
            </SectionCard>

            <SectionCard
                title="Uncertain units"
                description={`Paginated uncertainty queue (${PAGE_SIZE} snippets per page). Full texts are not loaded by default.`}
            >
                {!selectedModelId ? (
                    <EmptyState
                        icon={<ActiveIcon fontSize="large" />}
                        title="Select a trained model"
                        description="Train a classifier first, then score unannotated units to populate this queue."
                        action={
                            <Button
                                variant="contained"
                                onClick={() =>
                                    navigate(`/research/${ctx.projectId}/classification?tab=train`)
                                }
                            >
                                Go to training
                            </Button>
                        }
                    />
                ) : blindBlocksPredictions ? (
                    <Alert severity="warning">
                        Campaign &quot;{selectedCampaign?.name}&quot; is blind. Predictions are
                        not fetched or shown. Clear the campaign filter or use a non-blind
                        campaign to review uncertain units as a supervisor.
                    </Alert>
                ) : (
                    <QueryBoundary
                        isLoading={queueQuery.isLoading}
                        isError={queueQuery.isError}
                        error={queueQuery.error}
                        onRetry={() => void queueQuery.refetch()}
                    >
                        <Stack spacing={1.5}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={hideScores}
                                        onChange={(_, checked) => setHideScores(checked)}
                                    />
                                }
                                label="Hide model predictions (preview blind review)"
                            />

                            {items.length === 0 ? (
                                <Typography variant="body2" color="text.secondary">
                                    No uncertain predictions on this page. Run predict on
                                    unannotated units after training.
                                </Typography>
                            ) : null}

                            {items.map((item) => {
                                const checked = selectedUnitIds.includes(item.text_unit.id);
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
                                            py: 0.75,
                                            borderBottom: 1,
                                            borderColor: "divider",
                                        }}
                                    >
                                        <Checkbox
                                            checked={checked}
                                            onChange={(_, next) =>
                                                setSelectedUnitIds((ids) =>
                                                    next
                                                        ? [...ids, item.text_unit.id]
                                                        : ids.filter((id) => id !== item.text_unit.id)
                                                )
                                            }
                                        />
                                        <Box sx={{ flex: 1, minWidth: 0 }}>
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
                                                    variant="outlined"
                                                    label={`${item.text_unit.unit_type} #${item.text_unit.position}`}
                                                />
                                                {showPredictions ? (
                                                    <>
                                                        <Chip
                                                            size="small"
                                                            color="info"
                                                            label={`model: ${
                                                                item.prediction.predicted_labels.join(
                                                                    ", "
                                                                ) || "none"
                                                            }`}
                                                        />
                                                        <Chip
                                                            size="small"
                                                            variant="outlined"
                                                            label={`uncertainty ${formatScore(
                                                                item.prediction.uncertainty
                                                            )}`}
                                                        />
                                                        {scoreEntries.map(([label, score]) => (
                                                            <Chip
                                                                key={label}
                                                                size="small"
                                                                variant="outlined"
                                                                label={`${labelNameById.get(label) ?? label}: ${formatScore(score)}`}
                                                            />
                                                        ))}
                                                    </>
                                                ) : (
                                                    <Chip
                                                        size="small"
                                                        color="warning"
                                                        label="predictions hidden"
                                                    />
                                                )}
                                            </Stack>
                                        </Box>
                                    </Box>
                                );
                            })}

                            <TablePagination
                                component="div"
                                count={total}
                                page={page}
                                onPageChange={(_, next) => {
                                    setPage(next);
                                    setSelectedUnitIds([]);
                                }}
                                rowsPerPage={PAGE_SIZE}
                                rowsPerPageOptions={[PAGE_SIZE]}
                            />
                        </Stack>
                    </QueryBoundary>
                )}
            </SectionCard>

            <SectionCard
                title="Assign to annotators"
                description="Selected units become annotation tasks. After coding, freeze an updated dataset and retrain."
            >
                <Stack spacing={2}>
                    <TextField
                        select
                        size="small"
                        label="Annotators"
                        value={annotatorIds[0] ?? ""}
                        onChange={(e) => setAnnotatorIds(e.target.value ? [e.target.value] : [])}
                        sx={{ minWidth: 280, maxWidth: 420 }}
                        SelectProps={{ multiple: false }}
                    >
                        {(directoryQuery.data ?? []).map((user) => (
                            <MenuItem key={user.id} value={user.id}>
                                {user.full_name || user.email}
                            </MenuItem>
                        ))}
                    </TextField>
                    <Typography variant="body2" color="text.secondary">
                        {selectedUnitIds.length} unit(s) selected · {annotatorIds.length}{" "}
                        annotator(s)
                    </Typography>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                        <Button
                            variant="contained"
                            startIcon={<AssignIcon />}
                            disabled={
                                !selectedModelId ||
                                selectedUnitIds.length === 0 ||
                                annotatorIds.length === 0 ||
                                assignMutation.isPending
                            }
                            onClick={() => assignMutation.mutate()}
                        >
                            Assign selected
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={() => navigate(`/research/${ctx.projectId}/annotation`)}
                        >
                            Open annotation queue
                        </Button>
                        <Button
                            variant="outlined"
                            onClick={() =>
                                navigate(`/research/${ctx.projectId}/classification?tab=dataset`)
                            }
                        >
                            Freeze dataset → retrain
                        </Button>
                    </Stack>
                </Stack>
            </SectionCard>
        </Stack>
    );
}
