import { useEffect, useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    FormControl,
    FormControlLabel,
    InputLabel,
    LinearProgress,
    MenuItem,
    Radio,
    RadioGroup,
    Select,
    Slider,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {
    Assignment as TaskIcon,
    NavigateBefore as PrevIcon,
    NavigateNext as NextIcon,
    Save as SaveIcon,
    Verified as ReliabilityIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getAnnotationProgress,
    getTextUnitContext,
    listAnnotationQueue,
    listUncertainPredictions,
    listClassifiers,
    saveAnnotations,
} from "../../../api/textResearch";
import { AdvancedSettings } from "../../../components/ui/AdvancedSettings";
import { EmptyState } from "../../../components/ui/EmptyState";
import { HelpTooltip } from "../../../components/ui/HelpTooltip";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { useDebounce } from "../../../hooks/useDebounce";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { AnnotationLabelGuide } from "../components/AnnotationLabelGuide";
import { AnnotationSetupPanel } from "../components/AnnotationSetupPanel";
import {
    isBlindReliabilityCoding,
    shouldFetchPredictions,
} from "../annotationPredictions";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnnotationQueueItem } from "../types";

type LabelDecision = "yes" | "no" | "uncertain";

const ANNOTATION_TABS = ["setup", "workspace"] as const;
type AnnotationTab = (typeof ANNOTATION_TABS)[number];

const ANNOTATION_TAB_ITEMS: Array<{ value: AnnotationTab; label: string }> = [
    { value: "setup", label: "Setup" },
    { value: "workspace", label: "Workspace" },
];

function taskStatusColor(
    status: string
): "default" | "success" | "warning" | "info" {
    if (status === "completed") return "success";
    if (status === "in_progress") return "warning";
    if (status === "assigned") return "info";
    return "default";
}

export default function AnnotationView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [tab, setTab] = useTabQueryParam(ANNOTATION_TABS, "workspace");

    const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
    const [labelValues, setLabelValues] = useState<Record<string, LabelDecision>>({});
    const [focusedLabelIndex, setFocusedLabelIndex] = useState(0);
    const [comment, setComment] = useState("");
    const [confidence, setConfidence] = useState(0.8);
    const [queueFilter, setQueueFilter] = useState<"assigned" | "in_progress" | "all">("assigned");
    const [queuePage, setQueuePage] = useState(0);
    const [queueSearch, setQueueSearch] = useState("");
    const debouncedQueueSearch = useDebounce(queueSearch, 200);
    const queuePageSize = 50;

    const queueStatus = queueFilter === "all" ? undefined : queueFilter;
    const queueQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationQueue(queueStatus ?? "all", queuePage * queuePageSize),
        queryFn: ({ signal }) =>
            listAnnotationQueue(queueStatus, { limit: queuePageSize, offset: queuePage * queuePageSize }, signal),
        staleTime: QUERY_STALE_TIMES.researchAnnotationQueue,
    });

    const progressQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
        queryFn: ({ signal }) => getAnnotationProgress(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
        staleTime: QUERY_STALE_TIMES.researchAnnotationQueue,
    });

    const queueItems = useMemo(() => {
        const items = (queueQuery.data?.items ?? []).filter((item) => item.text_unit);
        return items as AnnotationQueueItem[];
    }, [queueQuery.data]);

    const searchableQueue = useMemo(
        () =>
            queueItems.map((item) => ({
                item,
                searchText: [
                    item.text_unit?.text ?? "",
                    item.task.status,
                    item.campaign?.name ?? "",
                ]
                    .join(" ")
                    .toLocaleLowerCase(),
            })),
        [queueItems]
    );

    const visibleQueueItems = useMemo(() => {
        const needle = debouncedQueueSearch.trim().toLocaleLowerCase();
        if (!needle) return searchableQueue.map((entry) => entry.item);
        return searchableQueue
            .filter((entry) => entry.searchText.includes(needle))
            .map((entry) => entry.item);
    }, [searchableQueue, debouncedQueueSearch]);

    const resolvedSelectedUnitId =
        selectedUnitId ?? visibleQueueItems[0]?.text_unit?.id ?? null;

    const selectedIndex = visibleQueueItems.findIndex(
        (item) => item.text_unit?.id === resolvedSelectedUnitId
    );
    const selectedItem = selectedIndex >= 0 ? visibleQueueItems[selectedIndex] : null;

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listClassifiers(ctx.projectId, ctx.selectedCorpusId, undefined, signal),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });
    const selectedModelId = classifiersQuery.data?.[0]?.id ?? null;

    const predictionsEnabled = shouldFetchPredictions({
        selectedModelId,
        blindPolicy: selectedItem?.blind_policy,
        campaign: selectedItem?.campaign,
    });

    const predictionsQuery = useQuery({
        queryKey: [
            "text-research",
            "uncertain-predictions",
            selectedModelId,
            selectedItem?.task.campaign_id,
            resolvedSelectedUnitId,
        ],
        queryFn: ({ signal }) =>
            listUncertainPredictions(
                selectedModelId!,
                {
                    campaignId: selectedItem?.task.campaign_id ?? undefined,
                    textUnitId: resolvedSelectedUnitId ?? undefined,
                },
                signal
            ),
        enabled: predictionsEnabled,
    });

    const contextQuery = useQuery({
        queryKey: ["text-research", "unit-context", resolvedSelectedUnitId],
        queryFn: ({ signal }) => getTextUnitContext(resolvedSelectedUnitId!, 2, signal),
        enabled: Boolean(resolvedSelectedUnitId),
    });

    const modelPrediction = predictionsQuery.data?.items.find(
        (item) => item.text_unit.id === resolvedSelectedUnitId
    );
    const blindCoding = isBlindReliabilityCoding(
        selectedItem?.blind_policy,
        selectedItem?.campaign
    );
    const aiAssisted =
        !blindCoding &&
        (selectedItem?.blind_policy?.ai_assistance_enabled === true ||
            selectedItem?.campaign?.ai_assistance_enabled === true ||
            selectedItem?.campaign?.annotation_mode === "ai_assisted");

    useEffect(() => {
        setLabelValues({});
        setComment("");
        setConfidence(0.8);
        setFocusedLabelIndex(0);
    }, [resolvedSelectedUnitId]);

    function selectIndex(index: number) {
        const item = visibleQueueItems[index];
        if (item?.text_unit) setSelectedUnitId(item.text_unit.id);
    }

    function setDecisionForFocusedLabel(decision: LabelDecision) {
        const label = ctx.labels[focusedLabelIndex] ?? ctx.labels[0];
        if (!label) return;
        setLabelValues((current) => ({ ...current, [label.id]: decision }));
    }

    const saveMutation = useMutation({
        mutationFn: async (options: { complete: boolean; advance: boolean }) => {
            if (!resolvedSelectedUnitId || !ctx.selectedCodebookId) {
                throw new Error("Select a text unit and codebook.");
            }
            const values = ctx.labels.map((label) => ({
                label_id: label.id,
                value: labelValues[label.id] ?? "no",
                confidence,
                comment: comment.trim() || undefined,
            }));
            await saveAnnotations({
                text_unit_id: resolvedSelectedUnitId,
                codebook_id: ctx.selectedCodebookId,
                campaign_id: selectedItem?.task.campaign_id ?? undefined,
                values,
                mark_task_complete: options.complete,
            });
            return options;
        },
        onSuccess: (options) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationQueueRoot,
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
            });
            showToast({
                message: options.complete ? "Saved and marked complete." : "Draft saved.",
                severity: "success",
            });
            if (options.advance) {
                const next = visibleQueueItems[selectedIndex + 1];
                if (next?.text_unit) setSelectedUnitId(next.text_unit.id);
            }
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save annotations."),
                severity: "error",
            }),
    });

    useEffect(() => {
        function onKeyDown(event: KeyboardEvent) {
            const target = event.target as HTMLElement | null;
            if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
            if (tab !== "workspace") return;

            if (event.key === "ArrowRight" || event.key === "j") {
                event.preventDefault();
                selectIndex(Math.min(visibleQueueItems.length - 1, selectedIndex + 1));
            } else if (event.key === "ArrowLeft" || event.key === "k") {
                event.preventDefault();
                selectIndex(Math.max(0, selectedIndex - 1));
            } else if (event.key === "Tab" && ctx.labels.length > 0) {
                event.preventDefault();
                const delta = event.shiftKey ? -1 : 1;
                setFocusedLabelIndex(
                    (index) => (index + delta + ctx.labels.length) % ctx.labels.length
                );
            } else if (event.key === "y" || event.key === "1") {
                event.preventDefault();
                setDecisionForFocusedLabel("yes");
            } else if (event.key === "n" || event.key === "2") {
                event.preventDefault();
                setDecisionForFocusedLabel("no");
            } else if (event.key === "u" || event.key === "3") {
                event.preventDefault();
                setDecisionForFocusedLabel("uncertain");
            } else if (event.key === "s" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                saveMutation.mutate({ complete: true, advance: true });
            } else if (event.key === "Enter" && !event.metaKey && !event.ctrlKey) {
                event.preventDefault();
                selectIndex(Math.min(visibleQueueItems.length - 1, selectedIndex + 1));
            }
        }
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [visibleQueueItems, selectedIndex, saveMutation, tab, ctx.labels, focusedLabelIndex]);

    const completionRate = Math.round((progressQuery.data?.completion_rate ?? 0) * 100);
    const decidedCount = ctx.labels.filter((label) => Boolean(labelValues[label.id])).length;

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={ANNOTATION_TAB_ITEMS}
                ariaLabel="Annotation workflow"
            />

            {tab === "setup" ? (
                <Box id="annotation-setup">
                    <SectionCard
                        title="Annotation setup"
                        description="Create tasks with sample size, annotators, and overlap for reliability."
                    >
                        <AnnotationSetupPanel />
                    </SectionCard>
                </Box>
            ) : null}

            {tab === "workspace" ? (
                <SectionCard
                    title="Annotation workspace"
                    description="Code the unit text first, then apply labels. Model suggestions stay separate and are never auto-applied."
                    action={
                        <Stack direction="row" spacing={1} alignItems="center">
                            <Button
                                size="small"
                                variant="outlined"
                                startIcon={<ReliabilityIcon />}
                                onClick={() =>
                                    navigate(`/research/${ctx.projectId}/reliability?tab=disagreements`)
                                }
                            >
                                Disagreements
                            </Button>
                            <FormControl size="small" sx={{ minWidth: 140 }}>
                                <InputLabel>Queue</InputLabel>
                                <Select
                                    label="Queue"
                                    value={queueFilter}
                                    onChange={(e) => {
                                        setQueueFilter(e.target.value as typeof queueFilter);
                                        setQueuePage(0);
                                        setSelectedUnitId(null);
                                    }}
                                >
                                    <MenuItem value="assigned">Assigned</MenuItem>
                                    <MenuItem value="in_progress">In progress</MenuItem>
                                    <MenuItem value="all">All</MenuItem>
                                </Select>
                            </FormControl>
                        </Stack>
                    }
                >
                    {progressQuery.data ? (
                        <Box sx={{ mb: 2 }}>
                            <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={1}
                                justifyContent="space-between"
                                alignItems={{ sm: "center" }}
                                sx={{ mb: 0.75 }}
                            >
                                <Typography variant="body2" color="text.secondary">
                                    Progress: {progressQuery.data.completed_tasks}/
                                    {progressQuery.data.total_tasks} tasks ({completionRate}%)
                                </Typography>
                                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                                    {selectedItem?.task.status ? (
                                        <Chip
                                            size="small"
                                            color={taskStatusColor(selectedItem.task.status)}
                                            label={`Status: ${selectedItem.task.status}`}
                                        />
                                    ) : null}
                                    {blindCoding ? (
                                        <Chip size="small" color="info" label="Blind reliability" />
                                    ) : null}
                                    {aiAssisted ? (
                                        <Chip size="small" color="warning" label="AI-assisted" />
                                    ) : null}
                                    {ctx.labels.length > 0 ? (
                                        <Chip
                                            size="small"
                                            variant="outlined"
                                            label={`${decidedCount}/${ctx.labels.length} labels decided`}
                                        />
                                    ) : null}
                                </Stack>
                            </Stack>
                            <LinearProgress variant="determinate" value={completionRate} />
                        </Box>
                    ) : null}

                    {!ctx.selectedCodebookId || ctx.labels.length === 0 ? (
                        <Alert
                            severity="info"
                            sx={{ mb: 2 }}
                            action={
                                <Button
                                    color="inherit"
                                    size="small"
                                    onClick={() => navigate(`/research/${ctx.projectId}/codebook`)}
                                >
                                    Open codebook
                                </Button>
                            }
                        >
                            Select or create a codebook with labels before coding.
                        </Alert>
                    ) : null}

                    <QueryBoundary
                        isLoading={queueQuery.isLoading}
                        isError={queueQuery.isError}
                        error={queueQuery.error}
                        onRetry={() => void queueQuery.refetch()}
                    >
                        {!queueItems.length ? (
                            <EmptyState
                                icon={<TaskIcon fontSize="large" />}
                                title="No tasks in this queue"
                                description="Use Annotation setup to assign units, then return here to code."
                                action={
                                    <Button variant="contained" onClick={() => setTab("setup")}>
                                        Create annotation tasks
                                    </Button>
                                }
                            />
                        ) : (
                            <Box
                                sx={{
                                    display: "grid",
                                    gap: 2,
                                    gridTemplateColumns: {
                                        xs: "1fr",
                                        md: "minmax(200px, 0.9fr) minmax(0, 2.4fr)",
                                        lg: "minmax(200px, 0.85fr) minmax(0, 2.2fr) minmax(240px, 1fr)",
                                    },
                                    alignItems: "start",
                                }}
                            >
                                <Stack spacing={1} sx={{ maxHeight: { md: "70vh" }, overflow: "auto" }}>
                                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                                        <Typography variant="subtitle2">
                                            Queue {queueQuery.data ? `(${queueQuery.data.total})` : ""}
                                        </Typography>
                                        <Stack direction="row" spacing={0.5}>
                                            <Button
                                                size="small"
                                                disabled={queuePage === 0}
                                                onClick={() => {
                                                    setQueuePage((page) => page - 1);
                                                    setSelectedUnitId(null);
                                                }}
                                            >
                                                Earlier
                                            </Button>
                                            <Button
                                                size="small"
                                                disabled={
                                                    !queueQuery.data ||
                                                    (queuePage + 1) * queuePageSize >= queueQuery.data.total
                                                }
                                                onClick={() => {
                                                    setQueuePage((page) => page + 1);
                                                    setSelectedUnitId(null);
                                                }}
                                            >
                                                Later
                                            </Button>
                                        </Stack>
                                    </Stack>
                                    <TextField
                                        size="small"
                                        label="Filter page"
                                        value={queueSearch}
                                        onChange={(event) => setQueueSearch(event.target.value)}
                                        placeholder="Search text / status"
                                    />
                                    {visibleQueueItems.map((item, index) => (
                                        <Button
                                            key={item.task.id}
                                            size="small"
                                            variant={
                                                item.text_unit?.id === resolvedSelectedUnitId
                                                    ? "contained"
                                                    : "outlined"
                                            }
                                            onClick={() => selectIndex(index)}
                                            sx={{
                                                justifyContent: "flex-start",
                                                textAlign: "left",
                                                alignItems: "flex-start",
                                                py: 1,
                                            }}
                                        >
                                            <Stack spacing={0.35} sx={{ width: "100%" }}>
                                                <Stack direction="row" spacing={0.75} alignItems="center">
                                                    <Typography variant="caption" component="span">
                                                        #{queuePage * queuePageSize + index + 1}
                                                    </Typography>
                                                    <Chip
                                                        size="small"
                                                        label={item.task.status}
                                                        color={taskStatusColor(item.task.status)}
                                                        sx={{ height: 20 }}
                                                    />
                                                </Stack>
                                                <Typography variant="body2" component="span">
                                                    {item.text_unit?.text.slice(0, 64) ?? "Unit"}…
                                                </Typography>
                                            </Stack>
                                        </Button>
                                    ))}
                                </Stack>

                                <Stack spacing={1.5}>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        <Button
                                            startIcon={<PrevIcon />}
                                            disabled={selectedIndex <= 0}
                                            onClick={() => selectIndex(selectedIndex - 1)}
                                        >
                                            Previous
                                        </Button>
                                        <Button
                                            endIcon={<NextIcon />}
                                            disabled={selectedIndex >= visibleQueueItems.length - 1}
                                            onClick={() => selectIndex(selectedIndex + 1)}
                                        >
                                            Next
                                        </Button>
                                        <Button
                                            variant="contained"
                                            startIcon={<SaveIcon />}
                                            disabled={
                                                !resolvedSelectedUnitId ||
                                                !ctx.selectedCodebookId ||
                                                saveMutation.isPending
                                            }
                                            onClick={() =>
                                                saveMutation.mutate({ complete: true, advance: true })
                                            }
                                        >
                                            Save & Next
                                        </Button>
                                        <Button
                                            variant="outlined"
                                            disabled={
                                                !resolvedSelectedUnitId ||
                                                !ctx.selectedCodebookId ||
                                                saveMutation.isPending
                                            }
                                            onClick={() =>
                                                saveMutation.mutate({ complete: false, advance: false })
                                            }
                                        >
                                            Save draft
                                        </Button>
                                    </Stack>
                                    <Typography variant="caption" color="text.secondary">
                                        Shortcuts: ←/k · →/j · Tab label · y/n/u or 1/2/3 · ⌘/Ctrl+S save &
                                        next
                                    </Typography>

                                    {selectedItem?.text_unit ? (
                                        <>
                                            {contextQuery.data?.document ? (
                                                <Typography variant="body2" color="text.secondary">
                                                    Source: {contextQuery.data.document.title ?? "Untitled"}
                                                    {contextQuery.data.document.organization
                                                        ? ` · ${contextQuery.data.document.organization}`
                                                        : ""}
                                                    {contextQuery.data.document.publication_year
                                                        ? ` · ${contextQuery.data.document.publication_year}`
                                                        : ""}
                                                </Typography>
                                            ) : null}

                                            <Box
                                                sx={{
                                                    p: 2,
                                                    borderRadius: 1,
                                                    border: "2px solid",
                                                    borderColor: "primary.main",
                                                    bgcolor: "background.paper",
                                                }}
                                            >
                                                <Typography variant="overline" color="text.secondary">
                                                    Text to code
                                                </Typography>
                                                <Typography variant="body1" sx={{ mt: 0.5 }}>
                                                    {selectedItem.text_unit.text}
                                                </Typography>
                                            </Box>

                                            {blindCoding ? (
                                                <Alert
                                                    severity="info"
                                                    action={<HelpTooltip termId="blind_reliability" />}
                                                >
                                                    Blind reliability coding — peer codes and model
                                                    suggestions are hidden.
                                                </Alert>
                                            ) : null}

                                            {aiAssisted ? (
                                                <Alert
                                                    severity="warning"
                                                    action={<HelpTooltip termId="ai_assisted_coding" />}
                                                >
                                                    AI-assisted mode — suggestions below are optional and
                                                    never auto-applied.
                                                </Alert>
                                            ) : null}

                                            {!blindCoding && modelPrediction ? (
                                                <Alert severity="warning">
                                                    Model suggestion (not applied):{" "}
                                                    {modelPrediction.prediction.predicted_labels.join(
                                                        ", "
                                                    ) || "none"}
                                                    {modelPrediction.prediction.uncertainty != null
                                                        ? ` · uncertainty ${modelPrediction.prediction.uncertainty.toFixed(3)}`
                                                        : ""}
                                                </Alert>
                                            ) : null}

                                            <Box>
                                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                                    Labels
                                                </Typography>
                                                <Stack spacing={1.25}>
                                                    {ctx.labels.map((label, index) => (
                                                        <Box
                                                            key={label.id}
                                                            onClick={() => setFocusedLabelIndex(index)}
                                                            sx={{
                                                                p: 1.25,
                                                                borderRadius: 1,
                                                                border: "1px solid",
                                                                borderColor:
                                                                    focusedLabelIndex === index
                                                                        ? "primary.main"
                                                                        : "divider",
                                                                bgcolor:
                                                                    focusedLabelIndex === index
                                                                        ? "action.hover"
                                                                        : "transparent",
                                                            }}
                                                        >
                                                            <Typography variant="subtitle2">
                                                                {label.name}
                                                            </Typography>
                                                            <RadioGroup
                                                                row
                                                                value={labelValues[label.id] ?? ""}
                                                                onChange={(e) => {
                                                                    setFocusedLabelIndex(index);
                                                                    setLabelValues((current) => ({
                                                                        ...current,
                                                                        [label.id]: e.target
                                                                            .value as LabelDecision,
                                                                    }));
                                                                }}
                                                            >
                                                                <FormControlLabel
                                                                    value="yes"
                                                                    control={<Radio size="small" />}
                                                                    label="Yes"
                                                                />
                                                                <FormControlLabel
                                                                    value="no"
                                                                    control={<Radio size="small" />}
                                                                    label="No"
                                                                />
                                                                <FormControlLabel
                                                                    value="uncertain"
                                                                    control={<Radio size="small" />}
                                                                    label="Uncertain"
                                                                />
                                                            </RadioGroup>
                                                        </Box>
                                                    ))}
                                                </Stack>
                                            </Box>

                                            <AdvancedSettings
                                                title="Confidence & comment"
                                                description="Optional annotator notes"
                                            >
                                                <Stack spacing={1.25}>
                                                    <Typography variant="caption">
                                                        Confidence: {confidence.toFixed(2)}
                                                    </Typography>
                                                    <Slider
                                                        min={0}
                                                        max={1}
                                                        step={0.05}
                                                        value={confidence}
                                                        onChange={(_, value) =>
                                                            setConfidence(value as number)
                                                        }
                                                    />
                                                    <TextField
                                                        label="Comment"
                                                        value={comment}
                                                        onChange={(e) => setComment(e.target.value)}
                                                        fullWidth
                                                        multiline
                                                        minRows={2}
                                                    />
                                                </Stack>
                                            </AdvancedSettings>

                                            <AdvancedSettings
                                                title="Evidence / context"
                                                description="Surrounding units (optional)"
                                            >
                                                <Stack spacing={1}>
                                                    {(contextQuery.data?.before ?? []).map((unit) => (
                                                        <Typography
                                                            key={unit.id}
                                                            variant="body2"
                                                            color="text.disabled"
                                                            sx={{ fontStyle: "italic" }}
                                                        >
                                                            {unit.text}
                                                        </Typography>
                                                    ))}
                                                    {(contextQuery.data?.after ?? []).map((unit) => (
                                                        <Typography
                                                            key={unit.id}
                                                            variant="body2"
                                                            color="text.disabled"
                                                            sx={{ fontStyle: "italic" }}
                                                        >
                                                            {unit.text}
                                                        </Typography>
                                                    ))}
                                                    {!contextQuery.data?.before.length &&
                                                    !contextQuery.data?.after.length ? (
                                                        <Typography variant="body2" color="text.secondary">
                                                            No surrounding units loaded.
                                                        </Typography>
                                                    ) : null}
                                                </Stack>
                                            </AdvancedSettings>
                                        </>
                                    ) : (
                                        <Typography color="text.secondary">
                                            Select a queued unit to begin coding.
                                        </Typography>
                                    )}
                                </Stack>

                                <Box sx={{ display: { xs: "none", lg: "block" } }}>
                                    <AnnotationLabelGuide
                                        labels={ctx.labels}
                                        codebookName={ctx.selectedCodebook?.name}
                                        codebookVersion={ctx.selectedCodebook?.version}
                                        frozen={ctx.selectedCodebook?.is_frozen}
                                    />
                                </Box>
                            </Box>
                        )}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
