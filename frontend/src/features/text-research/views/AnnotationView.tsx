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
import { EmptyState } from "../../../components/ui/EmptyState";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { AnnotationSetupPanel } from "../components/AnnotationSetupPanel";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnnotationLabel, AnnotationQueueItem } from "../types";

type LabelDecision = "yes" | "no" | "uncertain";

const ANNOTATION_TABS = ["setup", "workspace"] as const;
type AnnotationTab = (typeof ANNOTATION_TABS)[number];

const ANNOTATION_TAB_ITEMS: Array<{ value: AnnotationTab; label: string }> = [
    { value: "setup", label: "Setup" },
    { value: "workspace", label: "Workspace" },
];

function LabelGuide({ label }: { label: AnnotationLabel }) {
    return (
        <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: "action.hover" }}>
            <Typography variant="subtitle2">{label.name}</Typography>
            {label.description ? (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {label.description}
                </Typography>
            ) : null}
            {label.inclusion_criteria ? (
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Include: {label.inclusion_criteria}
                </Typography>
            ) : null}
            {label.exclusion_criteria ? (
                <Typography variant="caption" display="block">
                    Exclude: {label.exclusion_criteria}
                </Typography>
            ) : null}
            {label.positive_examples?.length ? (
                <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    + {label.positive_examples.join(" · ")}
                </Typography>
            ) : null}
            {label.negative_examples?.length ? (
                <Typography variant="caption" display="block">
                    − {label.negative_examples.join(" · ")}
                </Typography>
            ) : null}
            {label.is_placeholder ? <Chip size="small" label="demo" sx={{ mt: 1 }} /> : null}
        </Box>
    );
}

export default function AnnotationView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [tab, setTab] = useTabQueryParam(ANNOTATION_TABS, "workspace");

    const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
    const [labelValues, setLabelValues] = useState<Record<string, LabelDecision>>({});
    const [comment, setComment] = useState("");
    const [confidence, setConfidence] = useState(0.8);
    const [queueFilter, setQueueFilter] = useState<"assigned" | "in_progress" | "all">("assigned");
    const [queuePage, setQueuePage] = useState(0);
    const queuePageSize = 50;

    const queueStatus = queueFilter === "all" ? undefined : queueFilter;
    const queueQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationQueue(queueStatus ?? "all", queuePage * queuePageSize),
        queryFn: () => listAnnotationQueue(queueStatus, { limit: queuePageSize, offset: queuePage * queuePageSize }),
    });

    const progressQuery = useQuery({
        queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
        queryFn: () => getAnnotationProgress(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const contextQuery = useQuery({
        queryKey: ["text-research", "unit-context", selectedUnitId],
        queryFn: () => getTextUnitContext(selectedUnitId!, 2),
        enabled: Boolean(selectedUnitId),
    });

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listClassifiers(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });
    const selectedModelId = classifiersQuery.data?.[0]?.id ?? null;

    const predictionsQuery = useQuery({
        queryKey: ["text-research", "uncertain-predictions", selectedModelId],
        queryFn: () => listUncertainPredictions(selectedModelId!),
        enabled: Boolean(selectedModelId),
    });

    const queueItems = useMemo(() => {
        const items = (queueQuery.data?.items ?? []).filter((item) => item.text_unit);
        return items as AnnotationQueueItem[];
    }, [queueQuery.data]);

    const selectedIndex = queueItems.findIndex((item) => item.text_unit?.id === selectedUnitId);
    const selectedItem = selectedIndex >= 0 ? queueItems[selectedIndex] : null;
    const modelPrediction = predictionsQuery.data?.find(
        (item) => item.text_unit.id === selectedUnitId
    );

    useEffect(() => {
        if (!selectedUnitId && queueItems[0]?.text_unit?.id) {
            setSelectedUnitId(queueItems[0].text_unit.id);
        }
    }, [queueItems, selectedUnitId]);

    useEffect(() => {
        setLabelValues({});
        setComment("");
        setConfidence(0.8);
    }, [selectedUnitId]);

    function selectIndex(index: number) {
        const item = queueItems[index];
        if (item?.text_unit) setSelectedUnitId(item.text_unit.id);
    }

    const saveMutation = useMutation({
        mutationFn: async (options: { complete: boolean; advance: boolean }) => {
            if (!selectedUnitId || !ctx.selectedCodebookId) {
                throw new Error("Select a text unit and codebook.");
            }
            const values = ctx.labels.map((label) => ({
                label_id: label.id,
                value: labelValues[label.id] ?? "no",
                confidence,
                comment: comment.trim() || undefined,
            }));
            await saveAnnotations({
                text_unit_id: selectedUnitId,
                codebook_id: ctx.selectedCodebookId,
                values,
                mark_task_complete: options.complete,
            });
            return options;
        },
        onSuccess: (options) => {
            void client.invalidateQueries({
                queryKey: ["text-research", "annotation-queue"],
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.annotationProgress(ctx.selectedCorpusId),
            });
            showToast({
                message: options.complete ? "Saved and marked complete." : "Draft saved.",
                severity: "success",
            });
            if (options.advance) {
                const next = queueItems[selectedIndex + 1];
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

            if (event.key === "ArrowRight" || event.key === "j") {
                event.preventDefault();
                selectIndex(Math.min(queueItems.length - 1, selectedIndex + 1));
            } else if (event.key === "ArrowLeft" || event.key === "k") {
                event.preventDefault();
                selectIndex(Math.max(0, selectedIndex - 1));
            } else if (event.key === "s" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                saveMutation.mutate({ complete: true, advance: true });
            } else if (event.key === "n" && !event.metaKey && !event.ctrlKey) {
                event.preventDefault();
                selectIndex(Math.min(queueItems.length - 1, selectedIndex + 1));
            }
        }
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [queueItems, selectedIndex, saveMutation]);

    const completionRate = Math.round((progressQuery.data?.completion_rate ?? 0) * 100);

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
                description="Code queue units with codebook guidance. Model predictions stay visually separate and are never auto-applied."
                action={
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
                }
            >
                {progressQuery.data ? (
                    <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                            Progress: {progressQuery.data.completed_tasks}/
                            {progressQuery.data.total_tasks} tasks ({completionRate}%)
                        </Typography>
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
                                <Button
                                    variant="contained"
                                    onClick={() => setTab("setup")}
                                >
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
                                    lg: "240px minmax(0, 1fr) 280px",
                                },
                                alignItems: "start",
                            }}
                        >
                            <Stack spacing={1} sx={{ maxHeight: 640, overflow: "auto" }}>
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
                                            disabled={!queueQuery.data || (queuePage + 1) * queuePageSize >= queueQuery.data.total}
                                            onClick={() => {
                                                setQueuePage((page) => page + 1);
                                                setSelectedUnitId(null);
                                            }}
                                        >
                                            Later
                                        </Button>
                                    </Stack>
                                </Stack>
                                {queueItems.map((item, index) => (
                                    <Button
                                        key={item.task.id}
                                        size="small"
                                        variant={
                                            item.text_unit?.id === selectedUnitId
                                                ? "contained"
                                                : "outlined"
                                        }
                                        onClick={() => selectIndex(index)}
                                        sx={{ justifyContent: "flex-start", textAlign: "left" }}
                                    >
                                        #{queuePage * queuePageSize + index + 1} · {item.text_unit?.text.slice(0, 48) ?? "Unit"}
                                        …
                                    </Button>
                                ))}
                            </Stack>

                            <Stack spacing={1.5}>
                                <Stack direction="row" spacing={1}>
                                    <Button
                                        startIcon={<PrevIcon />}
                                        disabled={selectedIndex <= 0}
                                        onClick={() => selectIndex(selectedIndex - 1)}
                                    >
                                        Previous
                                    </Button>
                                    <Button
                                        endIcon={<NextIcon />}
                                        disabled={selectedIndex >= queueItems.length - 1}
                                        onClick={() => selectIndex(selectedIndex + 1)}
                                    >
                                        Next
                                    </Button>
                                    <Button
                                        onClick={() =>
                                            selectIndex(
                                                Math.min(queueItems.length - 1, selectedIndex + 1)
                                            )
                                        }
                                    >
                                        Skip
                                    </Button>
                                    <Button
                                        variant="contained"
                                        startIcon={<SaveIcon />}
                                        disabled={
                                            !selectedUnitId ||
                                            !ctx.selectedCodebookId ||
                                            saveMutation.isPending
                                        }
                                        onClick={() =>
                                            saveMutation.mutate({ complete: true, advance: true })
                                        }
                                    >
                                        Save & Next
                                    </Button>
                                </Stack>
                                <Typography variant="caption" color="text.secondary">
                                    Shortcuts: ←/k previous · →/j next · n skip · ⌘/Ctrl+S save & next
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

                                        {contextQuery.data?.before.map((unit) => (
                                            <Typography
                                                key={unit.id}
                                                variant="body2"
                                                color="text.disabled"
                                                sx={{ fontStyle: "italic" }}
                                            >
                                                {unit.text}
                                            </Typography>
                                        ))}

                                        <Box
                                            sx={{
                                                p: 2,
                                                borderRadius: 1,
                                                border: "1px solid",
                                                borderColor: "primary.main",
                                                bgcolor: "background.paper",
                                            }}
                                        >
                                            <Typography variant="body1">
                                                {selectedItem.text_unit.text}
                                            </Typography>
                                        </Box>

                                        {contextQuery.data?.after.map((unit) => (
                                            <Typography
                                                key={unit.id}
                                                variant="body2"
                                                color="text.disabled"
                                                sx={{ fontStyle: "italic" }}
                                            >
                                                {unit.text}
                                            </Typography>
                                        ))}

                                        {modelPrediction ? (
                                            <Alert severity="warning">
                                                Model suggestion (not applied):{" "}
                                                {modelPrediction.prediction.predicted_labels.join(", ") ||
                                                    "none"}
                                                {modelPrediction.prediction.uncertainty != null
                                                    ? ` · uncertainty ${modelPrediction.prediction.uncertainty.toFixed(3)}`
                                                    : ""}
                                            </Alert>
                                        ) : null}

                                        <Stack spacing={1.5}>
                                            {ctx.labels.map((label) => (
                                                <Box key={label.id}>
                                                    <Typography variant="subtitle2">{label.name}</Typography>
                                                    <RadioGroup
                                                        row
                                                        value={labelValues[label.id] ?? ""}
                                                        onChange={(e) =>
                                                            setLabelValues((current) => ({
                                                                ...current,
                                                                [label.id]: e.target
                                                                    .value as LabelDecision,
                                                            }))
                                                        }
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

                                        <Typography variant="caption">
                                            Confidence: {confidence.toFixed(2)}
                                        </Typography>
                                        <Slider
                                            min={0}
                                            max={1}
                                            step={0.05}
                                            value={confidence}
                                            onChange={(_, value) => setConfidence(value as number)}
                                        />
                                        <TextField
                                            label="Comment"
                                            value={comment}
                                            onChange={(e) => setComment(e.target.value)}
                                            fullWidth
                                            multiline
                                            minRows={2}
                                        />
                                    </>
                                ) : (
                                    <Typography color="text.secondary">
                                        Select a queued unit to begin coding.
                                    </Typography>
                                )}
                            </Stack>

                            <Stack spacing={1} sx={{ maxHeight: 640, overflow: "auto" }}>
                                <Typography variant="subtitle2">
                                    Codebook {ctx.selectedCodebook?.name}
                                    {ctx.selectedCodebook
                                        ? ` · v${ctx.selectedCodebook.version}`
                                        : ""}
                                    {ctx.selectedCodebook?.is_frozen ? " · frozen" : ""}
                                </Typography>
                                {ctx.labels.map((label) => (
                                    <LabelGuide key={label.id} label={label} />
                                ))}
                            </Stack>
                        </Box>
                    )}
                </QueryBoundary>
            </SectionCard>
            ) : null}
        </Stack>
    );
}
