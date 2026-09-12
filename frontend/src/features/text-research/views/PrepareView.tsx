import { useEffect, useMemo, useRef, useState } from "react";
import {
    Alert,
    Box,
    Button,
    FormControl,
    InputLabel,
    LinearProgress,
    MenuItem,
    Select,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import {
    PlayArrow as StartIcon,
    Science as PrepareIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getDashboardSummary,
    getRun,
    listRuns,
    segmentCorpus,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { NoCorpusEmptyState, RunStatusChip } from "../components/ResearchShared";
import { CleaningPanel } from "../components/CleaningPanel";
import { IngestionQaPanel } from "../components/IngestionQaPanel";
import { PreprocessingPanel } from "../components/PreprocessingPanel";
import { useResearchContext } from "../hooks/useResearchContext";
import {
    isSegmentationRunActive,
    segmentationProgressLines,
    segmentationRunStorageKey,
    segmentationStageLabel,
} from "../segmentationProgress";
import { UNIT_TYPE_OPTIONS, type UnitType } from "../types";

const PREPARE_TABS = ["segment", "ingestion", "cleaning", "preprocessing"] as const;
type PrepareTab = (typeof PREPARE_TABS)[number];

const PREPARE_TAB_ITEMS: Array<{ value: PrepareTab; label: string }> = [
    { value: "segment", label: "Segment" },
    { value: "ingestion", label: "Ingestion QA" },
    { value: "cleaning", label: "Cleaning" },
    { value: "preprocessing", label: "Preprocessing" },
];

function readStoredSegmentationRunId(corpusId: string | null): string | null {
    if (!corpusId) return null;
    return localStorage.getItem(segmentationRunStorageKey(corpusId));
}

function SegmentUnitControls({
    unitType,
    documentCount,
    active,
    onSegment,
    onManageDocuments,
}: {
    unitType: UnitType;
    documentCount: number;
    active: boolean;
    onSegment: (unitType: UnitType) => void;
    onManageDocuments: () => void;
}) {
    const [localUnitType, setLocalUnitType] = useState(unitType);

    return (
        <>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ sm: "center" }}>
                <FormControl size="small" sx={{ minWidth: 180 }}>
                    <InputLabel id="prepare-unit-type-label">Unit type</InputLabel>
                    <Select
                        labelId="prepare-unit-type-label"
                        label="Unit type"
                        value={localUnitType}
                        onChange={(e) => setLocalUnitType(e.target.value as UnitType)}
                        disabled={active}
                    >
                        {UNIT_TYPE_OPTIONS.map((option) => (
                            <MenuItem key={option.value} value={option.value}>
                                {option.label}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <Button
                    variant="contained"
                    startIcon={<StartIcon />}
                    onClick={() => onSegment(localUnitType)}
                    disabled={documentCount === 0 || active}
                >
                    {active ? "Segmentation in progress…" : "Start segmentation"}
                </Button>
                <Button variant="outlined" onClick={onManageDocuments}>
                    Manage documents
                </Button>
            </Stack>

            {documentCount === 0 ? (
                <Alert severity="info">
                    Upload or link documents in the Corpus stage before preparing text units.
                </Alert>
            ) : null}
        </>
    );
}

export default function PrepareView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const [tab, setTab] = useTabQueryParam(PREPARE_TABS, "segment");
    const [runSelection, setRunSelection] = useState<{ corpusId: string; runId: string | null } | null>(
        null
    );
    const lastRefreshedStatus = useRef<string | null>(null);

    const runId = useMemo(() => {
        if (!ctx.selectedCorpusId) return null;
        if (runSelection?.corpusId === ctx.selectedCorpusId) {
            return runSelection.runId;
        }
        return readStoredSegmentationRunId(ctx.selectedCorpusId);
    }, [ctx.selectedCorpusId, runSelection]);

    const setRunId = (nextRunId: string | null) => {
        if (!ctx.selectedCorpusId) return;
        setRunSelection({ corpusId: ctx.selectedCorpusId, runId: nextRunId });
        if (nextRunId) {
            localStorage.setItem(segmentationRunStorageKey(ctx.selectedCorpusId), nextRunId);
        } else {
            localStorage.removeItem(segmentationRunStorageKey(ctx.selectedCorpusId));
        }
    };

    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: ({ signal }) => getDashboardSummary(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: ({ signal }) => getRun(runId!, signal),
        enabled: Boolean(runId),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === "queued" || status === "pending" || status === "running"
                ? 1500
                : false;
        },
    });

    const recentRunsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(
            ctx.projectId,
            ctx.selectedCorpusId,
            "segmentation"
        ),
        queryFn: ({ signal }) => listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId,
                run_type: "segmentation",
                limit: 5,
                offset: 0,
            }, signal),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });

    useEffect(() => {
        const run = runQuery.data;
        if (!run || !ctx.selectedCorpusId) return;
        if (run.status !== "completed" && run.status !== "failed") return;
        if (lastRefreshedStatus.current === `${run.id}:${run.status}`) return;
        lastRefreshedStatus.current = `${run.id}:${run.status}`;

        void client.invalidateQueries({
            queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        });
        void client.invalidateQueries({
            queryKey: ["text-research", "corpus", ctx.selectedCorpusId, "documents"],
        });
        void client.invalidateQueries({
            queryKey: queryKeys.textResearch.runs(
                ctx.projectId,
                ctx.selectedCorpusId,
                "segmentation"
            ),
        });
    }, [runQuery.data, ctx.selectedCorpusId, ctx.projectId, client]);

    const segmentMutation = useMutation({
        mutationFn: (unitType: UnitType) => {
            ctx.setUnitType(unitType);
            return segmentCorpus(ctx.selectedCorpusId, unitType);
        },
        onSuccess: (run) => {
            setRunId(run.id);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.runs(
                    ctx.projectId,
                    ctx.selectedCorpusId,
                    "segmentation"
                ),
            });
            showToast({
                message:
                    run.status === "completed"
                        ? "Segmentation completed."
                        : "Segmentation started. Progress updates automatically.",
                severity: "success",
            });
            if (run.status === "completed") {
                void client.invalidateQueries({
                    queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
                });
            }
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to start segmentation."),
                severity: "error",
            }),
    });

    if (!ctx.corporaLoading && ctx.corpora.length === 0) {
        return (
            <Stack spacing={2}>
                <PageTabs
                    value={tab}
                    onChange={setTab}
                    tabs={PREPARE_TAB_ITEMS}
                    ariaLabel="Prepare workflow"
                />
                {tab === "preprocessing" ? <PreprocessingPanel /> : tab === "cleaning" ? <CleaningPanel /> : <NoCorpusEmptyState />}
            </Stack>
        );
    }

    if (!ctx.selectedCorpusId) {
        return (
            <Stack spacing={2}>
                <PageTabs
                    value={tab}
                    onChange={setTab}
                    tabs={PREPARE_TAB_ITEMS}
                    ariaLabel="Prepare workflow"
                />
                {tab === "segment" ? (
                    <SectionCard title="Prepare corpus">
                        <EmptyState
                            icon={<PrepareIcon fontSize="large" />}
                            title="Select a corpus"
                            description="Choose a corpus in the workspace context bar, then segment documents into research units."
                            action={
                                <Button
                                    variant="contained"
                                    onClick={() => navigate(`/research/${ctx.projectId}/corpus`)}
                                >
                                    Go to corpus
                                </Button>
                            }
                        />
                    </SectionCard>
                ) : tab === "preprocessing" ? <PreprocessingPanel /> : tab === "cleaning" ? <CleaningPanel /> : <NoCorpusEmptyState />}
            </Stack>
        );
    }

    const documentCount = dashboardQuery.data?.document_count ?? 0;
    const unitCount = dashboardQuery.data?.text_unit_counts?.[ctx.unitType] ?? 0;
    const active = isSegmentationRunActive(runQuery.data) || segmentMutation.isPending;
    const progressLines = segmentationProgressLines(runQuery.data);
    const metrics = (runQuery.data?.metrics ?? {}) as Record<string, unknown>;
    const documentsTotal = Number(metrics.documents_total ?? 0);
    const documentsDone = Number(metrics.documents_segmented ?? 0);
    const progressPct =
        documentsTotal > 0 ? Math.min(100, Math.round((documentsDone / documentsTotal) * 100)) : active ? 5 : 0;

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={PREPARE_TAB_ITEMS}
                ariaLabel="Prepare workflow"
            />

            {tab === "segment" ? (
            <>
            <SectionCard
                title="Preparation workspace"
                description="Segment corpus documents into reproducible research text units. Long jobs are queued and polled automatically."
            >
                <Stack spacing={2}>
                    <Typography variant="body2" color="text.secondary">
                        Corpus: <strong>{ctx.selectedCorpus?.name}</strong>
                        {" · "}
                        {documentCount.toLocaleString()} document{documentCount === 1 ? "" : "s"}
                        {" · "}
                        {unitCount.toLocaleString()} existing {ctx.unitType} unit
                        {unitCount === 1 ? "" : "s"}
                    </Typography>

                    <SegmentUnitControls
                        key={`${ctx.selectedCorpusId}:${ctx.unitType}`}
                        unitType={ctx.unitType}
                        documentCount={documentCount}
                        active={active}
                        onSegment={(unitType) => segmentMutation.mutate(unitType)}
                        onManageDocuments={() => navigate(`/research/${ctx.projectId}/corpus`)}
                    />
                </Stack>
            </SectionCard>

            <SectionCard title="Run status" description="Live progress for the active or latest segmentation job.">
                {!runId ? (
                    <Typography variant="body2" color="text.secondary">
                        No segmentation run selected yet. Start segmentation to track progress here.
                    </Typography>
                ) : (
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                        variant="inline"
                    >
                        {runQuery.data ? (
                            <Stack spacing={1.5}>
                                <Typography variant="body2">
                                    Run {runQuery.data.id} — <RunStatusChip status={runQuery.data.status} />
                                    {" · "}
                                    {segmentationStageLabel(runQuery.data.progress_stage)}
                                </Typography>
                                {active ? <LinearProgress variant="determinate" value={progressPct} /> : null}
                                <Box component="ul" sx={{ m: 0, pl: 2 }}>
                                    {progressLines.map((line) => (
                                        <Typography key={line} component="li" variant="body2">
                                            {line}
                                        </Typography>
                                    ))}
                                </Box>
                                {runQuery.data.error_message ? (
                                    <Alert severity="error">{runQuery.data.error_message}</Alert>
                                ) : null}
                                {runQuery.data.status === "completed" ? (
                                    <Alert severity="success">
                                        Segmentation finished. Dashboard and corpus unit counts have been refreshed.
                                    </Alert>
                                ) : null}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                )}
            </SectionCard>

            <SectionCard title="Recent segmentation runs" description="Latest preparation jobs for this corpus.">
                <QueryBoundary
                    isLoading={recentRunsQuery.isLoading}
                    isError={recentRunsQuery.isError}
                    error={recentRunsQuery.error}
                    onRetry={() => void recentRunsQuery.refetch()}
                    variant="inline"
                >
                    {recentRunsQuery.data?.items.length ? (
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Created</TableCell>
                                    <TableCell>Status</TableCell>
                                    <TableCell>Stage</TableCell>
                                    <TableCell>Units</TableCell>
                                    <TableCell />
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {recentRunsQuery.data.items.map((run) => {
                                    const runMetrics = (run.metrics ?? {}) as Record<string, unknown>;
                                    return (
                                        <TableRow key={run.id} selected={run.id === runId}>
                                            <TableCell>
                                                {new Date(run.created_at).toLocaleString()}
                                            </TableCell>
                                            <TableCell>
                                                <RunStatusChip status={run.status} />
                                            </TableCell>
                                            <TableCell>
                                                {segmentationStageLabel(run.progress_stage)}
                                            </TableCell>
                                            <TableCell>
                                                {Number(runMetrics.text_units_created ?? 0).toLocaleString()}
                                            </TableCell>
                                            <TableCell align="right">
                                                <Button
                                                    size="small"
                                                    onClick={() => setRunId(run.id)}
                                                >
                                                    Inspect
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    ) : (
                        <Typography variant="body2" color="text.secondary">
                            No segmentation runs for this corpus yet.
                        </Typography>
                    )}
                </QueryBoundary>
            </SectionCard>
            </>
            ) : null}

            {tab === "ingestion" ? <IngestionQaPanel /> : null}
            {tab === "cleaning" ? <CleaningPanel /> : null}
            {tab === "preprocessing" ? <PreprocessingPanel /> : null}
        </Stack>
    );
}
