import {
    Box,
    Button,
    Chip,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import {
    Assignment as AnnotationIcon,
    AutoStories as CorpusIcon,
    MenuBook as CodebookIcon,
    ModelTraining as ModelIcon,
    PlayArrow as RunIcon,
    Science as SeedIcon,
    Timeline as ProgressIcon,
    Verified as ReliabilityIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { StatCard } from "../../../components/ui/StatCard";
import { EmptyState } from "../../../components/ui/EmptyState";
import { FormGrid } from "../../../components/ui/FormGrid";
import { KeyValueList } from "../../../components/ui/KeyValueList";
import { MetricGrid } from "../../../components/ui/MetricGrid";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { layoutSpacing } from "../../../components/ui/layoutTokens";
import { getDashboardSummary } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { NoCorpusEmptyState } from "../components/NoCorpusEmptyState";
import { useResearchContext } from "../hooks/useResearchContext";
import { useResearchWorkflow } from "../hooks/useResearchWorkflow";
import {
    formatPercent,
    formatRunType,
    languageSummary,
    latestRunStatusLabel,
    pickReliabilityHighlights,
    totalTextUnits,
    workspaceHrefForRunType,
} from "../dashboardSummary";
import { STATUS_LABEL, stageActionLabel, workflowProgress } from "../workflowDisplayModel";
import type { DashboardSummary } from "../types";
import type { WorkflowStageState } from "../workflow";

export default function DashboardView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: ({ signal }) => getDashboardSummary(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });
    const { stages, isLoading: workflowLoading } = useResearchWorkflow("dashboard");

    if (!ctx.corporaLoading && ctx.corpora.length === 0) {
        return <NoCorpusEmptyState />;
    }

    if (!ctx.selectedCorpusId) {
        return (
            <SectionCard title="Dashboard">
                <EmptyState
                    icon={<CorpusIcon fontSize="large" />}
                    title="Select a corpus"
                    description="Choose an existing corpus in the workspace context bar, or create one to see KPIs."
                    action={
                        <Button
                            variant="contained"
                            startIcon={<SeedIcon />}
                            onClick={() => navigate(`/research/${ctx.projectId}/corpus`)}
                        >
                            Go to corpus
                        </Button>
                    }
                />
            </SectionCard>
        );
    }

    return (
        <QueryBoundary
            isLoading={dashboardQuery.isLoading || workflowLoading}
            isError={dashboardQuery.isError}
            error={dashboardQuery.error}
            onRetry={() => void dashboardQuery.refetch()}
        >
            {dashboardQuery.data ? (
                <DashboardContent
                    summary={dashboardQuery.data}
                    projectId={ctx.projectId}
                    stages={stages}
                />
            ) : null}
        </QueryBoundary>
    );
}

function DashboardContent({
    summary,
    projectId,
    stages,
}: {
    summary: DashboardSummary;
    projectId: string;
    stages: WorkflowStageState[];
}) {
    const navigate = useNavigate();
    const unitTotal = totalTextUnits(summary);
    const hasUnits = unitTotal > 0;
    const completionPct = Math.round(summary.annotation_completion_rate * 100);
    const runStatus = latestRunStatusLabel(summary);
    const progress = workflowProgress(stages);
    const blockers = stages.filter((stage) => stage.status === "blocked" || stage.status === "warning");
    const nextStage =
        stages.find((stage) => stage.status === "current" || stage.status === "incomplete") ??
        progress.current;
    const reliabilityHighlights = pickReliabilityHighlights(summary.latest_reliability?.metrics);
    const completeness = summary.metadata_completeness?.overall ?? null;
    const languageLine = languageSummary(summary.language_counts);
    const recentRuns = summary.recent_runs ?? [];

    return (
        <Box sx={{ display: "grid", gap: layoutSpacing.sectionGap }}>
            {!hasUnits && summary.document_count > 0 ? (
                <EmptyState
                    icon={<CorpusIcon fontSize="large" />}
                    title="Segment this corpus"
                    description={`${summary.document_count} documents are linked, but no text units exist yet.`}
                    action={
                        <Button
                            variant="contained"
                            onClick={() => navigate(`/research/${projectId}/prepare`)}
                        >
                            Open preparation
                        </Button>
                    }
                />
            ) : null}

            <MetricGrid columns={3}>
                <StatCard
                    label="Documents"
                    value={summary.document_count}
                    description="Source documents in corpus"
                    icon={<CorpusIcon />}
                />
                <StatCard
                    label="Text units"
                    value={unitTotal}
                    description={Object.entries(summary.text_unit_counts)
                        .map(([type, count]) => `${count} ${type}`)
                        .join(" · ")}
                    icon={<CorpusIcon />}
                    color="info"
                />
                <StatCard
                    label="Annotation progress"
                    value={`${summary.annotation_completed_count}/${summary.annotation_task_count}`}
                    description={`${completionPct}% completion rate`}
                    icon={<AnnotationIcon />}
                    color="warning"
                />
                <StatCard
                    label="Codebooks"
                    value={summary.codebook_count}
                    description="Project-level coding schemes"
                    icon={<CodebookIcon />}
                    color="success"
                />
                <StatCard
                    label="Trained models"
                    value={summary.trained_model_count}
                    description="Classifier models for this corpus"
                    icon={<ModelIcon />}
                    color="secondary"
                />
                <StatCard
                    label="Latest run status"
                    value={runStatus.value}
                    description={runStatus.description}
                    icon={<RunIcon />}
                    color={runStatus.color}
                />
            </MetricGrid>

            <FormGrid columns="6-6">
                <SectionCard
                    title="Research progress"
                    description="Current workflow stage and blockers for this corpus."
                    action={
                        nextStage ? (
                            <Button
                                size="small"
                                variant="contained"
                                onClick={() => navigate(`/research/${projectId}/${nextStage.route}`)}
                            >
                                {stageActionLabel(nextStage)}
                            </Button>
                        ) : null
                    }
                >
                    <Stack spacing={1.5}>
                        <Typography variant="body2" color="text.secondary">
                            {progress.completed}/{progress.total} stages complete ({progress.percent}%)
                        </Typography>
                        {progress.current ? (
                            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                                <Chip
                                    size="small"
                                    color={
                                        progress.current.status === "complete" ||
                                        progress.current.status === "current"
                                            ? "primary"
                                            : progress.current.status === "warning"
                                              ? "warning"
                                              : progress.current.status === "blocked"
                                                ? "default"
                                                : "default"
                                    }
                                    label={STATUS_LABEL[progress.current.status]}
                                />
                                <Typography variant="subtitle2">{progress.current.label}</Typography>
                                {progress.current.detail ? (
                                    <Typography variant="body2" color="text.secondary">
                                        {progress.current.detail}
                                    </Typography>
                                ) : null}
                            </Stack>
                        ) : null}
                        {blockers.length > 0 ? (
                            <Stack spacing={0.75}>
                                <Typography variant="subtitle2">Blockers</Typography>
                                {blockers.slice(0, 4).map((stage) => (
                                    <Stack
                                        key={stage.id}
                                        direction={{ xs: "column", sm: "row" }}
                                        spacing={1}
                                        alignItems={{ sm: "center" }}
                                        justifyContent="space-between"
                                    >
                                        <Typography variant="body2">
                                            {stage.label}
                                            {stage.blockedReason ? ` — ${stage.blockedReason}` : ""}
                                            {!stage.blockedReason && stage.detail ? ` — ${stage.detail}` : ""}
                                        </Typography>
                                        <Button
                                            size="small"
                                            variant="text"
                                            onClick={() =>
                                                navigate(`/research/${projectId}/${stage.route}`)
                                            }
                                        >
                                            Open
                                        </Button>
                                    </Stack>
                                ))}
                            </Stack>
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No blockers right now. Continue from the current stage.
                            </Typography>
                        )}
                    </Stack>
                </SectionCard>

                <SectionCard
                    title="Corpus summary"
                    description="Unit mix, languages, and metadata completeness (KPI totals are above)."
                    action={
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={() => navigate(`/research/${projectId}/corpus`)}
                        >
                            Open corpus
                        </Button>
                    }
                >
                    <KeyValueList
                        dense
                        items={[
                            {
                                key: "unit_breakdown",
                                label: "By unit type",
                                value: Object.entries(summary.text_unit_counts)
                                    .map(([type, count]) => `${type}: ${count}`)
                                    .join(" · ") || "—",
                            },
                            { key: "languages", label: "Languages", value: languageLine },
                            {
                                key: "metadata",
                                label: "Metadata completeness",
                                value:
                                    completeness == null
                                        ? "—"
                                        : formatPercent(completeness),
                                description:
                                    completeness == null
                                        ? undefined
                                        : "Average fill rate for title, organization, year, language, and country",
                            },
                        ]}
                    />
                </SectionCard>
            </FormGrid>

            <FormGrid columns="6-6">
                <SectionCard
                    title="Coding progress"
                    description="Annotation completion and latest agreement signals."
                    action={
                        <Stack direction="row" spacing={1}>
                            <Button
                                size="small"
                                variant="outlined"
                                onClick={() => navigate(`/research/${projectId}/annotation`)}
                            >
                                Annotation
                            </Button>
                            <Button
                                size="small"
                                variant="outlined"
                                onClick={() =>
                                    navigate(`/research/${projectId}/reliability?tab=disagreements`)
                                }
                            >
                                Disagreements
                            </Button>
                        </Stack>
                    }
                >
                    <Stack spacing={2}>
                        <KeyValueList
                            dense
                            items={[
                                {
                                    key: "annotation",
                                    label: "Annotation",
                                    value: `${summary.annotation_completed_count}/${summary.annotation_task_count} (${completionPct}%)`,
                                },
                                {
                                    key: "codebooks",
                                    label: "Codebooks",
                                    value: String(summary.codebook_count),
                                },
                            ]}
                        />
                        {reliabilityHighlights.length > 0 ? (
                            <Box>
                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                    Latest agreement
                                </Typography>
                                <KeyValueList
                                    dense
                                    items={reliabilityHighlights.map((item) => ({
                                        key: item.label,
                                        label: item.label,
                                        value: item.value,
                                    }))}
                                />
                                <Button
                                    size="small"
                                    sx={{ mt: 1 }}
                                    onClick={() =>
                                        navigate(`/research/${projectId}/reliability?tab=agreement`)
                                    }
                                >
                                    Open reliability
                                </Button>
                            </Box>
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No completed reliability run yet. Compute agreement after multi-coder
                                annotation.
                            </Typography>
                        )}
                    </Stack>
                </SectionCard>

                <SectionCard
                    title="Model summary"
                    description="Latest trained classifier for this corpus."
                    action={
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={() => navigate(`/research/${projectId}/models`)}
                        >
                            Model registry
                        </Button>
                    }
                >
                    {summary.latest_model ? (
                        <KeyValueList
                            dense
                            items={[
                                {
                                    key: "name",
                                    label: "Model",
                                    value:
                                        summary.latest_model.name ??
                                        summary.latest_model.id.slice(0, 8),
                                },
                                {
                                    key: "version",
                                    label: "Version",
                                    value: `v${summary.latest_model.version}`,
                                },
                                {
                                    key: "macro_f1",
                                    label: "Macro F1",
                                    value:
                                        summary.latest_model.macro_f1 != null
                                            ? Number(summary.latest_model.macro_f1).toFixed(3)
                                            : "—",
                                },
                                {
                                    key: "lifecycle",
                                    label: "Lifecycle",
                                    value: summary.latest_model.lifecycle_status ?? "candidate",
                                },
                            ]}
                        />
                    ) : (
                        <EmptyState
                            icon={<ModelIcon fontSize="large" />}
                            title="No trained models"
                            description="Train a classifier after you have labeled units and a frozen dataset snapshot."
                            action={
                                <Button
                                    variant="contained"
                                    onClick={() => navigate(`/research/${projectId}/classification`)}
                                >
                                    Open classification
                                </Button>
                            }
                        />
                    )}
                </SectionCard>
            </FormGrid>

            <SectionCard
                title="Recent activity"
                description="Latest analysis, segmentation, training, and export runs."
                action={
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<ProgressIcon />}
                        onClick={() => navigate(`/research/${projectId}/runs`)}
                    >
                        All runs
                    </Button>
                }
            >
                {recentRuns.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                        No runs yet. Segmentation, analysis, training, and exports will appear here.
                    </Typography>
                ) : (
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Type</TableCell>
                                <TableCell>Status</TableCell>
                                <TableCell>When</TableCell>
                                <TableCell align="right">Workspace</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {recentRuns.map((run) => (
                                <TableRow key={run.id} hover>
                                    <TableCell>{formatRunType(run.run_type)}</TableCell>
                                    <TableCell>
                                        <Chip size="small" label={run.status} variant="outlined" />
                                    </TableCell>
                                    <TableCell>
                                        {run.created_at
                                            ? new Date(run.created_at).toLocaleString()
                                            : "—"}
                                    </TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            onClick={() =>
                                                navigate(
                                                    workspaceHrefForRunType(projectId, run.run_type)
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
                )}
            </SectionCard>

            <SectionCard
                title="Quick links"
                description="Jump into the workspaces that usually follow from this overview."
            >
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<CorpusIcon />}
                        onClick={() => navigate(`/research/${projectId}/corpus`)}
                    >
                        Corpus
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${projectId}/prepare`)}
                    >
                        Prepare
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<AnnotationIcon />}
                        onClick={() => navigate(`/research/${projectId}/annotation`)}
                    >
                        Annotation
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<ReliabilityIcon />}
                        onClick={() => navigate(`/research/${projectId}/reliability`)}
                    >
                        Reliability
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${projectId}/analysis`)}
                    >
                        Analysis
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        startIcon={<ModelIcon />}
                        onClick={() => navigate(`/research/${projectId}/classification`)}
                    >
                        Classification
                    </Button>
                    <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/research/${projectId}/exports`)}
                    >
                        Exports
                    </Button>
                </Stack>
            </SectionCard>
        </Box>
    );
}
