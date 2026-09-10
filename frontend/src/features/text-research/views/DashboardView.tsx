import {
    Box,
    Button,
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
    ModelTraining as ModelIcon,
    Science as SeedIcon,
    Verified as ReliabilityIcon,
} from "@mui/icons-material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { StatCard } from "../../../components/ui/StatCard";
import { EmptyState } from "../../../components/ui/EmptyState";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getDashboardSummary } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { NoCorpusEmptyState } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

const DETAIL_TABS = ["units", "runs", "reliability"] as const;
type DetailTab = (typeof DETAIL_TABS)[number];

const DETAIL_TAB_ITEMS: Array<{ value: DetailTab; label: string }> = [
    { value: "units", label: "Text units" },
    { value: "runs", label: "Analysis runs" },
    { value: "reliability", label: "Latest reliability" },
];

export default function DashboardView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: () => getDashboardSummary(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

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
            isLoading={dashboardQuery.isLoading}
            isError={dashboardQuery.isError}
            error={dashboardQuery.error}
            onRetry={() => void dashboardQuery.refetch()}
        >
            {dashboardQuery.data ? (
                <StackContent summary={dashboardQuery.data} projectId={ctx.projectId} />
            ) : null}
        </QueryBoundary>
    );
}

function StackContent({
    summary,
    projectId,
}: {
    summary: Awaited<ReturnType<typeof getDashboardSummary>>;
    projectId: string;
}) {
    const navigate = useNavigate();
    const [detailTab, setDetailTab] = useState<DetailTab>("units");
    const completionPct = Math.round(summary.annotation_completion_rate * 100);
    const hasUnits = Object.values(summary.text_unit_counts).some((count) => count > 0);

    return (
        <Box sx={{ display: "grid", gap: 2 }}>
            {!hasUnits && summary.document_count > 0 ? (
                <EmptyState
                    icon={<CorpusIcon fontSize="large" />}
                    title="Segment this corpus"
                    description={`${summary.document_count} documents are linked, but no text units exist yet.`}
                    action={
                        <Button
                            variant="contained"
                            onClick={() => navigate(`/research/${projectId}/corpus`)}
                        >
                            Prepare corpus
                        </Button>
                    }
                />
            ) : null}

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                }}
            >
                <StatCard
                    label="Documents"
                    value={summary.document_count}
                    description="Source documents in corpus"
                    icon={<CorpusIcon />}
                />
                <StatCard
                    label="Annotation tasks"
                    value={`${summary.annotation_completed_count}/${summary.annotation_task_count}`}
                    description={`${completionPct}% completion rate`}
                    icon={<AnnotationIcon />}
                    color="warning"
                />
                <StatCard
                    label="Trained models"
                    value={summary.trained_model_count}
                    description="Classifier models for this corpus"
                    icon={<ModelIcon />}
                    color="secondary"
                />
                <StatCard
                    label="Codebooks"
                    value={summary.codebook_count}
                    description="Project-level coding schemes"
                    icon={<ReliabilityIcon />}
                    color="success"
                />
            </Box>

            <Box>
                <PageTabs
                    value={detailTab}
                    onChange={setDetailTab}
                    tabs={DETAIL_TAB_ITEMS}
                    ariaLabel="Dashboard details"
                />

                {detailTab === "units" ? (
                    <SectionCard
                        title="Text units"
                        description="Segmentation counts by unit type."
                    >
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Unit type</TableCell>
                                    <TableCell align="right">Count</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {Object.entries(summary.text_unit_counts).map(([type, count]) => (
                                    <TableRow key={type}>
                                        <TableCell>{type}</TableCell>
                                        <TableCell align="right">{count}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </SectionCard>
                ) : null}

                {detailTab === "runs" ? (
                    <SectionCard
                        title="Analysis runs"
                        description="Run history grouped by type and status."
                    >
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            By type
                        </Typography>
                        <Table size="small" sx={{ mb: 2 }}>
                            <TableBody>
                                {Object.entries(summary.analysis_run_counts_by_type).map(
                                    ([type, count]) => (
                                        <TableRow key={type}>
                                            <TableCell>{type}</TableCell>
                                            <TableCell align="right">{count}</TableCell>
                                        </TableRow>
                                    )
                                )}
                            </TableBody>
                        </Table>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            By status
                        </Typography>
                        <Table size="small">
                            <TableBody>
                                {Object.entries(summary.analysis_run_counts_by_status).map(
                                    ([status, count]) => (
                                        <TableRow key={status}>
                                            <TableCell>{status}</TableCell>
                                            <TableCell align="right">{count}</TableCell>
                                        </TableRow>
                                    )
                                )}
                            </TableBody>
                        </Table>
                    </SectionCard>
                ) : null}

                {detailTab === "reliability" ? (
                    <SectionCard
                        title="Latest reliability"
                        description="Most recent completed reliability run."
                    >
                        {summary.latest_reliability ? (
                            <>
                                <Typography variant="body2" sx={{ mb: 1 }}>
                                    Run {summary.latest_reliability.run_id}
                                </Typography>
                                <Box
                                    component="pre"
                                    sx={{
                                        p: 1.5,
                                        borderRadius: 2,
                                        bgcolor: "action.hover",
                                        fontSize: 12,
                                        m: 0,
                                        whiteSpace: "pre-wrap",
                                        wordBreak: "break-word",
                                    }}
                                >
                                    {JSON.stringify(summary.latest_reliability.metrics, null, 2)}
                                </Box>
                            </>
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No reliability runs completed yet for this corpus.
                            </Typography>
                        )}
                    </SectionCard>
                ) : null}
            </Box>

            {summary.latest_model ? (
                <SectionCard
                    title="Latest model"
                    description="Most recently trained classifier."
                    action={
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={() => navigate(`/research/${projectId}/models`)}
                        >
                            Model Registry
                        </Button>
                    }
                >
                    <Typography variant="body2">
                        {summary.latest_model.name ?? summary.latest_model.id} (v
                        {summary.latest_model.version})
                    </Typography>
                    <Box
                        component="pre"
                        sx={{ mt: 1, p: 1.5, borderRadius: 2, bgcolor: "action.hover", fontSize: 12 }}
                    >
                        {JSON.stringify(summary.latest_model.metrics, null, 2)}
                    </Box>
                </SectionCard>
            ) : null}
        </Box>
    );
}
