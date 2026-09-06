import {
    Box,
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
    Verified as ReliabilityIcon,
} from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { StatCard } from "../../../components/ui/StatCard";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getDashboardSummary } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { useResearchContext } from "../hooks/useResearchContext";

export default function DashboardView() {
    const ctx = useResearchContext();
    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: () => getDashboardSummary(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    if (!ctx.selectedCorpusId) {
        return (
            <SectionCard title="Dashboard">
                <Typography color="text.secondary">
                    Select or create a corpus to view dashboard KPIs.
                </Typography>
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
                <StackContent summary={dashboardQuery.data} />
            ) : null}
        </QueryBoundary>
    );
}

function StackContent({ summary }: { summary: Awaited<ReturnType<typeof getDashboardSummary>> }) {
    const completionPct = Math.round(summary.annotation_completion_rate * 100);
    return (
        <Box sx={{ display: "grid", gap: 2 }}>
            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", xl: "repeat(4, 1fr)" },
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

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                }}
            >
                <SectionCard title="Text units" description="Segmentation counts by unit type.">
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

                <SectionCard title="Analysis runs" description="Run history grouped by type and status.">
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        By type
                    </Typography>
                    <Table size="small" sx={{ mb: 2 }}>
                        <TableBody>
                            {Object.entries(summary.analysis_run_counts_by_type).map(([type, count]) => (
                                <TableRow key={type}>
                                    <TableCell>{type}</TableCell>
                                    <TableCell align="right">{count}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        By status
                    </Typography>
                    <Table size="small">
                        <TableBody>
                            {Object.entries(summary.analysis_run_counts_by_status).map(([status, count]) => (
                                <TableRow key={status}>
                                    <TableCell>{status}</TableCell>
                                    <TableCell align="right">{count}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </SectionCard>
            </Box>

            {summary.latest_model ? (
                <SectionCard title="Latest model" description="Most recently trained classifier.">
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

            {summary.latest_reliability ? (
                <SectionCard title="Latest reliability" description="Most recent completed reliability run.">
                    <Typography variant="body2" sx={{ mb: 1 }}>
                        Run {summary.latest_reliability.run_id}
                    </Typography>
                    <Box
                        component="pre"
                        sx={{ p: 1.5, borderRadius: 2, bgcolor: "action.hover", fontSize: 12 }}
                    >
                        {JSON.stringify(summary.latest_reliability.metrics, null, 2)}
                    </Box>
                </SectionCard>
            ) : null}
        </Box>
    );
}
