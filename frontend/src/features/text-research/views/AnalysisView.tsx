import { useState } from "react";
import { Button, Stack, Tab, Tabs, Typography } from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { getRun, runCorpusStats, runFrequencies } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

type AnalysisKind = "corpus-stats" | "frequencies";

export default function AnalysisView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [kind, setKind] = useState<AnalysisKind>("corpus-stats");
    const [runId, setRunId] = useState<string | null>(null);

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            return status === "running" || status === "pending" ? 2000 : false;
        },
    });

    const statsMutation = useMutation({
        mutationFn: () => runCorpusStats(ctx.selectedCorpusId, { unit_type: ctx.unitType }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Corpus stats run started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to run corpus stats."),
                severity: "error",
            }),
    });

    const freqMutation = useMutation({
        mutationFn: () => runFrequencies(ctx.selectedCorpusId, { unit_type: ctx.unitType, top_n: 50 }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Frequency analysis started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to run frequency analysis."),
                severity: "error",
            }),
    });

    const activeMutation = kind === "corpus-stats" ? statsMutation : freqMutation;

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Quantitative analysis"
                description="Run corpus statistics or term frequency analysis on segmented units."
            >
                <Tabs value={kind} onChange={(_, v: AnalysisKind) => setKind(v)} sx={{ mb: 2 }}>
                    <Tab label="Corpus stats" value="corpus-stats" />
                    <Tab label="Frequencies" value="frequencies" />
                </Tabs>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Unit type: {ctx.unitType}
                </Typography>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => activeMutation.mutate()}
                    disabled={!ctx.selectedCorpusId || activeMutation.isPending}
                >
                    Run {kind === "corpus-stats" ? "corpus stats" : "frequencies"}
                </Button>
            </SectionCard>

            {runId ? (
                <SectionCard title="Analysis output">
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2">
                                    {runQuery.data.run_type} — <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.results ? (
                                    <JsonBlock data={runQuery.data.results} />
                                ) : runQuery.data.metrics ? (
                                    <JsonBlock data={runQuery.data.metrics} />
                                ) : (
                                    <Typography color="text.secondary">
                                        Run in progress or no results yet.
                                    </Typography>
                                )}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
