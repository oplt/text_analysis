import { useState } from "react";
import { Alert, Button, Stack, Typography } from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { computeReliability, getRun } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function ReliabilityView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
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

    const reliabilityMutation = useMutation({
        mutationFn: () =>
            computeReliability(ctx.selectedCorpusId, {
                codebook_id: ctx.selectedCodebookId,
                label_ids: ctx.labels.map((l) => l.id),
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Reliability computation started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to compute reliability."),
                severity: "error",
            }),
    });

    const disabled =
        !ctx.selectedCorpusId || !ctx.selectedCodebookId || ctx.labels.length === 0;

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Inter-annotator reliability"
                description="Compute agreement metrics for the selected codebook labels."
            >
                {disabled ? (
                    <Alert severity="info" sx={{ mb: 2 }}>
                        Select a corpus, codebook, and ensure labels exist before running reliability.
                    </Alert>
                ) : null}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Labels included: {ctx.labels.map((l) => l.name).join(", ") || "none"}
                </Typography>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => reliabilityMutation.mutate()}
                    disabled={disabled || reliabilityMutation.isPending}
                >
                    Compute reliability
                </Button>
            </SectionCard>

            {(runId || reliabilityMutation.data) && (
                <SectionCard title="Run results" description="Metrics and results from the reliability run.">
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2">
                                    Run {runQuery.data.id} — <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.error_message ? (
                                    <Alert severity="error">{runQuery.data.error_message}</Alert>
                                ) : null}
                                {runQuery.data.metrics ? (
                                    <>
                                        <Typography variant="subtitle2">Metrics</Typography>
                                        <JsonBlock data={runQuery.data.metrics} />
                                    </>
                                ) : null}
                                {runQuery.data.results ? (
                                    <>
                                        <Typography variant="subtitle2">Results</Typography>
                                        <JsonBlock data={runQuery.data.results} />
                                    </>
                                ) : null}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            )}
        </Stack>
    );
}
