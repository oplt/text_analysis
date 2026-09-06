import { useState } from "react";
import { Button, Stack, TextField, Typography } from "@mui/material";
import { PlayArrow as TrainIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { getRun, trainTopicModel } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function TopicsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [nTopics, setNTopics] = useState(5);
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

    const trainMutation = useMutation({
        mutationFn: () =>
            trainTopicModel(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                n_topics: nTopics,
                algorithm: "lda",
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Topic model training started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to train topic model."),
                severity: "error",
            }),
    });

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Topic modeling"
                description="Train an LDA topic model on segmented text units."
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center" sx={{ mb: 2 }}>
                    <TextField
                        size="small"
                        type="number"
                        label="Number of topics"
                        value={nTopics}
                        onChange={(e) => setNTopics(Number(e.target.value) || 5)}
                        inputProps={{ min: 2, max: 50 }}
                        sx={{ width: 180 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                        Unit type: {ctx.unitType}
                    </Typography>
                </Stack>
                <Button
                    variant="contained"
                    startIcon={<TrainIcon />}
                    onClick={() => trainMutation.mutate()}
                    disabled={!ctx.selectedCorpusId || trainMutation.isPending}
                >
                    Train topic model
                </Button>
            </SectionCard>

            {runId ? (
                <SectionCard title="Topic model results">
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
                                {runQuery.data.results ? (
                                    <JsonBlock data={runQuery.data.results} />
                                ) : runQuery.data.metrics ? (
                                    <JsonBlock data={runQuery.data.metrics} />
                                ) : (
                                    <Typography color="text.secondary">
                                        Training in progress…
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
