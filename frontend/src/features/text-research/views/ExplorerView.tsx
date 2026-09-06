import { useState } from "react";
import {
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    TextField,
    Typography,
    Button,
} from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { getRun, runComparativePrevalence } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

const GROUP_BY_OPTIONS = [
    "organization",
    "organization_type",
    "publication_year",
    "country",
    "region",
    "cultural_sphere",
    "language",
    "publication_type",
];

export default function ExplorerView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [groupBy, setGroupBy] = useState("country");
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

    const prevalenceMutation = useMutation({
        mutationFn: () =>
            runComparativePrevalence(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: ctx.labels.map((l) => l.id),
                group_by: groupBy,
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Comparative prevalence analysis started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to run comparative analysis."),
                severity: "error",
            }),
    });

    const disabled =
        !ctx.selectedCorpusId || !ctx.selectedCodebookId || ctx.labels.length === 0;

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Comparative prevalence explorer"
                description="Compare label prevalence across metadata groups."
            >
                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                        <InputLabel id="group-by-label">Group by</InputLabel>
                        <Select
                            labelId="group-by-label"
                            label="Group by"
                            value={groupBy}
                            onChange={(e) => setGroupBy(e.target.value)}
                        >
                            {GROUP_BY_OPTIONS.map((option) => (
                                <MenuItem key={option} value={option}>
                                    {option.replace(/_/g, " ")}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <TextField
                        size="small"
                        label="Labels"
                        value={ctx.labels.map((l) => l.name).join(", ")}
                        InputProps={{ readOnly: true }}
                        sx={{ flex: 1 }}
                    />
                </Stack>
                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => prevalenceMutation.mutate()}
                    disabled={disabled || prevalenceMutation.isPending}
                >
                    Run prevalence analysis
                </Button>
            </SectionCard>

            {runId ? (
                <SectionCard title="Prevalence results">
                    <QueryBoundary
                        isLoading={runQuery.isLoading && !runQuery.data}
                        isError={runQuery.isError}
                        error={runQuery.error}
                        onRetry={() => void runQuery.refetch()}
                    >
                        {runQuery.data ? (
                            <Stack spacing={2}>
                                <Typography variant="body2">
                                    <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.results ? (
                                    <JsonBlock data={runQuery.data.results} />
                                ) : (
                                    <Typography color="text.secondary">Awaiting results…</Typography>
                                )}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
