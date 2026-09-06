import { useState } from "react";
import { Button, MenuItem, Stack, TextField, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { getRun, listDatasetSnapshots, runRobustnessSweep } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function RobustnessView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [snapshotId, setSnapshotId] = useState("");
    const [runId, setRunId] = useState<string | null>(null);
    const snapshotsQuery = useQuery({
        queryKey: queryKeys.textResearch.datasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listDatasetSnapshots(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });
    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: (query) => ["queued", "running"].includes(query.state.data?.status ?? "") ? 2000 : false,
    });
    const sweepMutation = useMutation({
        mutationFn: () => runRobustnessSweep({ snapshot_id: snapshotId }),
        onSuccess: (run) => {
            setRunId(run.id);
            showToast({ message: "Robustness sweep started.", severity: "success" });
        },
        onError: (error) => showToast({ message: getQueryErrorMessage(error, "Robustness sweep failed."), severity: "error" }),
    });

    return <Stack spacing={2}>
        <SectionCard title="Robustness testing" description="Evaluate the frozen training dataset across repeated grouped splits, cross-validation, preprocessing, class weights, organization, and temporal holdouts.">
            <QueryBoundary isLoading={snapshotsQuery.isLoading} isError={snapshotsQuery.isError} error={snapshotsQuery.error} onRetry={() => void snapshotsQuery.refetch()}>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                    <TextField select size="small" label="Frozen dataset snapshot" value={snapshotId} onChange={(event) => setSnapshotId(event.target.value)} sx={{ minWidth: 300 }}>
                        <MenuItem value="">Select snapshot</MenuItem>
                        {snapshotsQuery.data?.map((snapshot) => <MenuItem key={snapshot.id} value={snapshot.id}>{snapshot.name} · {snapshot.unit_type}</MenuItem>)}
                    </TextField>
                    <Button variant="contained" onClick={() => sweepMutation.mutate()} disabled={!snapshotId || sweepMutation.isPending}>Run sweep</Button>
                </Stack>
            </QueryBoundary>
        </SectionCard>
        {runId ? <SectionCard title="Robustness results" description="Poor or unstable results remain visible in the saved analysis run.">
            <QueryBoundary isLoading={runQuery.isLoading} isError={runQuery.isError} error={runQuery.error} onRetry={() => void runQuery.refetch()}>
                {runQuery.data ? <Stack spacing={1}><Typography>Run {runQuery.data.id} — <RunStatusChip status={runQuery.data.status} /></Typography>{runQuery.data.results ? <JsonBlock data={runQuery.data.results} /> : <Typography color="text.secondary">Computing controlled evaluations…</Typography>}</Stack> : null}
            </QueryBoundary>
        </SectionCard> : null}
    </Stack>;
}
