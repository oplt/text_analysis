import { Button, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { getRun, listRuns } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { JsonBlock, RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { useState } from "react";
import ExportsView from "./ExportsView";

export default function RunsView() {
    const ctx = useResearchContext();
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const runsQuery = useQuery({ queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId), queryFn: () => listRuns(ctx.projectId, { corpus_id: ctx.selectedCorpusId }), enabled: Boolean(ctx.projectId) });
    const runQuery = useQuery({ queryKey: queryKeys.textResearch.run(selectedRunId ?? ""), queryFn: () => getRun(selectedRunId!), enabled: Boolean(selectedRunId) });
    return <Stack spacing={2}>
        <SectionCard title="Runs & provenance" description="Persisted research operations, parameters, results, metrics, and artifacts.">
            <QueryBoundary isLoading={runsQuery.isLoading} isError={runsQuery.isError} error={runsQuery.error} onRetry={() => void runsQuery.refetch()}>
                {runsQuery.data?.items.length ? <Table size="small"><TableHead><TableRow><TableCell>Type</TableCell><TableCell>Status</TableCell><TableCell>Created</TableCell><TableCell /></TableRow></TableHead><TableBody>{runsQuery.data.items.map((run) => <TableRow key={run.id}><TableCell>{run.run_type}</TableCell><TableCell><RunStatusChip status={run.status} /></TableCell><TableCell>{new Date(run.created_at).toLocaleString()}</TableCell><TableCell><Button size="small" onClick={() => setSelectedRunId(run.id)}>Inspect</Button></TableCell></TableRow>)}</TableBody></Table> : <Typography color="text.secondary">No persisted runs for this corpus yet.</Typography>}
            </QueryBoundary>
        </SectionCard>
        {selectedRunId ? <SectionCard title="Run detail"><QueryBoundary isLoading={runQuery.isLoading} isError={runQuery.isError} error={runQuery.error} onRetry={() => void runQuery.refetch()}>{runQuery.data ? <Stack spacing={1}><Typography>{runQuery.data.run_type} — <RunStatusChip status={runQuery.data.status} /></Typography><Typography variant="subtitle2">Parameters</Typography><JsonBlock data={runQuery.data.parameters ?? {}} /><Typography variant="subtitle2">Metrics & results</Typography><JsonBlock data={{ metrics: runQuery.data.metrics, results: runQuery.data.results, error: runQuery.data.error_message }} /></Stack> : null}</QueryBoundary></SectionCard> : null}
        <ExportsView />
    </Stack>;
}
