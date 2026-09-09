import { useState } from "react";
import type { ReactNode } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
    getRun,
    runClustering,
    runDimensionalityReduction,
    runDuplicateDetection,
    runReadability,
    runSimilarity,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import type { AnalysisRun, UnitType } from "../types";
import { ResultsInspector } from "./ResearchCharts";
import { ResearchResultsTable } from "./ResearchResults";
import { RunStatusChip } from "./ResearchShared";

type BasePayload = { unit_type: UnitType; preprocessing_profile_id?: string; [key: string]: unknown };

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? value as Record<string, unknown>
        : null;
}

function ResultOutput({ run }: { run: AnalysisRun | undefined }) {
    if (!run) return null;
    const results = asRecord(run.results);
    const firstArray = Object.values(results ?? {}).find((value) => Array.isArray(value)) as unknown[] | undefined;
    const rows = (firstArray ?? [])
        .map(asRecord)
        .filter((row): row is Record<string, unknown> => Boolean(row))
        .map((row, index) => ({ ...row, id: index }));
    const columns = rows.length
        ? Object.keys(rows[0]).filter((key) => key !== "id").slice(0, 7).map((key) => ({
            id: key,
            label: key.replaceAll("_", " "),
            value: (row: Record<string, unknown>) => {
                const value = row[key];
                return typeof value === "object" ? JSON.stringify(value) : String(value ?? "");
            },
        }))
        : [];
    return (
        <Stack spacing={1.5}>
            <Typography variant="body2">Run <RunStatusChip status={run.status} /> {run.progress_stage ? ` · ${run.progress_stage}` : ""}</Typography>
            {run.error_message ? <Alert severity="error">{run.error_message}</Alert> : null}
            {isActiveRunStatus(run.status) ? <Typography color="text.secondary">Analysis in progress…</Typography> : null}
            {rows.length ? <ResearchResultsTable rows={rows} columns={columns} /> : null}
            {!isActiveRunStatus(run.status) ? <ResultsInspector data={{ metrics: run.metrics, results: run.results }} /> : null}
        </Stack>
    );
}

function useAnalysisRun() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [runId, setRunId] = useState<string | null>(null);
    const sseConnected = useRunEvents(runId, ctx.projectId);
    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });
    return { ctx, run: runQuery.data, runQuery, setRunId, showToast };
}

function useRunMutation(
    action: () => Promise<AnalysisRun>,
    label: string,
    setRunId: (runId: string) => void,
    showToast: ReturnType<typeof useSnackbar>["showToast"]
) {
    return useMutation({
        mutationFn: action,
        onSuccess: (run) => { setRunId(run.id); showToast({ message: `${label} started.`, severity: "success" }); },
        onError: (error) => showToast({ message: getQueryErrorMessage(error, `${label} failed.`), severity: "error" }),
    });
}

function PanelBody({ children, run, runQuery }: { children: ReactNode; run?: AnalysisRun; runQuery: ReturnType<typeof useQuery<AnalysisRun>> }) {
    return <Stack spacing={2}>{children}<QueryBoundary isLoading={runQuery.isLoading && !run} isError={runQuery.isError} error={runQuery.error} onRetry={() => void runQuery.refetch()} variant="inline"><ResultOutput run={run} /></QueryBoundary></Stack>;
}

export function SimilarityExplorer({ basePayload }: { basePayload: BasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [method, setMethod] = useState<"tfidf_cosine" | "jaccard">("tfidf_cosine");
    const [mode, setMode] = useState<"pairwise" | "query" | "group_centroid">("pairwise");
    const [queryText, setQueryText] = useState("");
    const [groupBy, setGroupBy] = useState("organization");
    const [topK, setTopK] = useState(20);
    const mutation = useRunMutation(() => runSimilarity(ctx.selectedCorpusId, { ...basePayload, method, mode, top_k: topK, ...(mode === "query" ? { query_text: queryText.trim() } : {}), ...(mode === "group_centroid" ? { group_by: groupBy.trim() } : {}) }), "Similarity analysis", setRunId, showToast);
    return <SectionCard title="Similarity explorer" description="Compare text units pairwise, against a query, or between metadata-group centroids."><PanelBody run={run} runQuery={runQuery}><Stack direction={{ xs: "column", sm: "row" }} spacing={1}><TextField select size="small" label="Method" value={method} onChange={(event) => setMethod(event.target.value as typeof method)}><MenuItem value="tfidf_cosine">TF-IDF cosine</MenuItem><MenuItem value="jaccard">Jaccard</MenuItem></TextField><TextField select size="small" label="Mode" value={mode} onChange={(event) => setMode(event.target.value as typeof mode)}><MenuItem value="pairwise">Pairwise</MenuItem><MenuItem value="query">Query</MenuItem><MenuItem value="group_centroid">Group centroid</MenuItem></TextField><TextField size="small" type="number" label="Top results" value={topK} onChange={(event) => setTopK(Math.max(1, Number(event.target.value) || 1))} /></Stack>{mode === "query" ? <TextField size="small" label="Query text" value={queryText} onChange={(event) => setQueryText(event.target.value)} /> : null}{mode === "group_centroid" ? <TextField size="small" label="Metadata field" value={groupBy} onChange={(event) => setGroupBy(event.target.value)} /> : null}<Button variant="contained" startIcon={<RunIcon />} onClick={() => mutation.mutate()} disabled={mutation.isPending || (mode === "query" && !queryText.trim()) || (mode === "group_centroid" && !groupBy.trim())} sx={{ alignSelf: "flex-start" }}>Run similarity</Button></PanelBody></SectionCard>;
}

export function DuplicateDetectionView({ basePayload }: { basePayload: BasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [lexical, setLexical] = useState(true);
    const [threshold, setThreshold] = useState(0.85);
    const mutation = useRunMutation(() => runDuplicateDetection(ctx.selectedCorpusId, { ...basePayload, methods: lexical ? ["exact", "normalized", "lexical"] : ["exact", "normalized"], lexical_threshold: threshold }), "Duplicate detection", setRunId, showToast);
    return <SectionCard title="Duplicate detection" description="Find exact, normalized, and optionally lexical near-duplicates among selected units."><PanelBody run={run} runQuery={runQuery}><Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}><FormControlLabel control={<Checkbox checked={lexical} onChange={(event) => setLexical(event.target.checked)} />} label="Include lexical matches" /><TextField size="small" type="number" label="Lexical threshold" value={threshold} onChange={(event) => setThreshold(Math.max(0, Math.min(1, Number(event.target.value))))} inputProps={{ min: 0, max: 1, step: 0.05 }} /></Stack><Button variant="contained" startIcon={<RunIcon />} onClick={() => mutation.mutate()} disabled={mutation.isPending} sx={{ alignSelf: "flex-start" }}>Find duplicates</Button></PanelBody></SectionCard>;
}

export function ClusterExplorer({ basePayload }: { basePayload: BasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [clusters, setClusters] = useState(5);
    const [algorithm, setAlgorithm] = useState<"kmeans" | "minibatch_kmeans">("kmeans");
    const [useSvd, setUseSvd] = useState(false);
    const mutation = useRunMutation(() => runClustering(ctx.selectedCorpusId, { ...basePayload, n_clusters: clusters, algorithm, use_svd: useSvd }), "Clustering", setRunId, showToast);
    return <SectionCard title="Cluster explorer" description="Group units using TF-IDF features, then inspect cluster assignments and top terms."><PanelBody run={run} runQuery={runQuery}><Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}><TextField size="small" type="number" label="Clusters" value={clusters} onChange={(event) => setClusters(Math.max(2, Number(event.target.value) || 2))} inputProps={{ min: 2 }} /><TextField select size="small" label="Algorithm" value={algorithm} onChange={(event) => setAlgorithm(event.target.value as typeof algorithm)}><MenuItem value="kmeans">K-means</MenuItem><MenuItem value="minibatch_kmeans">MiniBatch K-means</MenuItem></TextField><FormControlLabel control={<Checkbox checked={useSvd} onChange={(event) => setUseSvd(event.target.checked)} />} label="Reduce dimensions first" /></Stack><Button variant="contained" startIcon={<RunIcon />} onClick={() => mutation.mutate()} disabled={mutation.isPending} sx={{ alignSelf: "flex-start" }}>Run clustering</Button></PanelBody></SectionCard>;
}

export function DimensionalityReductionView({ basePayload }: { basePayload: BasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const [method, setMethod] = useState<"svd" | "pca">("svd");
    const [components, setComponents] = useState(2);
    const mutation = useRunMutation(() => runDimensionalityReduction(ctx.selectedCorpusId, { ...basePayload, method, n_components: components }), "Dimensionality reduction", setRunId, showToast);
    return <SectionCard title="Dimensionality reduction" description="Project unit vectors into two or three dimensions for exploratory inspection."><PanelBody run={run} runQuery={runQuery}><Stack direction={{ xs: "column", sm: "row" }} spacing={1}><TextField select size="small" label="Method" value={method} onChange={(event) => setMethod(event.target.value as typeof method)}><MenuItem value="svd">Truncated SVD</MenuItem><MenuItem value="pca">PCA</MenuItem></TextField><TextField size="small" type="number" label="Components" value={components} onChange={(event) => setComponents(Math.min(3, Math.max(2, Number(event.target.value) || 2)))} inputProps={{ min: 2, max: 3 }} /></Stack><Button variant="contained" startIcon={<RunIcon />} onClick={() => mutation.mutate()} disabled={mutation.isPending} sx={{ alignSelf: "flex-start" }}>Reduce dimensions</Button></PanelBody></SectionCard>;
}

export function ReadabilityView({ basePayload }: { basePayload: BasePayload }) {
    const { ctx, run, runQuery, setRunId, showToast } = useAnalysisRun();
    const mutation = useRunMutation(() => runReadability(ctx.selectedCorpusId, basePayload), "Readability analysis", setRunId, showToast);
    return <SectionCard title="Readability" description="Calculate corpus and unit-level readability measures from the selected text units."><PanelBody run={run} runQuery={runQuery}><Button variant="contained" startIcon={<RunIcon />} onClick={() => mutation.mutate()} disabled={mutation.isPending} sx={{ alignSelf: "flex-start" }}>Calculate readability</Button></PanelBody></SectionCard>;
}
