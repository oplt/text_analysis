import { useMemo, useState } from "react";
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
import { PlayArrow as TrainIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    compareRuns,
    getRun,
    listPreprocessingProfiles,
    listRuns,
    listTopicLabels,
    nameTopic,
    runTopicKSweep,
    runTopicSeedStability,
    trainTopicModel,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { ActiveRunActions } from "../components/ActiveRunActions";
import { AskAboutThisButton } from "../components/assistant/AskAboutThisButton";
import { useResearchContext } from "../hooks/useResearchContext";
import { PageTabs } from "../../../components/ui/PageTabs";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import {
    MetricCards,
    MatrixHeatmap,
    RankedBarChart,
    ResultsInspector,
    ScientificLineChart,
    SimpleLineLikeBars,
} from "../components/ResearchCharts";
import { ResearchResultsTable } from "../components/ResearchResults";
import { RunStatusChip } from "../components/ResearchShared";
import {
    TopicComparisonView,
    TopicDocumentExplorer,
    TopicKSweepView,
    TopicSeedStabilityView,
} from "../components/TopicDiagnosticsPanels";
import {
    DEFAULT_TOPIC_FILTERS,
    type SharedTopicFilters,
} from "../components/topicFilters";
import { useRunEvents } from "../hooks/useRunEvents";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";

const TOPIC_TABS = [
    "setup",
    "topics",
    "distribution",
    "metadata",
    "documents",
    "ksweep",
    "stability",
    "compare",
] as const;
type TopicTab = (typeof TOPIC_TABS)[number];

const TOPIC_TAB_ITEMS: Array<{ value: TopicTab; label: string }> = [
    { value: "setup", label: "Setup" },
    { value: "topics", label: "Topics" },
    { value: "distribution", label: "Distribution" },
    { value: "metadata", label: "Metadata" },
    { value: "documents", label: "Documents" },
    { value: "ksweep", label: "K sweep" },
    { value: "stability", label: "Seed stability" },
    { value: "compare", label: "Compare" },
];

type TopicTerm = { term?: string; weight?: number };
type TopicRow = { topic_id?: number | string; top_terms?: TopicTerm[] };

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function parseTopics(results: unknown): TopicRow[] {
    const data = asRecord(results);
    const topics = data?.topics;
    return Array.isArray(topics) ? (topics as TopicRow[]) : [];
}

function parseDominantCounts(results: unknown): Array<{ label: string; value: number }> {
    const data = asRecord(results);
    const counts = asRecord(data?.dominant_topic_counts);
    if (!counts) return [];
    return Object.entries(counts)
        .map(([key, value]) => ({
            label: `Topic ${key}`,
            value: typeof value === "number" ? value : Number(value) || 0,
        }))
        .sort((a, b) => Number(a.label.replace(/\D/g, "")) - Number(b.label.replace(/\D/g, "")));
}

function parseTopicPrevalence(results: unknown): Array<{ label: string; value: number }> {
    const prevalence = asRecord(asRecord(results)?.topic_prevalence);
    if (!prevalence) return [];
    return Object.entries(prevalence)
        .map(([topicId, value]) => ({ label: `Topic ${topicId}`, value: Number(value) || 0 }))
        .sort((a, b) => Number(a.label.replace(/\D/g, "")) - Number(b.label.replace(/\D/g, "")));
}

function parseMetadataBreakdowns(results: unknown): Record<string, Record<string, Record<string, number>>> {
    const raw = asRecord(asRecord(results)?.metadata_breakdowns);
    if (!raw) return {};
    return Object.fromEntries(
        Object.entries(raw).flatMap(([field, values]) => {
            const parsedValues = asRecord(values);
            if (!parsedValues) return [];
            return [[field, parsedValues as Record<string, Record<string, number>>]];
        })
    );
}

function diagnosticCards(metrics: unknown): Array<{ label: string; value: string | number }> {
    const data = asRecord(metrics);
    if (!data) return [];
    const keys = ["n_topics", "topic_diversity", "top_term_overlap", "perplexity", "coherence"] as const;
    return keys
        .filter((key) => data[key] != null)
        .map((key) => {
            const raw = data[key];
            const value =
                typeof raw === "number"
                    ? Number.isInteger(raw)
                        ? raw
                        : Number(raw.toFixed(4))
                    : String(raw);
            return { label: key.replace(/_/g, " "), value };
        });
}

function optionalText(value: string): string | undefined {
    const trimmed = value.trim();
    return trimmed ? trimmed : undefined;
}

function filterArgs(filters: SharedTopicFilters) {
    return {
        organization: optionalText(filters.organization),
        language: optionalText(filters.language),
        region: optionalText(filters.region),
        cultural_sphere: optionalText(filters.culturalSphere),
        publication_year_min: filters.publicationYearMin
            ? Number(filters.publicationYearMin)
            : undefined,
        publication_year_max: filters.publicationYearMax
            ? Number(filters.publicationYearMax)
            : undefined,
    };
}

export default function TopicsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [algorithm, setAlgorithm] = useState<"lda" | "nmf" | "semantic_stack" | "bertopic">("lda");
    const [nTopics, setNTopics] = useState(5);
    const [maxIterations, setMaxIterations] = useState(20);
    const [randomSeed, setRandomSeed] = useState(42);
    const [profileId, setProfileId] = useState("");
    const [holdoutFraction, setHoldoutFraction] = useState("");
    const [groupByText, setGroupByText] = useState("publication_year,organization");
    const [embeddingProvider, setEmbeddingProvider] = useState<"hashing" | "sentence_transformers">(
        "hashing"
    );
    const [embeddingModelName, setEmbeddingModelName] = useState("all-MiniLM-L6-v2");
    const [persistEmbeddings, setPersistEmbeddings] = useState(true);
    const [filters, setFilters] = useState<SharedTopicFilters>(DEFAULT_TOPIC_FILTERS);
    const [runId, setRunId] = useState<string | null>(null);
    const [kSweepRunId, setKSweepRunId] = useState<string | null>(null);
    const [stabilityRunId, setStabilityRunId] = useState<string | null>(null);
    const [compareAId, setCompareAId] = useState("");
    const [compareBId, setCompareBId] = useState("");
    const [selectedTopicId, setSelectedTopicId] = useState<string>("0");
    const [draftNames, setDraftNames] = useState<Record<string, string>>({});
    const [tab, setTab] = useTabQueryParam(TOPIC_TABS, "setup");
    const sseConnected = useRunEvents(runId, ctx.projectId);
    const kSweepSse = useRunEvents(kSweepRunId, ctx.projectId);
    const stabilitySse = useRunEvents(stabilityRunId, ctx.projectId);

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: ({ signal }) => listPreprocessingProfiles(ctx.projectId, signal),
        enabled: Boolean(ctx.projectId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: ({ signal }) => getRun(runId!, signal),
        enabled: Boolean(runId),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const kSweepQuery = useQuery({
        queryKey: queryKeys.textResearch.run(kSweepRunId ?? ""),
        queryFn: ({ signal }) => getRun(kSweepRunId!, signal),
        enabled: Boolean(kSweepRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, kSweepSse),
    });

    const stabilityQuery = useQuery({
        queryKey: queryKeys.textResearch.run(stabilityRunId ?? ""),
        queryFn: ({ signal }) => getRun(stabilityRunId!, signal),
        enabled: Boolean(stabilityRunId),
        refetchInterval: (query) => activeRunRefetchInterval(query, stabilitySse),
    });

    const topicRunsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId, "topic_model"),
        queryFn: ({ signal }) => listRuns(ctx.projectId, {
                corpus_id: ctx.selectedCorpusId || undefined,
                run_type: "topic_model",
                limit: 50,
            }, signal),
        enabled: Boolean(ctx.projectId) && (tab === "compare" || tab === "ksweep" || tab === "stability"),
    });

    const compareAQuery = useQuery({
        queryKey: queryKeys.textResearch.run(compareAId),
        queryFn: ({ signal }) => getRun(compareAId, signal),
        enabled: Boolean(compareAId),
    });

    const compareBQuery = useQuery({
        queryKey: queryKeys.textResearch.run(compareBId),
        queryFn: ({ signal }) => getRun(compareBId, signal),
        enabled: Boolean(compareBId),
    });

    const compareQuery = useQuery({
        queryKey: ["text-research", "topic-compare", compareAId, compareBId],
        queryFn: () => compareRuns(compareAId, compareBId),
        enabled: Boolean(compareAId && compareBId && compareAId !== compareBId),
    });

    const completed = runQuery.data?.status === "completed";
    const topics = parseTopics(runQuery.data?.results);
    const dominantCounts = parseDominantCounts(runQuery.data?.results);
    const topicPrevalence = parseTopicPrevalence(runQuery.data?.results);
    const cards = diagnosticCards(runQuery.data?.metrics);
    const resultsRecord = asRecord(runQuery.data?.results);
    const topicFamily =
        typeof resultsRecord?.family === "string" ? resultsRecord.family : null;
    const topicNotes = Array.isArray(resultsRecord?.notes)
        ? resultsRecord.notes.filter((n): n is string => typeof n === "string")
        : typeof resultsRecord?.notes === "string"
          ? [resultsRecord.notes]
          : [];
    const metadataBreakdowns = parseMetadataBreakdowns(runQuery.data?.results);

    const labelsQuery = useQuery({
        queryKey: queryKeys.textResearch.topicLabels(runId ?? ""),
        queryFn: () => listTopicLabels(runId!),
        enabled: Boolean(runId && completed),
    });

    const trainMutation = useMutation({
        mutationFn: () => {
            const holdout = holdoutFraction.trim() ? Number(holdoutFraction) : undefined;
            const groupBy = groupByText
                .split(/[,\s]+/)
                .map((part) => part.trim())
                .filter(Boolean);
            return trainTopicModel(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                algorithm,
                n_topics: nTopics,
                max_iterations: maxIterations,
                random_seed: randomSeed,
                preprocessing_profile_id: profileId || undefined,
                holdout_fraction:
                    holdout != null && !Number.isNaN(holdout) ? holdout : undefined,
                group_by: groupBy.length ? groupBy : undefined,
                embedding_provider:
                    algorithm === "semantic_stack" || algorithm === "bertopic"
                        ? embeddingProvider
                        : undefined,
                embedding_model_name:
                    (algorithm === "semantic_stack" || algorithm === "bertopic") &&
                    embeddingProvider === "sentence_transformers"
                        ? embeddingModelName.trim() || undefined
                        : undefined,
                persist_embedding_artifacts:
                    algorithm === "semantic_stack" || algorithm === "bertopic"
                        ? persistEmbeddings
                        : undefined,
                ...filterArgs(filters),
                run_async: true,
            });
        },
        onSuccess: (run) => {
            setRunId(run.id);
            setSelectedTopicId("0");
            setDraftNames({});
            setTab("topics");
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.runs(
                    ctx.projectId,
                    ctx.selectedCorpusId,
                    "topic_model"
                ),
            });
            showToast({ message: "Topic model training started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to train topic model."),
                severity: "error",
            }),
    });

    const kSweepMutation = useMutation({
        mutationFn: (
            payload: Omit<Parameters<typeof runTopicKSweep>[1], "unit_type" | "run_async">
        ) =>
            runTopicKSweep(ctx.selectedCorpusId, {
                ...payload,
                unit_type: ctx.unitType,
                run_async: true,
            }),
        onSuccess: (run) => {
            setKSweepRunId(run.id);
            showToast({ message: "Topic K sweep started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to start K sweep."),
                severity: "error",
            }),
    });

    const stabilityMutation = useMutation({
        mutationFn: (
            payload: Omit<Parameters<typeof runTopicSeedStability>[1], "unit_type" | "run_async">
        ) =>
            runTopicSeedStability(ctx.selectedCorpusId, {
                ...payload,
                unit_type: ctx.unitType,
                run_async: true,
            }),
        onSuccess: (run) => {
            setStabilityRunId(run.id);
            showToast({ message: "Seed stability run started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to start seed stability."),
                severity: "error",
            }),
    });

    const nameMutation = useMutation({
        mutationFn: (payload: { topic_id: number | string; human_name: string }) =>
            nameTopic(runId!, payload),
        onSuccess: () => {
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.topicLabels(runId ?? ""),
            });
            showToast({
                message: "Topic label saved. Model output is unchanged.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to save topic name."),
                severity: "error",
            }),
    });

    const labelByTopic = useMemo(
        () =>
            new Map((labelsQuery.data ?? []).map((label) => [String(label.topic_id), label.human_name])),
        [labelsQuery.data]
    );

    const selectedTopic =
        topics.find((topic) => String(topic.topic_id ?? "") === selectedTopicId) ?? topics[0];
    const termItems =
        selectedTopic?.top_terms?.map((term) => ({
            label: term.term ?? "",
            value: typeof term.weight === "number" ? term.weight : 0,
        })) ?? [];

    const topicDisplayName = (topicId: number | string | undefined) => {
        const key = String(topicId ?? "");
        return labelByTopic.get(key) || `Topic ${key}`;
    };

    const profileOptions = (profilesQuery.data ?? []).map((profile) => ({
        id: profile.id,
        name: profile.name,
    }));

    const resultTabs = new Set(["topics", "distribution", "metadata", "documents"]);

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={TOPIC_TAB_ITEMS}
                ariaLabel="Topic modeling workflow"
            />

            {tab === "setup" ? (
                <SectionCard
                    title="Topic modeling"
                    description="Train LDA or NMF on segmented units. Human names are labels only — they do not retrain the model."
                >
                    <Stack spacing={2} sx={{ mb: 2 }}>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                select
                                size="small"
                                label="Algorithm"
                                value={algorithm}
                                onChange={(e) =>
                                    setAlgorithm(
                                        e.target.value as "lda" | "nmf" | "semantic_stack" | "bertopic"
                                    )
                                }
                                sx={{ minWidth: 180 }}
                            >
                                <MenuItem value="lda">LDA (classical)</MenuItem>
                                <MenuItem value="nmf">NMF (classical)</MenuItem>
                                <MenuItem value="semantic_stack">Semantic stack</MenuItem>
                                <MenuItem value="bertopic">BERTopic (optional)</MenuItem>
                            </TextField>
                            <TextField
                                size="small"
                                type="number"
                                label="Number of topics"
                                value={nTopics}
                                onChange={(e) => setNTopics(Number(e.target.value) || 2)}
                                inputProps={{ min: 2, max: 50 }}
                                sx={{ width: 160 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Max iterations"
                                value={maxIterations}
                                onChange={(e) => setMaxIterations(Number(e.target.value) || 1)}
                                inputProps={{ min: 1, max: 500 }}
                                sx={{ width: 150 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Random seed"
                                value={randomSeed}
                                onChange={(e) => setRandomSeed(Number(e.target.value) || 0)}
                                sx={{ width: 140 }}
                            />
                            <TextField
                                select
                                size="small"
                                label="Preprocessing profile"
                                value={profileId}
                                onChange={(e) => setProfileId(e.target.value)}
                                sx={{ minWidth: 220 }}
                            >
                                <MenuItem value="">Default</MenuItem>
                                {(profilesQuery.data ?? []).map((profile) => (
                                    <MenuItem key={profile.id} value={profile.id}>
                                        {profile.name}
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                size="small"
                                type="number"
                                label="Holdout fraction"
                                value={holdoutFraction}
                                onChange={(e) => setHoldoutFraction(e.target.value)}
                                inputProps={{ min: 0, max: 0.5, step: 0.05 }}
                                sx={{ width: 150 }}
                                helperText="Optional LDA perplexity"
                            />
                            <TextField
                                size="small"
                                label="group_by fields"
                                value={groupByText}
                                onChange={(e) => setGroupByText(e.target.value)}
                                sx={{ minWidth: 240 }}
                                helperText="Comma-separated metadata fields"
                            />
                        </Stack>
                        {algorithm === "semantic_stack" || algorithm === "bertopic" ? (
                            <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={2}
                                flexWrap="wrap"
                                useFlexGap
                            >
                                <Alert severity="info" sx={{ width: "100%" }}>
                                    Semantic topics keep LDA/NMF available. UMAP/HDBSCAN are used when
                                    installed; otherwise TruncatedSVD + KMeans with explicit notes.
                                </Alert>
                                <TextField
                                    select
                                    size="small"
                                    label="Embedding provider"
                                    value={embeddingProvider}
                                    onChange={(e) =>
                                        setEmbeddingProvider(
                                            e.target.value as "hashing" | "sentence_transformers"
                                        )
                                    }
                                    sx={{ minWidth: 220 }}
                                >
                                    <MenuItem value="hashing">Hashing (lexical baseline)</MenuItem>
                                    <MenuItem value="sentence_transformers">
                                        Sentence transformers (optional)
                                    </MenuItem>
                                </TextField>
                                {embeddingProvider === "sentence_transformers" ? (
                                    <TextField
                                        size="small"
                                        label="ST model name"
                                        value={embeddingModelName}
                                        onChange={(e) => setEmbeddingModelName(e.target.value)}
                                        sx={{ minWidth: 220 }}
                                    />
                                ) : null}
                                <FormControlLabel
                                    control={
                                        <Checkbox
                                            checked={persistEmbeddings}
                                            onChange={(_, checked) => setPersistEmbeddings(checked)}
                                        />
                                    }
                                    label="Cache embeddings as artifacts"
                                />
                            </Stack>
                        ) : null}
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                size="small"
                                label="Organization filter"
                                value={filters.organization}
                                onChange={(e) =>
                                    setFilters((prev) => ({ ...prev, organization: e.target.value }))
                                }
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                label="Language filter"
                                value={filters.language}
                                onChange={(e) =>
                                    setFilters((prev) => ({ ...prev, language: e.target.value }))
                                }
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                label="Cultural sphere filter"
                                value={filters.culturalSphere}
                                onChange={(e) =>
                                    setFilters((prev) => ({
                                        ...prev,
                                        culturalSphere: e.target.value,
                                    }))
                                }
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Publication year from"
                                value={filters.publicationYearMin}
                                onChange={(e) =>
                                    setFilters((prev) => ({
                                        ...prev,
                                        publicationYearMin: e.target.value,
                                    }))
                                }
                                sx={{ width: 180 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Publication year to"
                                value={filters.publicationYearMax}
                                onChange={(e) =>
                                    setFilters((prev) => ({
                                        ...prev,
                                        publicationYearMax: e.target.value,
                                    }))
                                }
                                sx={{ width: 160 }}
                            />
                            <TextField
                                size="small"
                                label="Region filter"
                                value={filters.region}
                                onChange={(e) =>
                                    setFilters((prev) => ({ ...prev, region: e.target.value }))
                                }
                                placeholder="Optional"
                                sx={{ minWidth: 160 }}
                            />
                            <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center" }}>
                                Unit type: {ctx.unitType}
                            </Typography>
                        </Stack>
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
            ) : null}

            {tab === "ksweep" ? (
                <SectionCard
                    title="Topic K sweep"
                    description="Compare diagnostics across candidate topic counts without auto-selecting a winner."
                >
                    <TopicKSweepView
                        corpusId={ctx.selectedCorpusId}
                        unitType={ctx.unitType}
                        profiles={profileOptions}
                        filters={filters}
                        onFiltersChange={setFilters}
                        onSubmit={(payload) => kSweepMutation.mutate(payload)}
                        isPending={kSweepMutation.isPending}
                        run={kSweepQuery.data}
                    />
                </SectionCard>
            ) : null}

            {tab === "stability" ? (
                <SectionCard
                    title="Seed stability"
                    description="Measure topic solution sensitivity to random initialization."
                >
                    <TopicSeedStabilityView
                        corpusId={ctx.selectedCorpusId}
                        profiles={profileOptions}
                        filters={filters}
                        onFiltersChange={setFilters}
                        onSubmit={(payload) => stabilityMutation.mutate(payload)}
                        isPending={stabilityMutation.isPending}
                        run={stabilityQuery.data}
                    />
                </SectionCard>
            ) : null}

            {tab === "compare" ? (
                <SectionCard
                    title="Topic comparison across runs"
                    description="Diff parameters/metrics and inspect top terms side-by-side for two training runs."
                >
                    <QueryBoundary
                        isLoading={topicRunsQuery.isLoading}
                        isError={topicRunsQuery.isError}
                        error={topicRunsQuery.error}
                        onRetry={() => void topicRunsQuery.refetch()}
                    >
                        <TopicComparisonView
                            runs={topicRunsQuery.data?.items ?? []}
                            runA={compareAQuery.data}
                            runB={compareBQuery.data}
                            selectedAId={compareAId}
                            selectedBId={compareBId}
                            onSelectA={setCompareAId}
                            onSelectB={setCompareBId}
                            compare={compareQuery.data}
                            compareLoading={compareQuery.isLoading}
                            compareError={
                                compareQuery.isError
                                    ? getQueryErrorMessage(compareQuery.error, "Compare failed.")
                                    : null
                            }
                        />
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            {resultTabs.has(tab) && !runId ? (
                <EmptyState
                    icon={<TrainIcon fontSize="large" />}
                    title="No topic model yet"
                    description="Configure and train a topic model in Setup to explore topics, distributions, and documents."
                    action={
                        <Button variant="contained" onClick={() => setTab("setup")}>
                            Go to setup
                        </Button>
                    }
                />
            ) : null}

            {runId && resultTabs.has(tab) ? (
                <SectionCard title="Topic model results" description="Diagnostics stay visible while you switch result views.">
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

                                {completed && topics.length ? (
                                    <>
                                        {topicFamily || topicNotes.length ? (
                                            <Alert severity="info">
                                                {topicFamily ? `Family: ${topicFamily}. ` : null}
                                                {topicNotes.length
                                                    ? topicNotes.join(" ")
                                                    : "Classical LDA/NMF remain available alongside semantic algorithms."}
                                            </Alert>
                                        ) : null}
                                        {cards.length ? <MetricCards items={cards} /> : null}

                                        {tab === "topics" ? (
                                            <Stack spacing={2}>
                                                <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
                                                    <TextField
                                                        select
                                                        size="small"
                                                        label="Selected topic"
                                                        value={String(selectedTopic?.topic_id ?? selectedTopicId)}
                                                        onChange={(e) => setSelectedTopicId(e.target.value)}
                                                        sx={{ minWidth: 220 }}
                                                    >
                                                        {topics.map((topic) => (
                                                            <MenuItem
                                                                key={String(topic.topic_id)}
                                                                value={String(topic.topic_id)}
                                                            >
                                                                {topicDisplayName(topic.topic_id)}
                                                            </MenuItem>
                                                        ))}
                                                    </TextField>
                                                    <AskAboutThisButton
                                                        label="Ask about this topic"
                                                        intent="evidence"
                                                        question={`AI-assisted interpretation (not model output): interpret topic "${topicDisplayName(selectedTopic?.topic_id)}" using source evidence. Top terms: ${termItems.map((t) => t.label).join(", ")}.`}
                                                        onAsk={ctx.askAbout}
                                                    />
                                                    <AskAboutThisButton
                                                        label="Representative passages"
                                                        intent="evidence"
                                                        question={`Find representative passages for topic terms: ${termItems.map((t) => t.label).join(", ")}.`}
                                                        onAsk={ctx.askAbout}
                                                    />
                                                </Stack>

                                                <Alert severity="info">
                                                    Topic terms and distributions are computed by the topic
                                                    model. Ask Corpus provides AI-assisted interpretation only.
                                                </Alert>

                                                <Typography variant="subtitle2">
                                                    Top terms — {topicDisplayName(selectedTopic?.topic_id)}
                                                </Typography>
                                                <RankedBarChart
                                                    items={termItems}
                                                    valueFormatter={(v) =>
                                                        v == null ? "" : Number(v).toFixed(4)
                                                    }
                                                />

                                                <ResearchResultsTable
                                                    rows={topics.map((topic, index) => {
                                                        const topicId = topic.topic_id ?? index;
                                                        const key = String(topicId);
                                                        return {
                                                            id: key,
                                                            topicId,
                                                            name: topicDisplayName(topicId),
                                                            prevalence: asRecord(
                                                                asRecord(runQuery.data?.results)?.topic_prevalence
                                                            )?.[key],
                                                            dominant: asRecord(
                                                                asRecord(runQuery.data?.results)?.dominant_topic_counts
                                                            )?.[key],
                                                            terms:
                                                                topic.top_terms
                                                                    ?.map((term) => term.term)
                                                                    .filter(Boolean)
                                                                    .join(", ") ?? "",
                                                        };
                                                    })}
                                                    columns={[
                                                        { id: "topic", label: "Topic ID", value: (row) => row.topicId },
                                                        { id: "name", label: "Human name", value: (row) => row.name },
                                                        {
                                                            id: "prevalence",
                                                            label: "Prevalence",
                                                            value: (row) => row.prevalence as number | null,
                                                            align: "right",
                                                        },
                                                        {
                                                            id: "dominant",
                                                            label: "Dominant units",
                                                            value: (row) => row.dominant as number | null,
                                                            align: "right",
                                                        },
                                                        { id: "terms", label: "Top terms", value: (row) => row.terms },
                                                    ]}
                                                />

                                                <Typography variant="subtitle2">Human topic names</Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Rename topics for reporting. Names are metadata only and do
                                                    not change model output.
                                                </Typography>
                                                <Stack spacing={1.5}>
                                                    {topics.map((topic) => {
                                                        const key = String(topic.topic_id ?? "");
                                                        const saved = labelByTopic.get(key) ?? "";
                                                        const draft = draftNames[key] ?? saved;
                                                        return (
                                                            <Stack
                                                                key={key}
                                                                direction={{ xs: "column", sm: "row" }}
                                                                spacing={1}
                                                                alignItems={{ sm: "center" }}
                                                            >
                                                                <Typography variant="body2" sx={{ minWidth: 88 }}>
                                                                    Topic {key}
                                                                </Typography>
                                                                <TextField
                                                                    size="small"
                                                                    label="Custom name"
                                                                    value={draft}
                                                                    onChange={(e) =>
                                                                        setDraftNames((prev) => ({
                                                                            ...prev,
                                                                            [key]: e.target.value,
                                                                        }))
                                                                    }
                                                                    placeholder={`Topic ${key}`}
                                                                    sx={{ flex: 1 }}
                                                                />
                                                                <Button
                                                                    size="small"
                                                                    variant="outlined"
                                                                    disabled={
                                                                        !draft.trim() ||
                                                                        draft.trim() === saved ||
                                                                        nameMutation.isPending
                                                                    }
                                                                    onClick={() =>
                                                                        nameMutation.mutate({
                                                                            topic_id: topic.topic_id ?? key,
                                                                            human_name: draft.trim(),
                                                                        })
                                                                    }
                                                                >
                                                                    Save name
                                                                </Button>
                                                            </Stack>
                                                        );
                                                    })}
                                                </Stack>
                                                <ResultsInspector
                                                    title="topic results"
                                                    data={{
                                                        metrics: runQuery.data.metrics,
                                                        results: runQuery.data.results,
                                                    }}
                                                />
                                            </Stack>
                                        ) : null}

                                        {tab === "distribution" ? (
                                            <Stack spacing={2}>
                                                <Typography variant="subtitle2">Dominant topic counts</Typography>
                                                <SimpleLineLikeBars
                                                    items={dominantCounts.map((item) => ({
                                                        ...item,
                                                        label: topicDisplayName(item.label.replace(/\D/g, "")),
                                                    }))}
                                                />
                                                <Typography variant="subtitle2">Topic prevalence</Typography>
                                                <RankedBarChart
                                                    items={topicPrevalence.map((item) => ({
                                                        ...item,
                                                        label: topicDisplayName(item.label.replace(/\D/g, "")),
                                                    }))}
                                                    valueFormatter={(v) =>
                                                        v == null ? "" : `${(Number(v) * 100).toFixed(1)}%`
                                                    }
                                                />
                                            </Stack>
                                        ) : null}

                                        {tab === "metadata" ? (
                                            <Stack spacing={2}>
                                                {Object.keys(metadataBreakdowns).length ? (
                                                    <>
                                                        <Typography variant="subtitle2">
                                                            Topic prevalence by metadata
                                                        </Typography>
                                                        {Object.entries(metadataBreakdowns).map(
                                                            ([field, values]) => (
                                                                <Stack key={field} spacing={0.5}>
                                                                    <Typography
                                                                        variant="body2"
                                                                        color="text.secondary"
                                                                    >
                                                                        {field}
                                                                    </Typography>
                                                                    <MatrixHeatmap
                                                                        rowLabels={Object.keys(values)}
                                                                        colLabels={topics.map((topic) =>
                                                                            topicDisplayName(topic.topic_id)
                                                                        )}
                                                                        values={Object.values(values).map(
                                                                            (counts) =>
                                                                                topics.map((topic) =>
                                                                                    Number(
                                                                                        counts[
                                                                                            String(
                                                                                                topic.topic_id ?? ""
                                                                                            )
                                                                                        ] ?? 0
                                                                                    )
                                                                                )
                                                                        )}
                                                                        formatCell={(value) =>
                                                                            value == null ? "—" : String(value)
                                                                        }
                                                                    />
                                                                    {field === "publication_year" ? (
                                                                        <ScientificLineChart
                                                                            series={topics.map((topic) => ({
                                                                                label: topicDisplayName(
                                                                                    topic.topic_id
                                                                                ),
                                                                                points: Object.entries(values)
                                                                                    .map(([year, counts]) => ({
                                                                                        x: Number(year),
                                                                                        y: Number(
                                                                                            counts[
                                                                                                String(
                                                                                                    topic.topic_id ??
                                                                                                        ""
                                                                                                )
                                                                                            ] ?? 0
                                                                                        ),
                                                                                    }))
                                                                                    .filter((point) =>
                                                                                        Number.isFinite(point.x)
                                                                                    )
                                                                                    .sort((a, b) => a.x - b.x),
                                                                            }))}
                                                                        />
                                                                    ) : null}
                                                                </Stack>
                                                            )
                                                        )}
                                                    </>
                                                ) : (
                                                    <Typography variant="body2" color="text.secondary">
                                                        No metadata breakdowns available for this run. Set
                                                        group_by fields in Setup and retrain.
                                                    </Typography>
                                                )}
                                            </Stack>
                                        ) : null}

                                        {tab === "documents" ? (
                                            <TopicDocumentExplorer
                                                results={runQuery.data.results}
                                                topics={topics}
                                                selectedTopicId={String(
                                                    selectedTopic?.topic_id ?? selectedTopicId
                                                )}
                                                onSelectTopic={setSelectedTopicId}
                                                topicDisplayName={topicDisplayName}
                                            />
                                        ) : null}
                                    </>
                                ) : isActiveRunStatus(runQuery.data.status) ? (
                                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                                        <Typography color="text.secondary">Training in progress…</Typography>
                                        <ActiveRunActions
                                            run={runQuery.data}
                                            projectId={ctx.projectId}
                                            corpusId={ctx.selectedCorpusId}
                                        />
                                    </Stack>
                                ) : (
                                    <Typography color="text.secondary">No topic results yet.</Typography>
                                )}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
