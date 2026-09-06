import { useState } from "react";
import {
    Alert,
    Button,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { PlayArrow as TrainIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getRun,
    listPreprocessingProfiles,
    listTopicLabels,
    nameTopic,
    trainTopicModel,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
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
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";

const TOPIC_TABS = ["setup", "topics", "distribution", "metadata", "documents"] as const;
type TopicTab = (typeof TOPIC_TABS)[number];

const TOPIC_TAB_ITEMS: Array<{ value: TopicTab; label: string }> = [
    { value: "setup", label: "Setup" },
    { value: "topics", label: "Topics" },
    { value: "distribution", label: "Distribution" },
    { value: "metadata", label: "Metadata" },
    { value: "documents", label: "Documents" },
];

type TopicTerm = { term?: string; weight?: number };
type TopicRow = { topic_id?: number | string; top_terms?: TopicTerm[] };
type RepresentativeUnit = { document_title?: string | null; text?: string; weight?: number };

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

function parseRepresentatives(results: unknown, topicId: string): RepresentativeUnit[] {
    const representatives = asRecord(asRecord(results)?.representative_units);
    const units = representatives?.[topicId];
    return Array.isArray(units) ? (units as RepresentativeUnit[]) : [];
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
    const keys = ["n_topics", "topic_diversity", "top_term_overlap", "perplexity"] as const;
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

export default function TopicsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [algorithm, setAlgorithm] = useState<"lda" | "nmf">("lda");
    const [nTopics, setNTopics] = useState(5);
    const [maxIterations, setMaxIterations] = useState(20);
    const [randomSeed, setRandomSeed] = useState(42);
    const [profileId, setProfileId] = useState("");
    const [organization, setOrganization] = useState("");
    const [language, setLanguage] = useState("");
    const [region, setRegion] = useState("");
    const [culturalSphere, setCulturalSphere] = useState("");
    const [publicationYearMin, setPublicationYearMin] = useState("");
    const [publicationYearMax, setPublicationYearMax] = useState("");
    const [runId, setRunId] = useState<string | null>(null);
    const [selectedTopicId, setSelectedTopicId] = useState<string>("0");
    const [draftNames, setDraftNames] = useState<Record<string, string>>({});
    const [tab, setTab] = useTabQueryParam(TOPIC_TABS, "setup");

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: activeRunRefetchInterval,
    });

    const completed = runQuery.data?.status === "completed";
    const topics = parseTopics(runQuery.data?.results);
    const dominantCounts = parseDominantCounts(runQuery.data?.results);
    const topicPrevalence = parseTopicPrevalence(runQuery.data?.results);
    const cards = diagnosticCards(runQuery.data?.metrics);

    const labelsQuery = useQuery({
        queryKey: queryKeys.textResearch.topicLabels(runId ?? ""),
        queryFn: () => listTopicLabels(runId!),
        enabled: Boolean(runId && completed),
    });

    const trainMutation = useMutation({
        mutationFn: () =>
            trainTopicModel(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                algorithm,
                n_topics: nTopics,
                max_iterations: maxIterations,
                random_seed: randomSeed,
                preprocessing_profile_id: profileId || undefined,
                organization: optionalText(organization),
                language: optionalText(language),
                region: optionalText(region),
                cultural_sphere: optionalText(culturalSphere),
                publication_year_min: publicationYearMin ? Number(publicationYearMin) : undefined,
                publication_year_max: publicationYearMax ? Number(publicationYearMax) : undefined,
                run_async: true,
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            setSelectedTopicId("0");
            setDraftNames({});
            setTab("topics");
            showToast({ message: "Topic model training started.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to train topic model."),
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

    const labelByTopic = new Map(
        (labelsQuery.data ?? []).map((label) => [String(label.topic_id), label.human_name])
    );

    const selectedTopic =
        topics.find((topic) => String(topic.topic_id ?? "") === selectedTopicId) ?? topics[0];
    const termItems =
        selectedTopic?.top_terms?.map((term) => ({
            label: term.term ?? "",
            value: typeof term.weight === "number" ? term.weight : 0,
        })) ?? [];
    const representativeUnits = parseRepresentatives(
        runQuery.data?.results,
        String(selectedTopic?.topic_id ?? selectedTopicId)
    );
    const metadataBreakdowns = parseMetadataBreakdowns(runQuery.data?.results);

    const topicDisplayName = (topicId: number | string | undefined) => {
        const key = String(topicId ?? "");
        return labelByTopic.get(key) || `Topic ${key}`;
    };

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
                                onChange={(e) => setAlgorithm(e.target.value as "lda" | "nmf")}
                                sx={{ minWidth: 140 }}
                            >
                                <MenuItem value="lda">LDA</MenuItem>
                                <MenuItem value="nmf">NMF</MenuItem>
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
                        </Stack>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                size="small"
                                label="Organization filter"
                                value={organization}
                                onChange={(e) => setOrganization(e.target.value)}
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                label="Language filter"
                                value={language}
                                onChange={(e) => setLanguage(e.target.value)}
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                label="Cultural sphere filter"
                                value={culturalSphere}
                                onChange={(e) => setCulturalSphere(e.target.value)}
                                placeholder="Optional"
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Publication year from"
                                value={publicationYearMin}
                                onChange={(e) => setPublicationYearMin(e.target.value)}
                                sx={{ width: 180 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Publication year to"
                                value={publicationYearMax}
                                onChange={(e) => setPublicationYearMax(e.target.value)}
                                sx={{ width: 160 }}
                            />
                            <TextField
                                size="small"
                                label="Region filter"
                                value={region}
                                onChange={(e) => setRegion(e.target.value)}
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

            {tab !== "setup" && !runId ? (
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

            {runId && tab !== "setup" ? (
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
                                        {cards.length ? <MetricCards items={cards} /> : null}

                                        {tab === "topics" ? (
                                            <Stack spacing={2}>
                                                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
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
                                                </Stack>

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
                                                            terms: topic.top_terms?.map((term) => term.term).filter(Boolean).join(", ") ?? "",
                                                        };
                                                    })}
                                                    columns={[
                                                        { id: "topic", label: "Topic ID", value: (row) => row.topicId },
                                                        { id: "name", label: "Human name", value: (row) => row.name },
                                                        { id: "prevalence", label: "Prevalence", value: (row) => row.prevalence as number | null, align: "right" },
                                                        { id: "dominant", label: "Dominant units", value: (row) => row.dominant as number | null, align: "right" },
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
                                                        <Typography variant="subtitle2">Topic prevalence by metadata</Typography>
                                                        {Object.entries(metadataBreakdowns).map(([field, values]) => (
                                                            <Stack key={field} spacing={0.5}>
                                                                <Typography variant="body2" color="text.secondary">{field}</Typography>
                                                                <MatrixHeatmap
                                                                    rowLabels={Object.keys(values)}
                                                                    colLabels={topics.map((topic) => topicDisplayName(topic.topic_id))}
                                                                    values={Object.values(values).map((counts) =>
                                                                        topics.map((topic) => Number(counts[String(topic.topic_id ?? "")] ?? 0))
                                                                    )}
                                                                    formatCell={(value) => value == null ? "—" : String(value)}
                                                                />
                                                                {field === "publication_year" ? (
                                                                    <ScientificLineChart
                                                                        series={topics.map((topic) => ({
                                                                            label: topicDisplayName(topic.topic_id),
                                                                            points: Object.entries(values)
                                                                                .map(([year, counts]) => ({ x: Number(year), y: Number(counts[String(topic.topic_id ?? "")] ?? 0) }))
                                                                                .filter((point) => Number.isFinite(point.x))
                                                                                .sort((a, b) => a.x - b.x),
                                                                        }))}
                                                                    />
                                                                ) : null}
                                                            </Stack>
                                                        ))}
                                                    </>
                                                ) : (
                                                    <Typography variant="body2" color="text.secondary">
                                                        No metadata breakdowns available for this run.
                                                    </Typography>
                                                )}
                                            </Stack>
                                        ) : null}

                                        {tab === "documents" ? (
                                            <Stack spacing={2}>
                                                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
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
                                                </Stack>
                                                <Typography variant="subtitle2">
                                                    Representative units — {topicDisplayName(selectedTopic?.topic_id)}
                                                </Typography>
                                                <Stack spacing={1}>
                                                    {representativeUnits.length ? (
                                                        representativeUnits.map((unit, index) => (
                                                            <Typography key={index} variant="body2">
                                                                <strong>{unit.document_title || "Untitled document"}</strong>
                                                                {" · "}{Number(unit.weight || 0).toFixed(3)} — {unit.text}
                                                            </Typography>
                                                        ))
                                                    ) : (
                                                        <Typography variant="body2" color="text.secondary">
                                                            No representative units for this topic.
                                                        </Typography>
                                                    )}
                                                </Stack>
                                            </Stack>
                                        ) : null}
                                    </>
                                ) : isActiveRunStatus(runQuery.data.status) ? (
                                    <Typography color="text.secondary">Training in progress…</Typography>
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
