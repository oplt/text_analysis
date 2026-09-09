import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    MenuItem,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { MetricCards, RankedBarChart, ScientificLineChart } from "./ResearchCharts";
import { ResearchResultsTable } from "./ResearchResults";
import { RunStatusChip } from "./ResearchShared";
import type { AnalysisRun } from "../types";
import { topicFilterPayload, type SharedTopicFilters } from "./topicFilters";

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function num(value: unknown): number | null {
    if (typeof value === "number" && !Number.isNaN(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
}

function formatMetric(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function formatCell(value: unknown): string {
    if (value == null) return "—";
    if (typeof value === "number") return formatMetric(value);
    if (typeof value === "string") return value || "—";
    if (typeof value === "boolean") return String(value);
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

type TopicTerm = { term?: string; weight?: number };
type TopicRow = { topic_id?: number | string; top_terms?: TopicTerm[] };
type RepresentativeUnit = {
    text_unit_id?: string;
    document_title?: string | null;
    text?: string;
    weight?: number;
};

function parseTopics(results: unknown): TopicRow[] {
    const topics = asRecord(results)?.topics;
    return Array.isArray(topics) ? (topics as TopicRow[]) : [];
}

function parseRepresentatives(results: unknown, topicId: string): RepresentativeUnit[] {
    const representatives = asRecord(asRecord(results)?.representative_units);
    const units = representatives?.[topicId];
    return Array.isArray(units) ? (units as RepresentativeUnit[]) : [];
}

type ProfileOption = { id: string; name: string };

function CorpusFilterFields({
    filters,
    onChange,
}: {
    filters: SharedTopicFilters;
    onChange: (next: SharedTopicFilters) => void;
}) {
    const patch = (partial: Partial<SharedTopicFilters>) => onChange({ ...filters, ...partial });
    return (
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
            <TextField
                size="small"
                label="Organization"
                value={filters.organization}
                onChange={(e) => patch({ organization: e.target.value })}
                sx={{ minWidth: 160 }}
            />
            <TextField
                size="small"
                label="Language"
                value={filters.language}
                onChange={(e) => patch({ language: e.target.value })}
                sx={{ minWidth: 140 }}
            />
            <TextField
                size="small"
                label="Region"
                value={filters.region}
                onChange={(e) => patch({ region: e.target.value })}
                sx={{ minWidth: 140 }}
            />
            <TextField
                size="small"
                label="Cultural sphere"
                value={filters.culturalSphere}
                onChange={(e) => patch({ culturalSphere: e.target.value })}
                sx={{ minWidth: 160 }}
            />
            <TextField
                size="small"
                type="number"
                label="Year from"
                value={filters.publicationYearMin}
                onChange={(e) => patch({ publicationYearMin: e.target.value })}
                sx={{ width: 120 }}
            />
            <TextField
                size="small"
                type="number"
                label="Year to"
                value={filters.publicationYearMax}
                onChange={(e) => patch({ publicationYearMax: e.target.value })}
                sx={{ width: 120 }}
            />
        </Stack>
    );
}

export function TopicKSweepView({
    corpusId,
    unitType,
    profiles,
    filters,
    onFiltersChange,
    onSubmit,
    isPending,
    run,
}: {
    corpusId: string | null;
    unitType: string;
    profiles: ProfileOption[];
    filters: SharedTopicFilters;
    onFiltersChange: (next: SharedTopicFilters) => void;
    onSubmit: (payload: {
        algorithm: "lda" | "nmf";
        k_values: number[];
        max_iterations: number;
        random_seed: number;
        preprocessing_profile_id?: string;
        holdout_fraction?: number;
    } & ReturnType<typeof topicFilterPayload>) => void;
    isPending: boolean;
    run: AnalysisRun | null | undefined;
}) {
    const [algorithm, setAlgorithm] = useState<"lda" | "nmf">("lda");
    const [kValuesText, setKValuesText] = useState("3,5,8,10");
    const [maxIterations, setMaxIterations] = useState(20);
    const [randomSeed, setRandomSeed] = useState(42);
    const [profileId, setProfileId] = useState("");
    const [holdoutFraction, setHoldoutFraction] = useState("");

    const rows = useMemo(() => {
        const raw = asRecord(run?.results)?.rows;
        return Array.isArray(raw) ? raw.map((row) => asRecord(row)).filter(Boolean) : [];
    }, [run?.results]);

    const chartSeries = useMemo(() => {
        const pointsFor = (key: string) =>
            rows
                .map((row) => {
                    const k = num(row?.requested_n_topics) ?? num(row?.effective_n_topics);
                    const y = num(row?.[key]);
                    return k == null || y == null ? null : { x: k, y };
                })
                .filter((point): point is { x: number; y: number } => point != null)
                .sort((a, b) => a.x - b.x);
        return [
            { label: "coherence", points: pointsFor("coherence") },
            { label: "topic_diversity", points: pointsFor("topic_diversity") },
            { label: "top_term_overlap", points: pointsFor("top_term_overlap") },
            { label: "perplexity", points: pointsFor("perplexity") },
        ].filter((series) => series.points.length > 0);
    }, [rows]);

    return (
        <Stack spacing={2}>
            <Alert severity="info">
                K sweep fits multiple topic counts and reports diagnostics. It does not pick a
                “best” K — that remains a research judgment.
            </Alert>
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
                    label="K values (comma-separated)"
                    value={kValuesText}
                    onChange={(e) => setKValuesText(e.target.value)}
                    sx={{ minWidth: 220 }}
                    helperText="e.g. 3,5,8,10"
                />
                <TextField
                    size="small"
                    type="number"
                    label="Max iterations"
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(Number(e.target.value) || 1)}
                    sx={{ width: 140 }}
                />
                <TextField
                    size="small"
                    type="number"
                    label="Random seed"
                    value={randomSeed}
                    onChange={(e) => setRandomSeed(Number(e.target.value) || 0)}
                    sx={{ width: 130 }}
                />
                <TextField
                    select
                    size="small"
                    label="Preprocessing profile"
                    value={profileId}
                    onChange={(e) => setProfileId(e.target.value)}
                    sx={{ minWidth: 200 }}
                >
                    <MenuItem value="">Default</MenuItem>
                    {profiles.map((profile) => (
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
                    helperText="Optional (LDA perplexity)"
                />
            </Stack>
            <CorpusFilterFields filters={filters} onChange={onFiltersChange} />
            <Typography variant="body2" color="text.secondary">
                Unit type: {unitType}
            </Typography>
            <Button
                variant="contained"
                disabled={!corpusId || isPending}
                onClick={() => {
                    const kValues = kValuesText
                        .split(/[,\s]+/)
                        .map((part) => Number.parseInt(part.trim(), 10))
                        .filter((value) => Number.isFinite(value) && value >= 2);
                    if (!kValues.length) return;
                    const holdout = holdoutFraction.trim()
                        ? Number(holdoutFraction)
                        : undefined;
                    onSubmit({
                        algorithm,
                        k_values: kValues,
                        max_iterations: maxIterations,
                        random_seed: randomSeed,
                        preprocessing_profile_id: profileId || undefined,
                        holdout_fraction:
                            holdout != null && !Number.isNaN(holdout) ? holdout : undefined,
                        ...topicFilterPayload(filters),
                    });
                }}
                sx={{ alignSelf: "flex-start" }}
            >
                Run K sweep
            </Button>
            {run ? (
                <Typography variant="body2">
                    Sweep run {run.id} — <RunStatusChip status={run.status} />
                </Typography>
            ) : null}
            {run?.error_message ? <Alert severity="error">{run.error_message}</Alert> : null}
            {chartSeries.length ? <ScientificLineChart series={chartSeries} height={280} /> : null}
            {rows.length ? (
                <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>K</TableCell>
                                <TableCell>Status</TableCell>
                                <TableCell align="right">Coherence</TableCell>
                                <TableCell align="right">Diversity</TableCell>
                                <TableCell align="right">Overlap</TableCell>
                                <TableCell align="right">Perplexity</TableCell>
                                <TableCell>Top terms preview</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rows.map((row, index) => (
                                <TableRow key={String(row?.requested_n_topics ?? index)}>
                                    <TableCell>{formatMetric(num(row?.requested_n_topics), 0)}</TableCell>
                                    <TableCell>{String(row?.status ?? "—")}</TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.coherence))}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.topic_diversity))}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.top_term_overlap))}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.perplexity))}
                                    </TableCell>
                                    <TableCell>
                                        {Array.isArray(row?.top_terms_preview)
                                            ? (row.top_terms_preview as unknown[])
                                                  .map((terms) =>
                                                      Array.isArray(terms) ? terms.join(", ") : ""
                                                  )
                                                  .filter(Boolean)
                                                  .join(" | ")
                                            : "—"}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </Box>
            ) : run?.status === "completed" ? (
                <Typography variant="body2" color="text.secondary">
                    No sweep rows in results.
                </Typography>
            ) : null}
        </Stack>
    );
}

export function TopicSeedStabilityView({
    corpusId,
    profiles,
    filters,
    onFiltersChange,
    onSubmit,
    isPending,
    run,
}: {
    corpusId: string | null;
    profiles: ProfileOption[];
    filters: SharedTopicFilters;
    onFiltersChange: (next: SharedTopicFilters) => void;
    onSubmit: (payload: {
        algorithm: "lda" | "nmf";
        n_topics: number;
        seeds: number[];
        max_iterations: number;
        preprocessing_profile_id?: string;
    } & ReturnType<typeof topicFilterPayload>) => void;
    isPending: boolean;
    run: AnalysisRun | null | undefined;
}) {
    const [algorithm, setAlgorithm] = useState<"lda" | "nmf">("lda");
    const [nTopics, setNTopics] = useState(5);
    const [seedsText, setSeedsText] = useState("1,2,3,4,5");
    const [maxIterations, setMaxIterations] = useState(20);
    const [profileId, setProfileId] = useState("");

    const results = asRecord(run?.results);
    const pairwise = Array.isArray(results?.pairwise)
        ? results.pairwise.map((row) => asRecord(row)).filter(Boolean)
        : [];
    const meanStability = num(results?.mean_stability_jaccard) ?? num(run?.metrics?.mean_stability_jaccard);

    return (
        <Stack spacing={2}>
            <Alert severity="info">
                Multi-seed stability measures how much topic solutions change under different
                random initializations (Hungarian-matched Jaccard).
            </Alert>
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
                    label="Seeds (comma-separated)"
                    value={seedsText}
                    onChange={(e) => setSeedsText(e.target.value)}
                    sx={{ minWidth: 220 }}
                    helperText="At least 2 seeds"
                />
                <TextField
                    size="small"
                    type="number"
                    label="Max iterations"
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(Number(e.target.value) || 1)}
                    sx={{ width: 140 }}
                />
                <TextField
                    select
                    size="small"
                    label="Preprocessing profile"
                    value={profileId}
                    onChange={(e) => setProfileId(e.target.value)}
                    sx={{ minWidth: 200 }}
                >
                    <MenuItem value="">Default</MenuItem>
                    {profiles.map((profile) => (
                        <MenuItem key={profile.id} value={profile.id}>
                            {profile.name}
                        </MenuItem>
                    ))}
                </TextField>
            </Stack>
            <CorpusFilterFields filters={filters} onChange={onFiltersChange} />
            <Button
                variant="contained"
                disabled={!corpusId || isPending}
                onClick={() => {
                    const seeds = seedsText
                        .split(/[,\s]+/)
                        .map((part) => Number.parseInt(part.trim(), 10))
                        .filter((value) => Number.isFinite(value));
                    if (seeds.length < 2) return;
                    onSubmit({
                        algorithm,
                        n_topics: nTopics,
                        seeds,
                        max_iterations: maxIterations,
                        preprocessing_profile_id: profileId || undefined,
                        ...topicFilterPayload(filters),
                    });
                }}
                sx={{ alignSelf: "flex-start" }}
            >
                Run seed stability
            </Button>
            {run ? (
                <Typography variant="body2">
                    Stability run {run.id} — <RunStatusChip status={run.status} />
                </Typography>
            ) : null}
            {run?.error_message ? <Alert severity="error">{run.error_message}</Alert> : null}
            {meanStability != null ? (
                <MetricCards
                    items={[{ label: "Mean stability (Jaccard)", value: formatMetric(meanStability) }]}
                />
            ) : null}
            {pairwise.length ? (
                <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Seed A</TableCell>
                                <TableCell>Seed B</TableCell>
                                <TableCell align="right">Mean Jaccard</TableCell>
                                <TableCell align="right">Topic-word cosine</TableCell>
                                <TableCell align="right">Doc distribution sim.</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {pairwise.map((row, index) => (
                                <TableRow key={`${row?.seed_a}-${row?.seed_b}-${index}`}>
                                    <TableCell>{formatMetric(num(row?.seed_a), 0)}</TableCell>
                                    <TableCell>{formatMetric(num(row?.seed_b), 0)}</TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.mean_best_match_jaccard))}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.mean_topic_word_cosine))}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(num(row?.document_distribution_similarity))}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </Box>
            ) : null}
            {Array.isArray(results?.per_seed_diagnostics) ? (
                <ResearchResultsTable
                    rows={(results.per_seed_diagnostics as unknown[]).map((row, index) => {
                        const item = asRecord(row) ?? {};
                        const seedValue = num(item.seed);
                        return {
                            id: String(item.seed ?? index),
                            seed: seedValue ?? (item.seed != null ? String(item.seed) : null),
                            coherence: num(item.coherence),
                            diversity: num(item.topic_diversity),
                            overlap: num(item.top_term_overlap),
                            perplexity: num(item.perplexity),
                        };
                    })}
                    columns={[
                        { id: "seed", label: "Seed", value: (row) => row.seed },
                        {
                            id: "coherence",
                            label: "Coherence",
                            value: (row) => row.coherence,
                            align: "right",
                        },
                        {
                            id: "diversity",
                            label: "Diversity",
                            value: (row) => row.diversity,
                            align: "right",
                        },
                        {
                            id: "overlap",
                            label: "Overlap",
                            value: (row) => row.overlap,
                            align: "right",
                        },
                        {
                            id: "perplexity",
                            label: "Perplexity",
                            value: (row) => row.perplexity,
                            align: "right",
                        },
                    ]}
                />
            ) : null}
        </Stack>
    );
}

export function TopicDocumentExplorer({
    results,
    topics,
    selectedTopicId,
    onSelectTopic,
    topicDisplayName,
}: {
    results: unknown;
    topics: TopicRow[];
    selectedTopicId: string;
    onSelectTopic: (topicId: string) => void;
    topicDisplayName: (topicId: number | string | undefined) => string;
}) {
    const [query, setQuery] = useState("");
    const selectedTopic =
        topics.find((topic) => String(topic.topic_id ?? "") === selectedTopicId) ?? topics[0];
    const activeId = String(selectedTopic?.topic_id ?? selectedTopicId);
    const units = parseRepresentatives(results, activeId);
    const prevalence = asRecord(asRecord(results)?.topic_prevalence);
    const filtered = units.filter((unit) => {
        if (!query.trim()) return true;
        const haystack = `${unit.document_title ?? ""} ${unit.text ?? ""}`.toLowerCase();
        return haystack.includes(query.trim().toLowerCase());
    });
    const weightItems = units.map((unit, index) => ({
        label: unit.document_title || `Unit ${index + 1}`,
        value: Number(unit.weight || 0),
    }));

    if (!topics.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                Train a topic model to explore document–topic distributions.
            </Typography>
        );
    }

    return (
        <Stack spacing={2}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} flexWrap="wrap">
                <TextField
                    select
                    size="small"
                    label="Topic"
                    value={activeId}
                    onChange={(e) => onSelectTopic(e.target.value)}
                    sx={{ minWidth: 220 }}
                >
                    {topics.map((topic) => (
                        <MenuItem key={String(topic.topic_id)} value={String(topic.topic_id)}>
                            {topicDisplayName(topic.topic_id)}
                            {prevalence?.[String(topic.topic_id)] != null
                                ? ` · ${formatMetric(num(prevalence[String(topic.topic_id)]))}`
                                : ""}
                        </MenuItem>
                    ))}
                </TextField>
                <TextField
                    size="small"
                    label="Filter documents"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    sx={{ minWidth: 240, flex: 1 }}
                    placeholder="Search title or text"
                />
            </Stack>
            <Typography variant="subtitle2">
                Topic–document weights — {topicDisplayName(selectedTopic?.topic_id)}
            </Typography>
            {weightItems.length ? (
                <RankedBarChart
                    items={weightItems}
                    height={240}
                    valueFormatter={(v) => (v == null ? "" : Number(v).toFixed(3))}
                />
            ) : null}
            <ResearchResultsTable
                rows={filtered.map((unit, index) => ({
                    id: unit.text_unit_id ?? String(index),
                    title: unit.document_title || "Untitled",
                    weight: Number(unit.weight || 0),
                    text: unit.text ?? "",
                }))}
                columns={[
                    { id: "title", label: "Document", value: (row) => row.title },
                    {
                        id: "weight",
                        label: "Topic weight",
                        value: (row) => row.weight,
                        align: "right",
                    },
                    { id: "text", label: "Text", value: (row) => row.text },
                ]}
            />
            <Typography variant="caption" color="text.secondary">
                Showing highest-weight units stored with this run (topic–document distribution
                snapshot). Full corpus distributions remain in the run artifact.
            </Typography>
        </Stack>
    );
}

export function TopicComparisonView({
    runs,
    runA,
    runB,
    selectedAId,
    selectedBId,
    onSelectA,
    onSelectB,
    compare,
    compareLoading,
    compareError,
}: {
    runs: AnalysisRun[];
    runA: AnalysisRun | null | undefined;
    runB: AnalysisRun | null | undefined;
    selectedAId: string;
    selectedBId: string;
    onSelectA: (id: string) => void;
    onSelectB: (id: string) => void;
    compare:
        | {
              parameter_diff: Array<{
                  parameter?: string;
                  run_a: unknown;
                  run_b: unknown;
                  changed: boolean;
              }>;
              metric_diff: Array<{
                  metric?: string;
                  run_a: unknown;
                  run_b: unknown;
                  changed: boolean;
              }>;
          }
        | null
        | undefined;
    compareLoading: boolean;
    compareError: string | null;
}) {
    const trainingRuns = runs.filter((run) => {
        const mode = asRecord(run.parameters)?.mode;
        return run.run_type === "topic_model" && mode !== "k_sweep" && mode !== "seed_stability";
    });

    const topicsA = parseTopics(runA?.results);
    const topicsB = parseTopics(runB?.results);
    const maxTopics = Math.max(topicsA.length, topicsB.length);

    const termPreview = (topic: TopicRow | undefined) =>
        topic?.top_terms
            ?.map((term) => term.term)
            .filter(Boolean)
            .slice(0, 8)
            .join(", ") ?? "—";

    if (!trainingRuns.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                Train at least two topic models to compare runs.
            </Typography>
        );
    }

    return (
        <Stack spacing={2}>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                <TextField
                    select
                    size="small"
                    label="Run A"
                    value={selectedAId}
                    onChange={(e) => onSelectA(e.target.value)}
                    sx={{ minWidth: 280 }}
                >
                    <MenuItem value="">Select run</MenuItem>
                    {trainingRuns.map((run) => (
                        <MenuItem key={run.id} value={run.id}>
                            {run.id.slice(0, 8)} · {String(asRecord(run.parameters)?.algorithm ?? "?")} · K=
                            {String(asRecord(run.parameters)?.n_topics ?? "?")} · {run.status}
                        </MenuItem>
                    ))}
                </TextField>
                <TextField
                    select
                    size="small"
                    label="Run B"
                    value={selectedBId}
                    onChange={(e) => onSelectB(e.target.value)}
                    sx={{ minWidth: 280 }}
                >
                    <MenuItem value="">Select run</MenuItem>
                    {trainingRuns.map((run) => (
                        <MenuItem key={run.id} value={run.id}>
                            {run.id.slice(0, 8)} · {String(asRecord(run.parameters)?.algorithm ?? "?")} · K=
                            {String(asRecord(run.parameters)?.n_topics ?? "?")} · {run.status}
                        </MenuItem>
                    ))}
                </TextField>
            </Stack>

            {compareError ? <Alert severity="error">{compareError}</Alert> : null}
            {compareLoading ? (
                <Typography variant="body2" color="text.secondary">
                    Comparing runs…
                </Typography>
            ) : null}

            {compare ? (
                <Stack spacing={2}>
                    <Typography variant="subtitle2">Changed parameters</Typography>
                    <Box sx={{ overflowX: "auto" }}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Parameter</TableCell>
                                    <TableCell>Run A</TableCell>
                                    <TableCell>Run B</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {compare.parameter_diff
                                    .filter((row) => row.changed)
                                    .map((row) => (
                                        <TableRow key={row.parameter}>
                                            <TableCell>{row.parameter}</TableCell>
                                            <TableCell>{formatCell(row.run_a)}</TableCell>
                                            <TableCell>{formatCell(row.run_b)}</TableCell>
                                        </TableRow>
                                    ))}
                            </TableBody>
                        </Table>
                    </Box>
                    <Typography variant="subtitle2">Metrics</Typography>
                    <Box sx={{ overflowX: "auto" }}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Metric</TableCell>
                                    <TableCell align="right">Run A</TableCell>
                                    <TableCell align="right">Run B</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {compare.metric_diff.map((row) => (
                                    <TableRow key={row.metric}>
                                        <TableCell>{row.metric}</TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(row.run_a))}
                                        </TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(row.run_b))}
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </Box>
                </Stack>
            ) : null}

            {runA?.status === "completed" && runB?.status === "completed" ? (
                <Box sx={{ overflowX: "auto" }}>
                    <Typography variant="subtitle2" gutterBottom>
                        Topic terms side-by-side
                    </Typography>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Topic #</TableCell>
                                <TableCell>Run A terms</TableCell>
                                <TableCell>Run B terms</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {Array.from({ length: maxTopics }, (_, index) => (
                                <TableRow key={index}>
                                    <TableCell>{index}</TableCell>
                                    <TableCell>{termPreview(topicsA[index])}</TableCell>
                                    <TableCell>{termPreview(topicsB[index])}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </Box>
            ) : null}
        </Stack>
    );
}

