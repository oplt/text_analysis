import { useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Stack,
    Tab,
    Tabs,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { BarChart as AnalysisIcon, PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getRun,
    listDictionaries,
    listPreprocessingProfiles,
    runCooccurrence,
    runCorpusStats,
    runDictionaryAnalysis,
    runDfm,
    runFrequencies,
    runKeyness,
    runKwic,
    runNgrams,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import {
    MetricCards,
    RankedBarChart,
    ResultsInspector,
    type RankedItem,
} from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import type { AnalysisRun } from "../types";

type AnalysisTab =
    | "overview"
    | "frequencies"
    | "ngrams"
    | "kwic"
    | "dfm"
    | "keyness"
    | "dictionaries"
    | "cooccurrence";

type DfmWeighting = "count" | "binary" | "tfidf";
type KeynessFilterField = "organization" | "cultural_sphere";

const TABS: Array<{ value: AnalysisTab; label: string }> = [
    { value: "overview", label: "Overview" },
    { value: "frequencies", label: "Frequencies" },
    { value: "ngrams", label: "N-grams" },
    { value: "kwic", label: "KWIC" },
    { value: "dfm", label: "DFM" },
    { value: "keyness", label: "Keyness" },
    { value: "dictionaries", label: "Dictionaries" },
    { value: "cooccurrence", label: "Co-occurrence" },
];

const GROUP_BY_OPTIONS = [
    "organization",
    "organization_type",
    "cultural_sphere",
    "region",
    "language",
    "publication_year",
    "publication_type",
] as const;

function asRecord(value: unknown): Record<string, unknown> | null {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
    }
    return null;
}

function asArray(value: unknown): unknown[] {
    if (Array.isArray(value)) return value;
    const record = asRecord(value);
    if (!record) return [];
    for (const key of [
        "frequencies",
        "ngrams",
        "keyness",
        "cooccurrence",
        "matches",
        "items",
        "rows",
        "results",
        "terms",
    ]) {
        if (Array.isArray(record[key])) return record[key] as unknown[];
    }
    return [];
}

function pickString(row: Record<string, unknown>, keys: string[]): string | null {
    for (const key of keys) {
        const value = row[key];
        if (typeof value === "string" && value.trim()) return value;
        if (typeof value === "number") return String(value);
    }
    return null;
}

function pickNumber(row: Record<string, unknown>, keys: string[]): number | null {
    for (const key of keys) {
        const value = row[key];
        if (typeof value === "number" && Number.isFinite(value)) return value;
        if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value))) {
            return Number(value);
        }
    }
    return null;
}

function extractRankedItems(
    source: unknown,
    labelKeys: string[],
    valueKeys: string[]
): RankedItem[] {
    return asArray(source)
        .map((entry) => {
            const row = asRecord(entry);
            if (!row) return null;
            const label = pickString(row, labelKeys);
            const value = pickNumber(row, valueKeys);
            if (!label || value == null) return null;
            return { label, value };
        })
        .filter((item): item is RankedItem => item != null);
}

function extractCooccurrenceItems(source: unknown): RankedItem[] {
    return asArray(source)
        .map((entry) => {
            const row = asRecord(entry);
            if (!row) return null;
            const joined = pickString(row, ["pair", "label", "terms"]);
            const termA = pickString(row, ["term_a", "termA", "a", "left"]);
            const termB = pickString(row, ["term_b", "termB", "b", "right"]);
            const label = joined ?? (termA && termB ? `${termA} · ${termB}` : termA ?? termB);
            const value = pickNumber(row, ["count", "frequency", "score", "association_score", "pmi"]);
            if (!label || value == null) return null;
            return { label, value };
        })
        .filter((item): item is RankedItem => item != null);
}

function metricNumber(
    sources: Array<Record<string, unknown> | null | undefined>,
    keys: string[]
): number | string | null {
    for (const source of sources) {
        if (!source) continue;
        const value = pickNumber(source, keys);
        if (value != null) return value;
        const nested = asRecord(source.dimensions) ?? asRecord(source.summary);
        if (nested) {
            const nestedValue = pickNumber(nested, keys);
            if (nestedValue != null) return nestedValue;
        }
    }
    return null;
}

function formatMetric(value: number | string | null | undefined, digits = 3): string | number | null {
    if (value == null) return null;
    if (typeof value === "number") {
        return Number.isInteger(value) ? value : Number(value.toFixed(digits));
    }
    return value;
}

function parseCommaTerms(raw: string): string[] {
    return raw
        .split(",")
        .map((term) => term.trim())
        .filter(Boolean);
}

function runPayload(run: AnalysisRun | undefined): unknown {
    if (!run) return null;
    return run.results ?? run.metrics ?? null;
}

function KwicTable({ matches }: { matches: unknown[] }) {
    if (!matches.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No concordance lines in results.
            </Typography>
        );
    }
    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    <TableCell>Left</TableCell>
                    <TableCell>Keyword</TableCell>
                    <TableCell>Right</TableCell>
                    <TableCell>Document</TableCell>
                    <TableCell>Organization</TableCell>
                </TableRow>
            </TableHead>
            <TableBody>
                {matches.slice(0, 100).map((entry, index) => {
                    const row = asRecord(entry) ?? {};
                    return (
                        <TableRow key={index}>
                            <TableCell sx={{ maxWidth: 280 }}>{pickString(row, ["left_context", "left"]) ?? ""}</TableCell>
                            <TableCell>
                                <Typography component="span" fontWeight={600}>
                                    {pickString(row, ["keyword", "match", "term"]) ?? ""}
                                </Typography>
                            </TableCell>
                            <TableCell sx={{ maxWidth: 280 }}>{pickString(row, ["right_context", "right"]) ?? ""}</TableCell>
                            <TableCell>{pickString(row, ["document_title", "title", "document"]) ?? "—"}</TableCell>
                            <TableCell>{pickString(row, ["organization"]) ?? "—"}</TableCell>
                        </TableRow>
                    );
                })}
            </TableBody>
        </Table>
    );
}

function DfmPreviewTable({ preview }: { preview: Record<string, unknown> }) {
    const values = Array.isArray(preview.values) ? (preview.values as unknown[]) : [];
    const featureNames = Array.isArray(preview.feature_names)
        ? (preview.feature_names as unknown[]).map(String)
        : [];
    if (!values.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No DFM preview matrix in results.
            </Typography>
        );
    }
    return (
        <Table size="small">
            <TableHead>
                <TableRow>
                    <TableCell>Unit</TableCell>
                    {featureNames.map((name) => (
                        <TableCell key={name} align="right">
                            {name}
                        </TableCell>
                    ))}
                </TableRow>
            </TableHead>
            <TableBody>
                {values.slice(0, 10).map((row, rowIndex) => {
                    const cells = Array.isArray(row) ? row : [];
                    return (
                        <TableRow key={rowIndex}>
                            <TableCell>{rowIndex + 1}</TableCell>
                            {cells.map((cell, colIndex) => (
                                <TableCell key={colIndex} align="right">
                                    {typeof cell === "number" ? formatMetric(cell) : String(cell ?? "")}
                                </TableCell>
                            ))}
                        </TableRow>
                    );
                })}
            </TableBody>
        </Table>
    );
}

function AnalysisResults({ tab, run }: { tab: AnalysisTab; run: AnalysisRun }) {
    const results = asRecord(run.results);
    const metrics = asRecord(run.metrics);
    const summary = asRecord(results?.summary) ?? asRecord(metrics?.summary);
    const payload = runPayload(run);

    if (isActiveRunStatus(run.status) && !results && !metrics) {
        return (
            <Typography color="text.secondary">
                Run in progress…
            </Typography>
        );
    }

    if (run.status === "failed") {
        return (
            <Alert severity="error">{run.error_message || "Analysis run failed."}</Alert>
        );
    }

    if (tab === "overview") {
        const breakdowns = asRecord(results?.breakdowns);
        const orgItems = Object.entries(asRecord(breakdowns?.by_organization) ?? {}).map(
            ([label, value]) => ({ label, value: Number(value) || 0 })
        );
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        {
                            label: "Documents / units",
                            value: formatMetric(
                                metricNumber([metrics, results], ["document_count", "unit_count", "n_units"])
                            ),
                        },
                        {
                            label: "Tokens",
                            value: formatMetric(metricNumber([metrics, results], ["token_count", "total_tokens"])),
                        },
                        {
                            label: "Vocabulary",
                            value: formatMetric(
                                metricNumber([metrics, results], ["vocabulary_size", "vocab", "vocab_size"])
                            ),
                        },
                        {
                            label: "Mean length",
                            value: formatMetric(metricNumber([metrics, results], ["mean_length", "avg_length"])),
                        },
                        {
                            label: "Median length",
                            value: formatMetric(metricNumber([metrics, results], ["median_length"])),
                        },
                        {
                            label: "Min / max length",
                            value: (() => {
                                const min = metricNumber([metrics, results], ["min_length"]);
                                const max = metricNumber([metrics, results], ["max_length"]);
                                if (min == null && max == null) return null;
                                return `${min ?? "—"} / ${max ?? "—"}`;
                            })(),
                        },
                    ]}
                />
                {orgItems.length > 0 ? (
                    <Stack spacing={1}>
                        <Typography variant="subtitle2">Units by organization</Typography>
                        <RankedBarChart items={orgItems} />
                    </Stack>
                ) : null}
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "frequencies") {
        const items = extractRankedItems(results?.frequencies ?? payload, ["term", "token", "feature", "ngram"], [
            "raw_count",
            "count",
            "frequency",
            "score",
        ]);
        return (
            <Stack spacing={2}>
                <RankedBarChart items={items} />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "ngrams") {
        const items = extractRankedItems(results?.ngrams ?? payload, ["ngram", "term", "token", "feature"], [
            "raw_count",
            "count",
            "frequency",
            "score",
        ]);
        return (
            <Stack spacing={2}>
                <RankedBarChart items={items} />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "kwic") {
        const matches = asArray(results?.matches ?? payload);
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        {
                            label: "Matches",
                            value: formatMetric(
                                metricNumber([metrics, results], ["match_count"]) ?? matches.length
                            ),
                        },
                        {
                            label: "Units searched",
                            value: formatMetric(metricNumber([metrics, results], ["unit_count"])),
                        },
                    ]}
                />
                <KwicTable matches={matches} />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "dfm") {
        const unitCount = metricNumber([summary, metrics, results], ["unit_count", "units", "n_units"]);
        const featureCount = metricNumber([summary, metrics, results], [
            "feature_count",
            "features",
            "n_features",
        ]);
        const density = metricNumber([summary, metrics, results], ["density"]);
        const sparsity =
            metricNumber([summary, metrics, results], ["sparsity"]) ??
            (typeof density === "number" ? 1 - density : null);
        const preview =
            asRecord(results?.preview) ??
            asRecord(summary?.preview) ??
            asRecord(asRecord(results?.summary)?.preview);
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        { label: "Units", value: formatMetric(unitCount) },
                        { label: "Features", value: formatMetric(featureCount) },
                        { label: "Density", value: formatMetric(density) },
                        { label: "Sparsity", value: formatMetric(sparsity) },
                    ]}
                />
                {preview ? <DfmPreviewTable preview={preview} /> : null}
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "keyness") {
        const items = extractRankedItems(results?.keyness ?? payload, ["feature", "term", "token", "ngram"], [
            "keyness_statistic",
            "keyness",
            "score",
            "g2",
            "count",
        ]);
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        {
                            label: "Group A units",
                            value: formatMetric(metricNumber([metrics, results], ["unit_count_a"])),
                        },
                        {
                            label: "Group B units",
                            value: formatMetric(metricNumber([metrics, results], ["unit_count_b"])),
                        },
                        {
                            label: "Features",
                            value: formatMetric(
                                metricNumber([metrics, results], ["features_returned"]) ?? items.length
                            ),
                        },
                    ]}
                />
                <RankedBarChart items={items} />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "dictionaries") {
        const byGroup = asRecord(results?.by_group);
        const groupItems = Object.entries(byGroup ?? {}).map(([label, value]) => {
            const bucket = asRecord(value);
            const hits = bucket ? pickNumber(bucket, ["hits", "count"]) : Number(value) || 0;
            return { label, value: hits ?? 0 };
        });
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        {
                            label: "Total hits",
                            value: formatMetric(
                                metricNumber([metrics, results], ["total_hits", "hits"])
                            ),
                        },
                        {
                            label: "Hits / 1000 tokens",
                            value: formatMetric(
                                metricNumber([metrics, results], [
                                    "hits_per_1000_tokens",
                                    "per_1000",
                                ])
                            ),
                        },
                        {
                            label: "Document prevalence",
                            value: formatMetric(
                                metricNumber([metrics, results], [
                                    "document_prevalence",
                                    "unit_prevalence",
                                ])
                            ),
                        },
                    ]}
                />
                {groupItems.length > 0 ? (
                    <Stack spacing={1}>
                        <Typography variant="subtitle2">Hits by group</Typography>
                        <RankedBarChart items={groupItems} />
                    </Stack>
                ) : null}
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    const items = extractCooccurrenceItems(results?.cooccurrence ?? payload);
    return (
        <Stack spacing={2}>
            <RankedBarChart items={items} />
            <ResultsInspector data={payload} />
        </Stack>
    );
}

export default function AnalysisView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();

    const [tab, setTab] = useState<AnalysisTab>("overview");
    const [runId, setRunId] = useState<string | null>(null);

    const [profileId, setProfileId] = useState("");
    const [topN, setTopN] = useState(50);
    const [ngramN, setNgramN] = useState(2);
    const [kwicKeyword, setKwicKeyword] = useState("");
    const [kwicWindow, setKwicWindow] = useState(5);
    const [kwicCaseSensitive, setKwicCaseSensitive] = useState(false);
    const [dfmWeighting, setDfmWeighting] = useState<DfmWeighting>("count");
    const [keynessField, setKeynessField] = useState<KeynessFilterField>("organization");
    const [keynessA, setKeynessA] = useState("");
    const [keynessB, setKeynessB] = useState("");
    const [dictionaryId, setDictionaryId] = useState("");
    const [dictionaryTerms, setDictionaryTerms] = useState("");
    const [groupBy, setGroupBy] = useState("");
    const [coocWindow, setCoocWindow] = useState(5);

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const dictionariesQuery = useQuery({
        queryKey: queryKeys.textResearch.dictionaries(ctx.projectId),
        queryFn: () => listDictionaries(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: activeRunRefetchInterval,
    });

    const basePayload = {
        unit_type: ctx.unitType,
        ...(profileId ? { preprocessing_profile_id: profileId } : {}),
    };

    const onRunSuccess = (run: AnalysisRun, message: string) => {
        setRunId(run.id);
        showToast({ message, severity: "success" });
    };

    const onRunError = (error: unknown, fallback: string) => {
        showToast({ message: getQueryErrorMessage(error, fallback), severity: "error" });
    };

    const overviewMutation = useMutation({
        mutationFn: () => runCorpusStats(ctx.selectedCorpusId, basePayload),
        onSuccess: (run) => onRunSuccess(run, "Corpus overview started."),
        onError: (error) => onRunError(error, "Failed to run corpus overview."),
    });

    const frequenciesMutation = useMutation({
        mutationFn: () =>
            runFrequencies(ctx.selectedCorpusId, {
                ...basePayload,
                top_n: topN,
            }),
        onSuccess: (run) => onRunSuccess(run, "Frequency analysis started."),
        onError: (error) => onRunError(error, "Failed to run frequencies."),
    });

    const ngramsMutation = useMutation({
        mutationFn: () =>
            runNgrams(ctx.selectedCorpusId, {
                ...basePayload,
                n: ngramN,
                top_n: topN,
            }),
        onSuccess: (run) => onRunSuccess(run, "N-gram analysis started."),
        onError: (error) => onRunError(error, "Failed to run n-grams."),
    });

    const kwicMutation = useMutation({
        mutationFn: () =>
            runKwic(ctx.selectedCorpusId, {
                ...basePayload,
                keyword: kwicKeyword.trim(),
                window_size: kwicWindow,
                case_sensitive: kwicCaseSensitive,
            }),
        onSuccess: (run) => onRunSuccess(run, "KWIC search started."),
        onError: (error) => onRunError(error, "Failed to run KWIC."),
    });

    const dfmMutation = useMutation({
        mutationFn: () =>
            runDfm(ctx.selectedCorpusId, {
                ...basePayload,
                weighting: dfmWeighting,
            }),
        onSuccess: (run) => onRunSuccess(run, "DFM build started."),
        onError: (error) => onRunError(error, "Failed to build DFM."),
    });

    const keynessMutation = useMutation({
        mutationFn: () =>
            runKeyness(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                ...(profileId ? { preprocessing_profile_id: profileId } : {}),
                filters_a: { [keynessField]: keynessA.trim() },
                filters_b: { [keynessField]: keynessB.trim() },
                top_n: topN,
            }),
        onSuccess: (run) => onRunSuccess(run, "Keyness comparison started."),
        onError: (error) => onRunError(error, "Failed to run keyness."),
    });

    const dictionaryMutation = useMutation({
        mutationFn: () => {
            const selected = dictionariesQuery.data?.find((d) => d.id === dictionaryId);
            const terms = parseCommaTerms(dictionaryTerms);
            const resolvedTerms = terms.length > 0 ? terms : (selected?.terms ?? []);
            return runDictionaryAnalysis(ctx.selectedCorpusId, {
                ...basePayload,
                ...(dictionaryId ? { dictionary_id: dictionaryId } : {}),
                ...(resolvedTerms.length ? { dictionary_terms: resolvedTerms } : {}),
                ...(groupBy ? { group_by: groupBy } : {}),
            });
        },
        onSuccess: (run) => onRunSuccess(run, "Dictionary analysis started."),
        onError: (error) => onRunError(error, "Failed to run dictionary analysis."),
    });

    const cooccurrenceMutation = useMutation({
        mutationFn: () =>
            runCooccurrence(ctx.selectedCorpusId, {
                ...basePayload,
                window_size: coocWindow,
                top_n: topN,
            }),
        onSuccess: (run) => onRunSuccess(run, "Co-occurrence analysis started."),
        onError: (error) => onRunError(error, "Failed to run co-occurrence."),
    });

    const mutationByTab = {
        overview: overviewMutation,
        frequencies: frequenciesMutation,
        ngrams: ngramsMutation,
        kwic: kwicMutation,
        dfm: dfmMutation,
        keyness: keynessMutation,
        dictionaries: dictionaryMutation,
        cooccurrence: cooccurrenceMutation,
    } as const;

    const activeMutation = mutationByTab[tab];
    const selectedDictionary = dictionariesQuery.data?.find((d) => d.id === dictionaryId);
    const resolvedDictionaryTerms = parseCommaTerms(dictionaryTerms);
    const hasDictionaryInput =
        resolvedDictionaryTerms.length > 0 || Boolean(selectedDictionary?.terms?.length);

    const canRun = (() => {
        if (!ctx.selectedCorpusId || activeMutation.isPending) return false;
        if (tab === "kwic") return Boolean(kwicKeyword.trim());
        if (tab === "keyness") return Boolean(keynessA.trim() && keynessB.trim());
        if (tab === "dictionaries") return hasDictionaryInput;
        return true;
    })();

    const runLabel = TABS.find((entry) => entry.value === tab)?.label ?? "analysis";

    if (!ctx.selectedCorpusId) {
        return (
            <SectionCard title="Quantitative analysis" description="Run corpus statistics and lexical analyses.">
                <EmptyState
                    icon={<AnalysisIcon fontSize="large" />}
                    title="Select a corpus"
                    description="Choose a corpus in the context bar, then run overview stats, frequencies, KWIC, and related tools."
                />
            </SectionCard>
        );
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Quantitative analysis"
                description="Expose corpus statistics, frequencies, n-grams, KWIC, DFM, keyness, dictionaries, and co-occurrence with charts and inspectable raw results."
            >
                <Tabs
                    value={tab}
                    onChange={(_, value: AnalysisTab) => setTab(value)}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{ mb: 2 }}
                >
                    {TABS.map((entry) => (
                        <Tab key={entry.value} label={entry.label} value={entry.value} />
                    ))}
                </Tabs>

                <Stack spacing={2}>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                        <TextField
                            select
                            size="small"
                            label="Preprocessing profile"
                            value={profileId}
                            onChange={(event) => setProfileId(event.target.value)}
                            sx={{ minWidth: 220 }}
                        >
                            <MenuItem value="">Default / none</MenuItem>
                            {(profilesQuery.data ?? []).map((profile) => (
                                <MenuItem key={profile.id} value={profile.id}>
                                    {profile.name}
                                </MenuItem>
                            ))}
                        </TextField>
                        <Typography variant="body2" color="text.secondary" sx={{ alignSelf: "center" }}>
                            Unit type: {ctx.unitType}
                        </Typography>
                        {tab === "frequencies" ||
                        tab === "ngrams" ||
                        tab === "keyness" ||
                        tab === "cooccurrence" ? (
                            <TextField
                                size="small"
                                type="number"
                                label="Top N"
                                value={topN}
                                onChange={(event) => setTopN(Math.max(1, Number(event.target.value) || 1))}
                                inputProps={{ min: 1, max: 500 }}
                                sx={{ width: 120 }}
                            />
                        ) : null}
                    </Stack>

                    {tab === "ngrams" ? (
                        <TextField
                            size="small"
                            type="number"
                            label="N"
                            value={ngramN}
                            onChange={(event) => setNgramN(Math.max(1, Number(event.target.value) || 1))}
                            inputProps={{ min: 1, max: 5 }}
                            sx={{ width: 120 }}
                        />
                    ) : null}

                    {tab === "kwic" ? (
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center">
                            <TextField
                                size="small"
                                label="Keyword"
                                value={kwicKeyword}
                                onChange={(event) => setKwicKeyword(event.target.value)}
                                sx={{ minWidth: 220 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Window size"
                                value={kwicWindow}
                                onChange={(event) =>
                                    setKwicWindow(Math.max(1, Number(event.target.value) || 1))
                                }
                                inputProps={{ min: 1, max: 20 }}
                                sx={{ width: 140 }}
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={kwicCaseSensitive}
                                        onChange={(event) => setKwicCaseSensitive(event.target.checked)}
                                    />
                                }
                                label="Case sensitive"
                            />
                        </Stack>
                    ) : null}

                    {tab === "dfm" ? (
                        <TextField
                            select
                            size="small"
                            label="Weighting"
                            value={dfmWeighting}
                            onChange={(event) => setDfmWeighting(event.target.value as DfmWeighting)}
                            sx={{ minWidth: 180 }}
                        >
                            <MenuItem value="count">count</MenuItem>
                            <MenuItem value="binary">binary</MenuItem>
                            <MenuItem value="tfidf">tfidf</MenuItem>
                        </TextField>
                    ) : null}

                    {tab === "keyness" ? (
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                            <TextField
                                select
                                size="small"
                                label="Compare by"
                                value={keynessField}
                                onChange={(event) =>
                                    setKeynessField(event.target.value as KeynessFilterField)
                                }
                                sx={{ minWidth: 180 }}
                            >
                                <MenuItem value="organization">organization</MenuItem>
                                <MenuItem value="cultural_sphere">cultural_sphere</MenuItem>
                            </TextField>
                            <TextField
                                size="small"
                                label="Group A value"
                                value={keynessA}
                                onChange={(event) => setKeynessA(event.target.value)}
                                sx={{ minWidth: 180 }}
                            />
                            <TextField
                                size="small"
                                label="Group B value"
                                value={keynessB}
                                onChange={(event) => setKeynessB(event.target.value)}
                                sx={{ minWidth: 180 }}
                            />
                        </Stack>
                    ) : null}

                    {tab === "dictionaries" ? (
                        <Stack spacing={2}>
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                                <TextField
                                    select
                                    size="small"
                                    label="Dictionary"
                                    value={dictionaryId}
                                    onChange={(event) => setDictionaryId(event.target.value)}
                                    sx={{ minWidth: 240 }}
                                >
                                    <MenuItem value="">Custom terms</MenuItem>
                                    {(dictionariesQuery.data ?? []).map((dictionary) => (
                                        <MenuItem key={dictionary.id} value={dictionary.id}>
                                            {dictionary.name} · {dictionary.terms.length} terms
                                        </MenuItem>
                                    ))}
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    label="Group by (optional)"
                                    value={groupBy}
                                    onChange={(event) => setGroupBy(event.target.value)}
                                    sx={{ minWidth: 200 }}
                                >
                                    <MenuItem value="">None</MenuItem>
                                    {GROUP_BY_OPTIONS.map((option) => (
                                        <MenuItem key={option} value={option}>
                                            {option}
                                        </MenuItem>
                                    ))}
                                </TextField>
                            </Stack>
                            <TextField
                                size="small"
                                label="Comma-separated terms"
                                value={dictionaryTerms}
                                onChange={(event) => setDictionaryTerms(event.target.value)}
                                helperText={
                                    selectedDictionary && !dictionaryTerms.trim()
                                        ? `Using ${selectedDictionary.terms.length} terms from ${selectedDictionary.name}`
                                        : "Overrides the selected dictionary when provided"
                                }
                                fullWidth
                            />
                        </Stack>
                    ) : null}

                    {tab === "cooccurrence" ? (
                        <TextField
                            size="small"
                            type="number"
                            label="Window size"
                            value={coocWindow}
                            onChange={(event) =>
                                setCoocWindow(Math.max(1, Number(event.target.value) || 1))
                            }
                            inputProps={{ min: 1, max: 20 }}
                            sx={{ width: 140 }}
                        />
                    ) : null}

                    <Button
                        variant="contained"
                        startIcon={<RunIcon />}
                        onClick={() => activeMutation.mutate()}
                        disabled={!canRun}
                        sx={{ alignSelf: "flex-start" }}
                    >
                        Run {runLabel}
                    </Button>
                </Stack>
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
                                    {runQuery.data.run_type} —{" "}
                                    <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                <AnalysisResults tab={tab} run={runQuery.data} />
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
