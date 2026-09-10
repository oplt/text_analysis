import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
    Alert,
    Button,
    Checkbox,
    FormControlLabel,
    MenuItem,
    Skeleton,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TablePagination,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import { BarChart as AnalysisIcon, PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getCorpusMetadataFacets,
    getRun,
    listAnalysisEngines,
    listDictionaries,
    listPreprocessingProfiles,
    researchExportUrl,
    runCooccurrence,
    runCorpusStats,
    runDictionaryAnalysis,
    runDfm,
    runEngineComparison,
    runFrequencies,
    runKeyness,
    runKwic,
    runNgrams,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { PageTabs } from "../../../components/ui/PageTabs";
import { useDebounce } from "../../../hooks/useDebounce";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES, researchRunStaleTime } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import {
    MetricCards,
    CooccurrenceNetwork,
    DivergingBarChart,
    RankedBarChart,
    ResultsInspector,
    type RankedItem,
} from "../components/ResearchCharts";
import { ChartTableToggle, ResearchResultPanel, ResearchResultsTable } from "../components/ResearchResults";
import { MetadataFilterBar } from "../components/MetadataFilterBar";
import { AdvancedDfmPanel } from "../components/AdvancedDfmPanel";
import { DEFAULT_DFM_CONFIG, type AdvancedDfmConfig } from "../components/advancedDfmConfig";
import { ScientificWarnings } from "../components/ScientificWarnings";
import { collectScientificWarnings } from "../components/scientificWarnings";
import { useResearchContext } from "../hooks/useResearchContext";
import { useRunEvents } from "../hooks/useRunEvents";
import { filterKwicRows, kwicRowsToCsv, toKwicSearchRows } from "../kwicTableModel";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";
import {
    canonicalAnalysisResults,
    rEngineOptionLabel,
    rEngineSelectionState,
} from "../analysisEngine";
import type { AnalysisRun } from "../types";

const StatisticalModelView = lazy(() => import("./StatisticalModelView"));
const MeasurementComparisonView = lazy(() => import("./MeasurementComparisonView"));
const ClusterExplorer = lazy(() => import("../components/ClusterExplorer"));
const DimensionalityReductionView = lazy(
    () => import("../components/DimensionalityReductionView")
);
const SimilarityExplorer = lazy(() =>
    import("../components/AdvancedAnalysisPanels").then((m) => ({ default: m.SimilarityExplorer }))
);
const DuplicateDetectionView = lazy(() =>
    import("../components/AdvancedAnalysisPanels").then((m) => ({
        default: m.DuplicateDetectionView,
    }))
);
const ReadabilityView = lazy(() =>
    import("../components/AdvancedAnalysisPanels").then((m) => ({ default: m.ReadabilityView }))
);

function AnalysisPanelFallback() {
    return <Skeleton variant="rounded" height={220} sx={{ borderRadius: 2 }} />;
}

type AnalysisTab =
    | "overview"
    | "frequencies"
    | "ngrams"
    | "kwic"
    | "dfm"
    | "keyness"
    | "dictionaries"
    | "cooccurrence"
    | "similarity"
    | "duplicates"
    | "clustering"
    | "dimensionality"
    | "readability"
    | "statistical"
    | "measurement";

const ANALYSIS_TAB_VALUES = [
    "overview",
    "frequencies",
    "ngrams",
    "kwic",
    "dfm",
    "keyness",
    "dictionaries",
    "cooccurrence",
    "similarity",
    "duplicates",
    "clustering",
    "dimensionality",
    "readability",
    "statistical",
    "measurement",
] as const satisfies readonly AnalysisTab[];

const TABS: Array<{ value: AnalysisTab; label: string }> = [
    { value: "overview", label: "Overview" },
    { value: "frequencies", label: "Frequencies" },
    { value: "ngrams", label: "N-grams" },
    { value: "kwic", label: "KWIC" },
    { value: "dfm", label: "DFM" },
    { value: "keyness", label: "Keyness" },
    { value: "dictionaries", label: "Dictionaries" },
    { value: "cooccurrence", label: "Co-occurrence" },
    { value: "similarity", label: "Similarity" },
    { value: "duplicates", label: "Duplicates" },
    { value: "clustering", label: "Clustering" },
    { value: "dimensionality", label: "Dimensions" },
    { value: "readability", label: "Readability" },
    { value: "statistical", label: "Statistical model" },
    { value: "measurement", label: "Measurement" },
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

function optionalNumber(value: string): number | undefined {
    if (!value.trim()) return undefined;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
}

function dfmTrimPayload(config: AdvancedDfmConfig) {
    const minTermFrequency = optionalNumber(config.minTermFrequency);
    const maxTermFrequency = optionalNumber(config.maxTermFrequency);
    const minDocumentFrequency = optionalNumber(config.minDocumentFrequency);
    const maxDocumentFrequency = optionalNumber(config.maxDocumentFrequency);
    const topN = optionalNumber(config.topN);
    if (
        minTermFrequency == null && maxTermFrequency == null &&
        minDocumentFrequency == null && maxDocumentFrequency == null && topN == null
    ) return undefined;
    return {
        ...(minTermFrequency != null ? { min_term_frequency: minTermFrequency } : {}),
        ...(maxTermFrequency != null ? { max_term_frequency: maxTermFrequency } : {}),
        term_frequency_type: config.termFrequencyType,
        ...(minDocumentFrequency != null ? { min_document_frequency: minDocumentFrequency } : {}),
        ...(maxDocumentFrequency != null ? { max_document_frequency: maxDocumentFrequency } : {}),
        document_frequency_type: config.documentFrequencyType,
        ...(topN != null ? { top_n: topN } : {}),
    };
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

/**
 * Small client-side CSV for preview tables.
 * Prefer backend run/corpus exports for large artifacts; if a heavier client
 * transform is required, move it to a Web Worker rather than blocking the UI.
 */
function downloadCsv(filename: string, rows: unknown[]) {
    const records = rows.map(asRecord).filter((row): row is Record<string, unknown> => row != null);
    if (!records.length) return;
    const headers = Array.from(new Set(records.flatMap((row) => Object.keys(row))));
    const csv = [headers.join(","), ...records.map((row) => headers.map((header) => JSON.stringify(row[header] ?? "")).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

function runPayload(run: AnalysisRun | undefined): unknown {
    if (!run) return null;
    return run.results ?? run.metrics ?? null;
}

const KWIC_PAGE_SIZE = 50;

function KwicTable({ matches, runId }: { matches: unknown[]; runId?: string | null }) {
    const [filter, setFilter] = useState("");
    const [page, setPage] = useState(0);
    const debouncedFilter = useDebounce(filter, 200);
    const searchableRows = useMemo(() => toKwicSearchRows(matches), [matches]);
    const filteredRows = useMemo(
        () => filterKwicRows(searchableRows, debouncedFilter),
        [searchableRows, debouncedFilter]
    );
    const pageCount = Math.max(1, Math.ceil(filteredRows.length / KWIC_PAGE_SIZE));
    const currentPage = Math.min(page, pageCount - 1);
    const visibleRows = filteredRows.slice(
        currentPage * KWIC_PAGE_SIZE,
        (currentPage + 1) * KWIC_PAGE_SIZE
    );

    function exportClientCsv() {
        const blob = new Blob([kwicRowsToCsv(filteredRows)], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "kwic-results.csv";
        anchor.click();
        URL.revokeObjectURL(url);
    }

    if (!matches.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No concordance lines in results.
            </Typography>
        );
    }
    return (
        <Stack spacing={1}>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                <TextField
                    size="small"
                    label="Filter concordance"
                    value={filter}
                    onChange={(event) => {
                        setFilter(event.target.value);
                        setPage(0);
                    }}
                    sx={{ minWidth: 220 }}
                />
                {runId ? (
                    <Button
                        size="small"
                        variant="outlined"
                        href={researchExportUrl(`/research/runs/${runId}/export.json`)}
                        target="_blank"
                        rel="noopener"
                    >
                        Backend export
                    </Button>
                ) : null}
                <Button size="small" variant="outlined" onClick={exportClientCsv}>
                    Export filtered CSV
                </Button>
            </Stack>
            {filteredRows.length > KWIC_PAGE_SIZE ? (
                <Typography variant="caption" color="text.secondary">
                    Large result sets: prefer backend export. Client CSV is for the current filter
                    only (consider a Web Worker if transforms grow heavier).
                </Typography>
            ) : null}
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
                    {visibleRows.map((row, index) => (
                        <TableRow key={`${currentPage}-${index}-${row.keyword}`}>
                            <TableCell sx={{ maxWidth: 280 }}>{row.left}</TableCell>
                            <TableCell>
                                <Typography component="span" fontWeight={600}>
                                    {row.keyword}
                                </Typography>
                            </TableCell>
                            <TableCell sx={{ maxWidth: 280 }}>{row.right}</TableCell>
                            <TableCell>{row.document || "—"}</TableCell>
                            <TableCell>{row.organization || "—"}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
            <TablePagination
                component="div"
                count={filteredRows.length}
                page={currentPage}
                onPageChange={(_, next) => setPage(next)}
                rowsPerPage={KWIC_PAGE_SIZE}
                rowsPerPageOptions={[KWIC_PAGE_SIZE]}
            />
        </Stack>
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
    const [display, setDisplay] = useState<"chart" | "table" | "both">("both");
    const [networkLimit, setNetworkLimit] = useState<20 | 50>(20);
    const [minimumEdgeStrength, setMinimumEdgeStrength] = useState(1);
    const results = canonicalAnalysisResults(run.results);
    const metrics = asRecord(run.metrics);
    const summary = asRecord(results?.summary) ?? asRecord(metrics?.summary);
    const payload = runPayload(run);
    const comparisonEnvelope = asRecord(run.results);
    const comparison = asRecord(comparisonEnvelope?.comparison);

    if (isActiveRunStatus(run.status) && !results && !metrics && !comparison) {
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

    if (run.run_type === "engine_comparison" || comparison) {
        return (
            <Stack spacing={2}>
                <Alert severity="info">
                    Compared independent Python and R runs
                    {comparisonEnvelope?.python_run_id
                        ? ` (Python ${String(comparisonEnvelope.python_run_id)}`
                        : ""}
                    {comparisonEnvelope?.r_run_id
                        ? `${comparisonEnvelope?.python_run_id ? ", " : " ("}R ${String(comparisonEnvelope.r_run_id)})`
                        : comparisonEnvelope?.python_run_id
                          ? ")"
                          : ""}
                    .
                </Alert>
                <MetricCards
                    items={[
                        {
                            label: "Analysis",
                            value: String(comparison?.analysis_type ?? tab),
                        },
                        {
                            label: "Equal",
                            value:
                                comparison?.matches_equal === true ||
                                comparison?.count_equal === true ||
                                comparison?.cells_equal === true
                                    ? "yes"
                                    : comparison?.matches_equal === false ||
                                        comparison?.count_equal === false ||
                                        comparison?.cells_equal === false
                                      ? "no"
                                      : "—",
                        },
                        {
                            label: "Python run",
                            value: String(comparisonEnvelope?.python_run_id ?? "—"),
                        },
                        {
                            label: "R run",
                            value: String(comparisonEnvelope?.r_run_id ?? "—"),
                        },
                    ]}
                />
                <ResultsInspector data={comparisonEnvelope ?? {}} />
            </Stack>
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
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                    <ChartTableToggle value={display} onChange={setDisplay} />
                    <Button size="small" variant="outlined" onClick={() => downloadCsv("frequencies.csv", asArray(results?.frequencies ?? payload))}>Export CSV</Button>
                </Stack>
                {display !== "table" ? <RankedBarChart items={items} /> : null}
                {display !== "chart" ? <ResearchResultsTable
                    rows={asArray(results?.frequencies ?? payload).map((entry, index) => ({
                        ...(asRecord(entry) ?? {}),
                        id: index,
                        rank: index + 1,
                    }))}
                    columns={[
                        { id: "rank", label: "Rank", value: (row) => row.rank, align: "right" },
                        { id: "term", label: "Term", value: (row) => pickString(row, ["term", "token", "feature"]) },
                        { id: "count", label: "Count", value: (row) => pickNumber(row, ["raw_count", "count"]), align: "right" },
                        { id: "relative", label: "Share", value: (row) => pickNumber(row, ["relative_frequency", "frequency"]), align: "right" },
                    ]}
                /> : null}
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
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                    <ChartTableToggle value={display} onChange={setDisplay} />
                    <Button size="small" variant="outlined" onClick={() => downloadCsv("ngrams.csv", asArray(results?.ngrams ?? payload))}>Export CSV</Button>
                </Stack>
                {display !== "table" ? <RankedBarChart items={items} /> : null}
                {display !== "chart" ? <ResearchResultsTable
                    rows={asArray(results?.ngrams ?? payload).map((entry, index) => ({ ...(asRecord(entry) ?? {}), id: index, rank: index + 1 }))}
                    columns={[
                        { id: "rank", label: "Rank", value: (row) => row.rank, align: "right" },
                        { id: "ngram", label: "N-gram", value: (row) => pickString(row, ["ngram", "term"]) },
                        { id: "n", label: "N", value: (row) => pickNumber(row, ["n"]), align: "right" },
                        { id: "count", label: "Count", value: (row) => pickNumber(row, ["raw_count", "count"]), align: "right" },
                        { id: "relative", label: "Share", value: (row) => pickNumber(row, ["relative_frequency", "frequency"]), align: "right" },
                    ]}
                /> : null}
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
                <KwicTable matches={matches} runId={run.id} />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "dfm") {
        const unitCount = metricNumber([summary, metrics, results], ["unit_count", "units", "n_units", "documents"]);
        const featureCount = metricNumber([summary, metrics, results], [
            "feature_count",
            "features",
            "n_features",
        ]);
        const density = metricNumber([summary, metrics, results], ["density"]);
        const sparsity =
            metricNumber([summary, metrics, results], ["sparsity"]) ??
            (typeof density === "number" ? 1 - density : null);
        const nnz = metricNumber([summary, metrics, results], ["nnz", "non_zero_cells"]);
        const memoryBytes = metricNumber([summary, metrics, results], [
            "estimated_memory_bytes",
        ]);
        const preview =
            asRecord(results?.preview) ??
            asRecord(summary?.preview) ??
            asRecord(asRecord(results?.summary)?.preview);
        const dfmWarnings = collectScientificWarnings(summary, metrics, results);
        return (
            <Stack spacing={2}>
                <MetricCards
                    items={[
                        { label: "Documents", value: formatMetric(unitCount) },
                        { label: "Features", value: formatMetric(featureCount) },
                        { label: "Non-zero cells", value: formatMetric(nnz, 0) },
                        { label: "Density", value: formatMetric(density) },
                        { label: "Sparsity", value: formatMetric(sparsity) },
                        {
                            label: "Est. memory",
                            value:
                                typeof memoryBytes !== "number"
                                    ? "—"
                                    : `${(memoryBytes / (1024 * 1024)).toFixed(1)} MiB`,
                        },
                    ]}
                />
                <ScientificWarnings
                    title="DFM diagnostics are review signals, not automatic quality gates."
                    warnings={dfmWarnings}
                />
                {preview ? <DfmPreviewTable preview={preview} /> : null}
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "keyness") {
        const keynessResults = results?.keyness ?? results?.features ?? payload;
        const items = extractRankedItems(keynessResults, ["feature", "term", "token", "ngram"], [
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
                <DivergingBarChart
                    items={asArray(keynessResults).map((entry) => {
                        const row = asRecord(entry) ?? {};
                        const score = pickNumber(row, ["keyness_statistic", "keyness", "g2"]) ?? 0;
                        return {
                            label: pickString(row, ["feature", "term"]) ?? "",
                            value: pickString(row, ["effect_direction", "direction"]) === "a" ? -score : score,
                        };
                    })}
                />
                <ResearchResultsTable
                    rows={asArray(keynessResults).map((entry, index) => ({ ...(asRecord(entry) ?? {}), id: index }))}
                    columns={[
                        { id: "feature", label: "Feature", value: (row) => pickString(row, ["feature", "term"]) },
                        { id: "a", label: "Group A", value: (row) => pickNumber(row, ["freq_a", "count_a"]), align: "right" },
                        { id: "b", label: "Group B", value: (row) => pickNumber(row, ["freq_b", "count_b"]), align: "right" },
                        { id: "keyness", label: "Keyness", value: (row) => pickNumber(row, ["keyness_statistic", "keyness", "g2"]), align: "right" },
                        { id: "p", label: "p", value: (row) => pickNumber(row, ["p_value", "p"]), align: "right" },
                        { id: "padj", label: "p (BH)", value: (row) => pickNumber(row, ["p_adjusted", "p_value_adjusted"]), align: "right" },
                        { id: "log_ratio", label: "Log ratio", value: (row) => pickNumber(row, ["log_ratio", "effect_size"]), align: "right" },
                        { id: "direction", label: "Direction", value: (row) => pickString(row, ["effect_direction", "direction"]) },
                    ]}
                />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    if (tab === "dictionaries") {
        const byGroup = asRecord(results?.by_group) ?? asRecord(results?.by_category);
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
                <ResearchResultsTable
                    rows={((Object.entries(byGroup ?? {}).length
                        ? Object.entries(byGroup ?? {})
                        : [["All units", { hits: results?.total_hits, units: undefined }]]) as Array<
                        [string, unknown]
                    >)
                        .map(([group, value], index) => {
                        const bucket = asRecord(value) ?? {};
                        const hits = pickNumber(bucket, ["hits", "count"]) ?? 0;
                        const grouped = Object.keys(byGroup ?? {}).length > 0;
                        return {
                            id: index,
                            group,
                            hits,
                            normalized: grouped
                                ? null
                                : metricNumber([metrics, results], ["hits_per_1000_tokens", "per_1000"]),
                            prevalence: grouped
                                ? null
                                : metricNumber([metrics, results], ["document_prevalence", "unit_prevalence"]),
                        };
                    })}
                    columns={[
                        { id: "dictionary", label: "Dictionary", value: () => pickString(run.parameters ?? {}, ["dictionary_id"]) ?? "Custom terms" },
                        { id: "group", label: "Group", value: (row) => row.group },
                        { id: "hits", label: "Hits", value: (row) => row.hits, align: "right" },
                        { id: "normalized", label: "Normalized hits", value: (row) => row.normalized, align: "right" },
                        { id: "prevalence", label: "Prevalence", value: (row) => row.prevalence, align: "right" },
                    ]}
                />
                <ResultsInspector data={payload} />
            </Stack>
        );
    }

    const networkPayload = asRecord(results?.network) ?? asRecord(asRecord(results?.report)?.network);
    const networkEdges = (
        asArray(networkPayload?.edges).length
            ? asArray(networkPayload?.edges)
            : asArray(results?.cooccurrence ?? results?.pairs ?? payload)
    )
        .map((entry) => asRecord(entry))
        .filter((row): row is Record<string, unknown> => row != null)
        .map((row) => ({
            termA: pickString(row, ["term_a", "termA", "a", "source"]) ?? "",
            termB: pickString(row, ["term_b", "termB", "b", "target"]) ?? "",
            count: pickNumber(row, ["count", "frequency"]) ?? 0,
            association: pickNumber(row, ["weight", "association_score", "pmi", "score"]) ?? 0,
        }))
        .filter((edge) => edge.termA && edge.termB && edge.count >= minimumEdgeStrength)
        .slice(0, networkLimit);
    return (
        <Stack spacing={2}>
            <MetricCards
                items={[
                    {
                        label: "Nodes",
                        value: formatMetric(
                            pickNumber(networkPayload ?? {}, ["node_count"]) ??
                                new Set(networkEdges.flatMap((e) => [e.termA, e.termB])).size
                        ),
                    },
                    {
                        label: "Edges",
                        value: formatMetric(
                            pickNumber(networkPayload ?? {}, ["edge_count"]) ?? networkEdges.length
                        ),
                    },
                ]}
            />
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                <Button size="small" variant={networkLimit === 20 ? "contained" : "outlined"} onClick={() => setNetworkLimit(20)}>Top 20</Button>
                <Button size="small" variant={networkLimit === 50 ? "contained" : "outlined"} onClick={() => setNetworkLimit(50)}>Top 50</Button>
                <TextField size="small" type="number" label="Minimum edge count" value={minimumEdgeStrength} onChange={(event) => setMinimumEdgeStrength(Math.max(1, Number(event.target.value) || 1))} sx={{ width: 180 }} />
            </Stack>
            <CooccurrenceNetwork edges={networkEdges} />
            <ResearchResultsTable
                rows={asArray(results?.cooccurrence ?? results?.pairs ?? payload).map((entry, index) => ({ ...(asRecord(entry) ?? {}), id: index }))}
                columns={[
                    { id: "termA", label: "Term A", value: (row) => pickString(row, ["term_a", "termA", "a"]) },
                    { id: "termB", label: "Term B", value: (row) => pickString(row, ["term_b", "termB", "b"]) },
                    { id: "count", label: "Count", value: (row) => pickNumber(row, ["count", "frequency"]), align: "right" },
                    { id: "association", label: "Association", value: (row) => pickNumber(row, ["association_score", "pmi", "score"]), align: "right" },
                    { id: "pmi", label: "PMI", value: (row) => pickNumber(row, ["pmi"]), align: "right" },
                    { id: "npmi", label: "NPMI", value: (row) => pickNumber(row, ["npmi"]), align: "right" },
                    { id: "dice", label: "Dice", value: (row) => pickNumber(row, ["dice", "log_dice"]), align: "right" },
                    { id: "method", label: "Method", value: (row) => pickString(row, ["association_method", "method"]) },
                ]}
            />
            <ResultsInspector data={payload} />
        </Stack>
    );
}

export default function AnalysisView() {
    const ctx = useResearchContext();
    const navigate = useNavigate();
    const { showToast } = useSnackbar();

    const [tab, setTab] = useTabQueryParam(ANALYSIS_TAB_VALUES, "overview");
    const [runId, setRunId] = useState<string | null>(null);
    const sseConnected = useRunEvents(runId, ctx.projectId);

    const [profileId, setProfileId] = useState("");
    const [topN, setTopN] = useState(50);
    const [ngramN, setNgramN] = useState(2);
    const [kwicKeyword, setKwicKeyword] = useState("");
    const [kwicWindow, setKwicWindow] = useState(5);
    const [kwicCaseSensitive, setKwicCaseSensitive] = useState(false);
    const [kwicQueryMode, setKwicQueryMode] = useState("auto");
    const [dfmConfig, setDfmConfig] = useState<AdvancedDfmConfig>(DEFAULT_DFM_CONFIG);
    const [keynessField, setKeynessField] = useState("organization");
    const [keynessA, setKeynessA] = useState("");
    const [keynessB, setKeynessB] = useState("");
    const [keynessMethod, setKeynessMethod] = useState("log_likelihood");
    const [keynessCorrection, setKeynessCorrection] = useState("bh");
    const [dictionaryId, setDictionaryId] = useState("");
    const [dictionaryTerms, setDictionaryTerms] = useState("");
    const [groupBy, setGroupBy] = useState("");
    const [coocWindow, setCoocWindow] = useState(5);
    const [coocMethod, setCoocMethod] = useState("pmi");
    const [coocDirectional, setCoocDirectional] = useState(false);
    const [coocMinFreq, setCoocMinFreq] = useState(1);
    const [coocMinCount, setCoocMinCount] = useState(1);
    const [metadataFilters, setMetadataFilters] = useState<Record<string, string>>({});
    const [engineRuntime, setEngineRuntime] = useState<"python" | "r">("python");

    useEffect(() => {
        if (engineRuntime !== "r") return;
        setKwicQueryMode("word");
        setDfmConfig((current) => ({
            ...current,
            weighting: "count",
            forceSparseOnly: false,
            minTermFrequency: "",
            maxTermFrequency: "",
            minDocumentFrequency: "",
            maxDocumentFrequency: "",
            topN: "",
        }));
    }, [engineRuntime]);

    const enginesQuery = useQuery({
        queryKey: ["text-research", "analysis-engines"],
        queryFn: listAnalysisEngines,
        staleTime: QUERY_STALE_TIMES.researchReference,
    });

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchReference,
    });

    const dictionariesQuery = useQuery({
        queryKey: queryKeys.textResearch.dictionaries(ctx.projectId),
        queryFn: () => listDictionaries(ctx.projectId),
        enabled: Boolean(ctx.projectId),
        staleTime: QUERY_STALE_TIMES.researchReference,
    });

    const facetsQuery = useQuery({
        queryKey: queryKeys.textResearch.metadataFacets(ctx.selectedCorpusId),
        queryFn: () => getCorpusMetadataFacets(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
        staleTime: QUERY_STALE_TIMES.researchMetadataFacets,
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        staleTime: (query) => researchRunStaleTime(query.state.data?.status),
        refetchInterval: (query) => activeRunRefetchInterval(query, sseConnected),
    });

    const rSelectionState = rEngineSelectionState(enginesQuery.data?.engines, tab);
    const selectedEngineSupportsTab =
        engineRuntime === "python" || rSelectionState === "available";
    const basePayload = {
        unit_type: ctx.unitType,
        ...metadataFilters,
        ...(profileId ? { preprocessing_profile_id: profileId } : {}),
        ...(engineRuntime === "r"
            ? { engine: { runtime: "r" as const, implementation: "quanteda", preprocessing_mode: "standardized" as const } }
            : {}),
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
                query_mode: engineRuntime === "r" ? "word" : kwicQueryMode,
            }),
        onSuccess: (run) => onRunSuccess(run, "KWIC search started."),
        onError: (error) => onRunError(error, "Failed to run KWIC."),
    });

    const dfmMutation = useMutation({
        mutationFn: () =>
            runDfm(ctx.selectedCorpusId, {
                ...basePayload,
                weighting: engineRuntime === "r" ? "count" : dfmConfig.weighting,
                ...(engineRuntime === "r"
                    ? {}
                    : {
                          ...(dfmConfig.weighting === "bm25"
                              ? { k1: dfmConfig.k1, b: dfmConfig.b }
                              : {}),
                          ...((dfmConfig.weighting === "tfidf" ||
                              dfmConfig.weighting === "sublinear_tf")
                              ? { smooth_idf: dfmConfig.smoothIdf }
                              : {}),
                          force_sparse_only: dfmConfig.forceSparseOnly,
                          ...(dfmTrimPayload(dfmConfig) ? { trim: dfmTrimPayload(dfmConfig) } : {}),
                      }),
            }),
        onSuccess: (run) => onRunSuccess(run, "DFM build started."),
        onError: (error) => onRunError(error, "Failed to build DFM."),
    });

    const keynessMutation = useMutation({
        mutationFn: () =>
            runKeyness(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                ...(profileId ? { preprocessing_profile_id: profileId } : {}),
                ...(engineRuntime === "r"
                    ? {
                          engine: {
                              runtime: "r" as const,
                              implementation: "quanteda",
                              preprocessing_mode: "standardized" as const,
                          },
                      }
                    : {}),
                filters_a: { [keynessField]: keynessA.trim() },
                filters_b: { [keynessField]: keynessB.trim() },
                group_field: keynessField,
                method: keynessMethod,
                correction: keynessCorrection,
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
                association_method: coocMethod,
                directional: coocDirectional,
                min_frequency: coocMinFreq,
                min_count: coocMinCount,
            }),
        onSuccess: (run) => onRunSuccess(run, "Co-occurrence analysis started."),
        onError: (error) => onRunError(error, "Failed to run co-occurrence."),
    });

    const compareCompatible =
        (tab === "frequencies" || tab === "dfm" || tab === "kwic") &&
        rSelectionState === "available";

    const compareMutation = useMutation({
        mutationFn: () => {
            const analysis_type = tab as "frequencies" | "dfm" | "kwic";
            const analysis_parameters: Record<string, unknown> =
                analysis_type === "frequencies"
                    ? { top_n: topN }
                    : analysis_type === "dfm"
                      ? { weighting: "count" }
                      : {
                            keyword: kwicKeyword.trim(),
                            window_size: kwicWindow,
                            case_sensitive: kwicCaseSensitive,
                            query_mode: "word",
                        };
            const { engine: _engine, ...compareBase } = basePayload;
            return runEngineComparison(ctx.selectedCorpusId, {
                ...compareBase,
                analysis_type,
                analysis_parameters,
            });
        },
        onSuccess: (run) => onRunSuccess(run, "Python ↔ R comparison started."),
        onError: (error) => onRunError(error, "Failed to compare engines."),
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
        similarity: overviewMutation,
        duplicates: overviewMutation,
        clustering: overviewMutation,
        dimensionality: overviewMutation,
        readability: overviewMutation,
    } as const;

    // statistical / measurement tabs own their mutations in child views.
    const activeMutation =
        tab in mutationByTab
            ? mutationByTab[tab as keyof typeof mutationByTab]
            : null;
    const selectedDictionary = dictionariesQuery.data?.find((d) => d.id === dictionaryId);
    const resolvedDictionaryTerms = parseCommaTerms(dictionaryTerms);
    const hasDictionaryInput =
        resolvedDictionaryTerms.length > 0 || Boolean(selectedDictionary?.terms?.length);

    const canRun = (() => {
        if (!ctx.selectedCorpusId || !activeMutation || activeMutation.isPending || !selectedEngineSupportsTab) return false;
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

    if (tab === "statistical" || tab === "measurement") {
        return (
            <Stack spacing={2}>
                <PageTabs value={tab} onChange={setTab} tabs={TABS} ariaLabel="Analysis methods" />
                <Alert severity="info">
                    Statistical modeling and measurement comparison live under Analysis. APIs:{" "}
                    <code>POST …/analysis/statistical-model</code> and{" "}
                    <code>POST …/analysis/measurement-comparison</code>.
                </Alert>
                <Suspense fallback={<AnalysisPanelFallback />}>
                    {tab === "statistical" ? <StatisticalModelView /> : <MeasurementComparisonView />}
                </Suspense>
            </Stack>
        );
    }

    const advancedPanel =
        tab === "similarity" ? <SimilarityExplorer basePayload={basePayload} />
            : tab === "duplicates" ? <DuplicateDetectionView basePayload={basePayload} />
            : tab === "clustering" ? <ClusterExplorer basePayload={basePayload} />
            : tab === "dimensionality" ? <DimensionalityReductionView basePayload={basePayload} />
            : tab === "readability" ? <ReadabilityView basePayload={basePayload} />
            : null;

    if (advancedPanel) {
        return (
            <Stack spacing={2}>
                <PageTabs value={tab} onChange={setTab} tabs={TABS} ariaLabel="Analysis methods" />
                <SectionCard title="Analysis selection" description="Apply the same corpus selection and preprocessing profile to this analysis.">
                    <Stack spacing={1.5}>
                        <MetadataFilterBar corpusId={ctx.selectedCorpusId} value={metadataFilters} onChange={setMetadataFilters} />
                        <TextField select size="small" label="Preprocessing profile" value={profileId} onChange={(event) => setProfileId(event.target.value)} sx={{ maxWidth: 300 }}>
                            <MenuItem value="">Default / none</MenuItem>
                            {(profilesQuery.data ?? []).map((profile) => <MenuItem key={profile.id} value={profile.id}>{profile.name}</MenuItem>)}
                        </TextField>
                    </Stack>
                </SectionCard>
                <Suspense fallback={<AnalysisPanelFallback />}>{advancedPanel}</Suspense>
            </Stack>
        );
    }

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={TABS}
                ariaLabel="Analysis methods"
            />

            <SectionCard
                title="Quantitative analysis"
                description="Expose corpus statistics, frequencies, n-grams, KWIC, DFM, keyness, dictionaries, and co-occurrence with charts and inspectable raw results."
            >
                <Stack spacing={2}>
                    <MetadataFilterBar corpusId={ctx.selectedCorpusId} value={metadataFilters} onChange={setMetadataFilters} />
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
                        <TextField
                            select
                            size="small"
                            label="Analysis engine"
                            value={engineRuntime}
                            onChange={(event) => setEngineRuntime(event.target.value as "python" | "r")}
                            sx={{ minWidth: 180 }}
                        >
                            <MenuItem value="python">Python</MenuItem>
                            <MenuItem value="r" disabled={rSelectionState !== "available"}>
                                {rEngineOptionLabel(rSelectionState)}
                            </MenuItem>
                        </TextField>
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

                    {engineRuntime === "r" && !selectedEngineSupportsTab ? (
                        <Alert severity="info">
                            R / quanteda supports frequencies, DFM, KWIC, dictionaries, keyness, and co-occurrence when available.
                        </Alert>
                    ) : null}

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
                                label="Query"
                                value={kwicKeyword}
                                onChange={(event) => setKwicKeyword(event.target.value)}
                                sx={{ minWidth: 220 }}
                            />
                            <TextField
                                select
                                size="small"
                                label="Query mode"
                                value={engineRuntime === "r" ? "word" : kwicQueryMode}
                                disabled={engineRuntime === "r"}
                                onChange={(event) => setKwicQueryMode(event.target.value)}
                                sx={{ width: 160 }}
                                helperText={
                                    engineRuntime === "r"
                                        ? "R KWIC requires literal word mode"
                                        : undefined
                                }
                            >
                                <MenuItem value="auto" disabled={engineRuntime === "r"}>
                                    Auto
                                </MenuItem>
                                <MenuItem value="word">Word</MenuItem>
                                <MenuItem value="phrase" disabled={engineRuntime === "r"}>
                                    Phrase
                                </MenuItem>
                                <MenuItem value="exact_phrase" disabled={engineRuntime === "r"}>
                                    Exact phrase
                                </MenuItem>
                                <MenuItem value="regex" disabled={engineRuntime === "r"}>
                                    Regex
                                </MenuItem>
                                <MenuItem value="wildcard" disabled={engineRuntime === "r"}>
                                    Wildcard
                                </MenuItem>
                                <MenuItem value="lemma" disabled={engineRuntime === "r"}>
                                    Lemma
                                </MenuItem>
                            </TextField>
                            <TextField
                                size="small"
                                type="number"
                                label="Window size"
                                value={kwicWindow}
                                onChange={(event) =>
                                    setKwicWindow(Math.max(0, Number(event.target.value) || 0))
                                }
                                inputProps={{ min: 0, max: 50 }}
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
                        <AdvancedDfmPanel
                            config={dfmConfig}
                            onChange={setDfmConfig}
                            rEngine={engineRuntime === "r"}
                        />
                    ) : null}

                    {tab === "keyness" ? (
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                select
                                size="small"
                                label="Compare by metadata"
                                value={keynessField}
                                onChange={(event) => {
                                    setKeynessField(event.target.value);
                                    setKeynessA("");
                                    setKeynessB("");
                                }}
                                sx={{ minWidth: 200 }}
                            >
                                {Object.keys(facetsQuery.data ?? {}).map((field) => (
                                    <MenuItem key={field} value={field}>
                                        {field.replace(/_/g, " ")}
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                select
                                size="small"
                                label="Group A"
                                value={keynessA}
                                onChange={(event) => setKeynessA(event.target.value)}
                                sx={{ minWidth: 180 }}
                            >
                                {(facetsQuery.data?.[keynessField] ?? []).map((item) => (
                                    <MenuItem key={`a-${item.value}`} value={item.value}>
                                        {item.value} ({item.count})
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                select
                                size="small"
                                label="Group B"
                                value={keynessB}
                                onChange={(event) => setKeynessB(event.target.value)}
                                sx={{ minWidth: 180 }}
                            >
                                {(facetsQuery.data?.[keynessField] ?? []).map((item) => (
                                    <MenuItem key={`b-${item.value}`} value={item.value}>
                                        {item.value} ({item.count})
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                select
                                size="small"
                                label="Method"
                                value={keynessMethod}
                                onChange={(event) => setKeynessMethod(event.target.value)}
                                sx={{ minWidth: 160 }}
                            >
                                <MenuItem value="log_likelihood">Log-likelihood (G²)</MenuItem>
                                <MenuItem value="chi_square">Chi-square</MenuItem>
                                <MenuItem value="fisher">Fisher exact</MenuItem>
                            </TextField>
                            <TextField
                                select
                                size="small"
                                label="Correction"
                                value={keynessCorrection}
                                onChange={(event) => setKeynessCorrection(event.target.value)}
                                sx={{ minWidth: 140 }}
                            >
                                <MenuItem value="bh">BH FDR</MenuItem>
                                <MenuItem value="none">None</MenuItem>
                            </TextField>
                        </Stack>
                    ) : null}

                    {tab === "dictionaries" ? (
                        <Stack spacing={2}>
                            <Stack direction="row" spacing={1}>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() =>
                                        navigate(`/research/${ctx.projectId}/dictionaries`)
                                    }
                                >
                                    Manage dictionaries
                                </Button>
                            </Stack>
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
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                size="small"
                                type="number"
                                label="Window size"
                                value={coocWindow}
                                onChange={(event) =>
                                    setCoocWindow(Math.max(1, Number(event.target.value) || 1))
                                }
                                inputProps={{ min: 1, max: 50 }}
                                sx={{ width: 140 }}
                            />
                            <TextField
                                select
                                size="small"
                                label="Association"
                                value={coocMethod}
                                onChange={(event) => setCoocMethod(event.target.value)}
                                sx={{ width: 150 }}
                            >
                                <MenuItem value="count">count</MenuItem>
                                <MenuItem value="pmi">PMI</MenuItem>
                                <MenuItem value="npmi">NPMI</MenuItem>
                                <MenuItem value="dice">Dice</MenuItem>
                                <MenuItem value="log_dice">logDice</MenuItem>
                                <MenuItem value="t_score">t-score</MenuItem>
                            </TextField>
                            <TextField
                                size="small"
                                type="number"
                                label="Min frequency"
                                value={coocMinFreq}
                                onChange={(event) =>
                                    setCoocMinFreq(Math.max(0, Number(event.target.value) || 0))
                                }
                                inputProps={{ min: 0 }}
                                sx={{ width: 140 }}
                            />
                            <TextField
                                size="small"
                                type="number"
                                label="Min co-occurrence"
                                value={coocMinCount}
                                onChange={(event) =>
                                    setCoocMinCount(Math.max(0, Number(event.target.value) || 0))
                                }
                                inputProps={{ min: 0 }}
                                sx={{ width: 160 }}
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={coocDirectional}
                                        onChange={(event) => setCoocDirectional(event.target.checked)}
                                    />
                                }
                                label="Directional"
                            />
                        </Stack>
                    ) : null}

                    <Stack direction="row" spacing={1.5} sx={{ alignSelf: "flex-start" }}>
                        <Button
                            variant="contained"
                            startIcon={<RunIcon />}
                            onClick={() => activeMutation?.mutate()}
                            disabled={!canRun}
                        >
                            Run {runLabel}
                        </Button>
                        {compareCompatible ? (
                            <Button
                                variant="outlined"
                                onClick={() => compareMutation.mutate()}
                                disabled={
                                    compareMutation.isPending ||
                                    (tab === "kwic" && !kwicKeyword.trim())
                                }
                            >
                                Compare Python ↔ R
                            </Button>
                        ) : null}
                    </Stack>
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
                            <ResearchResultPanel run={runQuery.data} title={runLabel}>
                                <AnalysisResults tab={tab} run={runQuery.data} />
                            </ResearchResultPanel>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
