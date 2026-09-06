import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Divider,
    Drawer,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getRun,
    listClassifiers,
    runComparativePrevalence,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MatrixHeatmap, ResultsInspector, ScientificLineChart } from "../components/ResearchCharts";
import { ResearchResultsTable } from "../components/ResearchResults";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import { activeRunRefetchInterval, isActiveRunStatus } from "../runPolling";

const GROUP_BY_OPTIONS = [
    "organization",
    "organization_type",
    "publication_year",
    "country",
    "region",
    "cultural_sphere",
    "language",
    "publication_type",
] as const;

const PROVENANCE_MODES = [
    { value: "human_only", label: "Human only" },
    { value: "model_only", label: "Model only" },
    { value: "human_preferred", label: "Human preferred" },
] as const;

type CellSelection = { label: string; group: string };

type PrevalenceCell = {
    total?: number;
    yes?: number;
    prevalence?: number;
    document_count?: number;
    provenance_counts?: Record<string, number>;
    mean_uncertainty?: number | null;
    temporal_distribution?: Record<string, number>;
};

type PassageExample = {
    text_unit_id?: string;
    text?: string;
    document_id?: string;
    document_title?: string | null;
    organization?: string | null;
    publication_year?: number | null;
    country?: string | null;
    provenance?: string;
    uncertainty?: number | null;
    document_metadata?: Record<string, string | null>;
};

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function optionalText(value: string): string | undefined {
    const trimmed = value.trim();
    return trimmed ? trimmed : undefined;
}

function optionalInt(value: string): number | undefined {
    if (!value.trim()) return undefined;
    const n = Number(value);
    return Number.isFinite(n) ? n : undefined;
}

function buildHeatmap(prevalence: Record<string, unknown> | null): {
    labels: string[];
    groups: string[];
    values: number[][];
} {
    if (!prevalence) return { labels: [], groups: [], values: [] };
    const labels = Object.keys(prevalence);
    const groupSet = new Set<string>();
    for (const label of labels) {
        const byGroup = asRecord(prevalence[label]);
        if (!byGroup) continue;
        Object.keys(byGroup).forEach((group) => groupSet.add(group));
    }
    const groups = Array.from(groupSet).sort((a, b) => a.localeCompare(b));
    // Rows = groups, columns = discourse labels (group × label matrix).
    const values = groups.map((group) =>
        labels.map((label) => {
            const byGroup = asRecord(prevalence[label]) ?? {};
            const cell = asRecord(byGroup[group]) as PrevalenceCell | null;
            return typeof cell?.prevalence === "number" ? cell.prevalence : 0;
        })
    );
    return { labels, groups, values };
}

export default function ExplorerView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();

    const [groupBy, setGroupBy] = useState<string>("country");
    const [provenanceMode, setProvenanceMode] = useState<string>("human_only");
    const [modelId, setModelId] = useState("");
    const [organization, setOrganization] = useState("");
    const [organizationType, setOrganizationType] = useState("");
    const [yearMin, setYearMin] = useState("");
    const [yearMax, setYearMax] = useState("");
    const [country, setCountry] = useState("");
    const [region, setRegion] = useState("");
    const [culturalSphere, setCulturalSphere] = useState("");
    const [language, setLanguage] = useState("");
    const [publicationType, setPublicationType] = useState("");
    const [runId, setRunId] = useState<string | null>(null);
    const [selection, setSelection] = useState<CellSelection | null>(null);

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listClassifiers(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId && provenanceMode !== "human_only"),
    });

    const runQuery = useQuery({
        queryKey: queryKeys.textResearch.run(runId ?? ""),
        queryFn: () => getRun(runId!),
        enabled: Boolean(runId),
        refetchInterval: activeRunRefetchInterval,
    });

    const prevalenceMutation = useMutation({
        mutationFn: () =>
            runComparativePrevalence(ctx.selectedCorpusId, {
                unit_type: ctx.unitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: ctx.labels.map((label) => label.id),
                group_by: groupBy,
                provenance_mode: provenanceMode,
                model_id:
                    provenanceMode === "human_only" ? undefined : modelId || undefined,
                organization: optionalText(organization),
                organization_type: optionalText(organizationType),
                publication_year_min: optionalInt(yearMin),
                publication_year_max: optionalInt(yearMax),
                country: optionalText(country),
                region: optionalText(region),
                cultural_sphere: optionalText(culturalSphere),
                language: optionalText(language),
                publication_type: optionalText(publicationType),
            }),
        onSuccess: (run) => {
            setRunId(run.id);
            setSelection(null);
            showToast({
                message: "Comparative prevalence analysis started.",
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to run comparative analysis."),
                severity: "error",
            }),
    });

    const results = asRecord(runQuery.data?.results);
    const prevalence = asRecord(results?.prevalence);
    const examples = asRecord(results?.examples);
    const heatmap = useMemo(() => buildHeatmap(prevalence), [prevalence]);

    const selectedCell = useMemo(() => {
        if (!selection || !prevalence) return null;
        const byGroup = asRecord(prevalence[selection.label]);
        return (asRecord(byGroup?.[selection.group]) as PrevalenceCell | null) ?? null;
    }, [prevalence, selection]);

    const selectedExamples = useMemo(() => {
        if (!selection || !examples) return [] as PassageExample[];
        const byGroup = asRecord(examples[selection.label]);
        const rows = byGroup?.[selection.group];
        return Array.isArray(rows) ? (rows as PassageExample[]) : [];
    }, [examples, selection]);

    const disabled =
        !ctx.selectedCorpusId ||
        !ctx.selectedCodebookId ||
        ctx.labels.length === 0 ||
        (provenanceMode !== "human_only" && !modelId);

    const completed = runQuery.data?.status === "completed";

    return (
        <Stack spacing={2}>
            <Alert severity="info">
                This explorer shows patterns and evidence only. Interpretation — including any claim
                about “Western bias” — belongs to the researcher. The tool does not conclude bias.
            </Alert>

            <SectionCard
                title="Comparative prevalence explorer"
                description="Compare discourse-label prevalence across metadata groups with optional provenance filtering."
            >
                <Stack spacing={2} sx={{ mb: 2 }}>
                    <Typography variant="subtitle2">Global filters</Typography>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                        <TextField
                            size="small"
                            label="Organization"
                            value={organization}
                            onChange={(e) => setOrganization(e.target.value)}
                            sx={{ minWidth: 160 }}
                        />
                        <TextField
                            size="small"
                            label="Organization type"
                            value={organizationType}
                            onChange={(e) => setOrganizationType(e.target.value)}
                            sx={{ minWidth: 160 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="Year min"
                            value={yearMin}
                            onChange={(e) => setYearMin(e.target.value)}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            type="number"
                            label="Year max"
                            value={yearMax}
                            onChange={(e) => setYearMax(e.target.value)}
                            sx={{ width: 120 }}
                        />
                        <TextField
                            size="small"
                            label="Country"
                            value={country}
                            onChange={(e) => setCountry(e.target.value)}
                            sx={{ minWidth: 140 }}
                        />
                        <TextField
                            size="small"
                            label="Region"
                            value={region}
                            onChange={(e) => setRegion(e.target.value)}
                            sx={{ minWidth: 140 }}
                        />
                        <TextField
                            size="small"
                            label="Cultural sphere"
                            value={culturalSphere}
                            onChange={(e) => setCulturalSphere(e.target.value)}
                            sx={{ minWidth: 160 }}
                        />
                        <TextField
                            size="small"
                            label="Language"
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                            sx={{ minWidth: 140 }}
                        />
                        <TextField
                            size="small"
                            label="Publication type"
                            value={publicationType}
                            onChange={(e) => setPublicationType(e.target.value)}
                            sx={{ minWidth: 160 }}
                        />
                    </Stack>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap">
                        <TextField
                            select
                            size="small"
                            label="Group by"
                            value={groupBy}
                            onChange={(e) => setGroupBy(e.target.value)}
                            sx={{ minWidth: 200 }}
                        >
                            {GROUP_BY_OPTIONS.map((option) => (
                                <MenuItem key={option} value={option}>
                                    {option.replace(/_/g, " ")}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Provenance mode"
                            value={provenanceMode}
                            onChange={(e) => {
                                setProvenanceMode(e.target.value);
                                if (e.target.value === "human_only") setModelId("");
                            }}
                            sx={{ minWidth: 200 }}
                        >
                            {PROVENANCE_MODES.map((mode) => (
                                <MenuItem key={mode.value} value={mode.value}>
                                    {mode.label}
                                </MenuItem>
                            ))}
                        </TextField>
                        {provenanceMode !== "human_only" ? (
                            <TextField
                                select
                                size="small"
                                label="Classifier model"
                                value={modelId}
                                onChange={(e) => setModelId(e.target.value)}
                                sx={{ minWidth: 240 }}
                            >
                                <MenuItem value="">Select model</MenuItem>
                                {(classifiersQuery.data ?? []).map((model) => (
                                    <MenuItem key={model.id} value={model.id}>
                                        {model.name || model.model_family} · v{model.version}
                                    </MenuItem>
                                ))}
                            </TextField>
                        ) : null}
                        <TextField
                            size="small"
                            label="Labels"
                            value={ctx.labels.map((label) => label.name).join(", ")}
                            InputProps={{ readOnly: true }}
                            sx={{ flex: 1, minWidth: 220 }}
                        />
                    </Stack>
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
                                    Run {runQuery.data.id} —{" "}
                                    <RunStatusChip status={runQuery.data.status} />
                                </Typography>
                                {runQuery.data.error_message ? (
                                    <Alert severity="error">{runQuery.data.error_message}</Alert>
                                ) : null}

                                {completed && heatmap.labels.length ? (
                                    <>
                                        <Typography variant="subtitle2">
                                            Group × discourse label prevalence
                                        </Typography>
                                        <MatrixHeatmap
                                            rowLabels={heatmap.groups}
                                            colLabels={heatmap.labels}
                                            values={heatmap.values}
                                            formatCell={(value) =>
                                                value == null ? "—" : `${(value * 100).toFixed(0)}%`
                                            }
                                            onCellClick={(rowIndex, colIndex) =>
                                                setSelection({
                                                    group: heatmap.groups[rowIndex],
                                                    label: heatmap.labels[colIndex],
                                                })
                                            }
                                        />
                                        {groupBy === "publication_year" ? (
                                            <ScientificLineChart
                                                series={heatmap.labels.map((label, columnIndex) => ({
                                                    label,
                                                    points: heatmap.groups
                                                        .map((year, rowIndex) => ({ x: Number(year), y: heatmap.values[rowIndex]?.[columnIndex] ?? 0 }))
                                                        .filter((point) => Number.isFinite(point.x))
                                                        .sort((a, b) => a.x - b.x),
                                                }))}
                                            />
                                        ) : null}
                                        <ResearchResultsTable
                                            rows={heatmap.groups.flatMap((group) =>
                                                heatmap.labels.map((label) => {
                                                    const cell = asRecord(asRecord(prevalence?.[label])?.[group]) as PrevalenceCell | null;
                                                    return {
                                                        id: `${group}:${label}`,
                                                        group,
                                                        label,
                                                        yes: cell?.yes ?? null,
                                                        total: cell?.total ?? null,
                                                        prevalence: cell?.prevalence ?? null,
                                                        documentCount: cell?.document_count ?? null,
                                                        provenance: cell?.provenance_counts
                                                            ? Object.entries(cell.provenance_counts).map(([source, count]) => `${source}: ${count}`).join(", ")
                                                            : null,
                                                    };
                                                })
                                            )}
                                            columns={[
                                                { id: "group", label: "Metadata group", value: (row) => row.group },
                                                { id: "label", label: "Discourse label", value: (row) => row.label },
                                                { id: "yes", label: "Yes", value: (row) => row.yes, align: "right" },
                                                { id: "total", label: "Total", value: (row) => row.total, align: "right" },
                                                { id: "prevalence", label: "Prevalence", value: (row) => row.prevalence, align: "right" },
                                                { id: "documents", label: "Document count", value: (row) => row.documentCount, align: "right" },
                                                { id: "provenance", label: "Provenance", value: (row) => row.provenance },
                                            ]}
                                        />
                                        <ResultsInspector
                                            title="prevalence results"
                                            data={{
                                                metrics: runQuery.data.metrics,
                                                results: runQuery.data.results,
                                            }}
                                        />
                                    </>
                                ) : isActiveRunStatus(runQuery.data.status) ? (
                                    <Typography color="text.secondary">Awaiting results…</Typography>
                                ) : (
                                    <Typography color="text.secondary">No prevalence matrix yet.</Typography>
                                )}
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            <Drawer
                anchor="right"
                open={Boolean(selection)}
                onClose={() => setSelection(null)}
                PaperProps={{ sx: { width: { xs: "100%", sm: 440 }, p: 2 } }}
            >
                {selection ? (
                    <Stack spacing={2}>
                        <Typography variant="h6">
                            {selection.label} · {selection.group}
                        </Typography>
                        <Alert severity="warning">
                            Evidence only — this panel does not conclude “Western bias” or any other
                            institutional judgment.
                        </Alert>
                        <Stack spacing={0.5}>
                            <Typography variant="body2">
                                Units total: {selectedCell?.total ?? "—"}
                            </Typography>
                            <Typography variant="body2">
                                Documents: {selectedCell?.document_count ?? "—"}
                            </Typography>
                            <Typography variant="body2">
                                Units yes: {selectedCell?.yes ?? "—"}
                            </Typography>
                            <Typography variant="body2">
                                Prevalence:{" "}
                                {typeof selectedCell?.prevalence === "number"
                                    ? `${(selectedCell.prevalence * 100).toFixed(1)}%`
                                    : "—"}
                            </Typography>
                        </Stack>

                        <Divider />
                        <Typography variant="subtitle2">Cell provenance</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Human: {String(selectedCell?.provenance_counts?.human ?? "—")} · Model:{" "}
                            {String(selectedCell?.provenance_counts?.model ?? "—")} · Missing:{" "}
                            {String(selectedCell?.provenance_counts?.missing ?? "—")}
                        </Typography>
                        {typeof selectedCell?.mean_uncertainty === "number" ? (
                            <Typography variant="body2" color="text.secondary">
                                Mean model uncertainty: {selectedCell.mean_uncertainty.toFixed(3)}
                            </Typography>
                        ) : null}

                        {selectedCell?.temporal_distribution ? (
                            <Typography variant="body2" color="text.secondary">
                                Temporal distribution: {Object.entries(selectedCell.temporal_distribution)
                                    .sort(([a], [b]) => a.localeCompare(b))
                                    .map(([year, count]) => `${year}: ${count}`)
                                    .join(" · ") || "No publication years"}
                            </Typography>
                        ) : null}

                        <Divider />
                        <Typography variant="subtitle2">Representative passages / KWIC context</Typography>
                        {selectedExamples.length ? (
                            selectedExamples.map((example, index) => (
                                <Box
                                    key={example.text_unit_id ?? index}
                                    sx={{ p: 1.5, bgcolor: "action.hover", borderRadius: 1 }}
                                >
                                    <Typography variant="caption" color="text.secondary" display="block">
                                        {[
                                            example.document_title || example.document_id,
                                            example.organization,
                                            example.publication_year,
                                            example.country,
                                            example.provenance,
                                            typeof example.uncertainty === "number"
                                                ? `uncertainty ${example.uncertainty.toFixed(3)}`
                                                : null,
                                        ]
                                            .filter((part) => part != null && part !== "")
                                            .join(" · ")}
                                    </Typography>
                                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                                        {example.text || "—"}
                                    </Typography>
                                    {example.document_metadata ? (
                                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                                            {Object.entries(example.document_metadata)
                                                .filter(([, value]) => value)
                                                .map(([key, value]) => `${key.replace(/_/g, " ")}: ${value}`)
                                                .join(" · ")}
                                        </Typography>
                                    ) : null}
                                </Box>
                            ))
                        ) : (
                            <Typography variant="body2" color="text.secondary">
                                No representative passages for this cell.
                            </Typography>
                        )}
                    </Stack>
                ) : null}
            </Drawer>
        </Stack>
    );
}
