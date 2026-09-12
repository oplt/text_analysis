import { useMemo, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Checkbox,
    FormControlLabel,
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
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    listClassifiers,
    runComparativePrevalence,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MetricCards, RankedBarChart, ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnalysisRun, UnitType } from "../types";

const GROUP_BY_OPTIONS = [
    "organization",
    "organization_type",
    "publication_year",
    "publication_type",
    "country",
    "region",
    "cultural_sphere",
    "language",
] as const;

const PROVENANCE_MODES = [
    { value: "human_only", label: "Human only" },
    { value: "model_only", label: "Model only" },
    { value: "human_preferred", label: "Human preferred (model fallback)" },
] as const;

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function num(value: unknown): number | null {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
}

type PrevalenceBucket = {
    group: string;
    total: number;
    yes: number;
    prevalence: number;
    document_count: number;
};

function flattenPrevalence(
    prevalence: Record<string, unknown> | null
): Array<PrevalenceBucket & { label: string }> {
    if (!prevalence) return [];
    const rows: Array<PrevalenceBucket & { label: string }> = [];
    for (const [label, groups] of Object.entries(prevalence)) {
        const groupMap = asRecord(groups);
        if (!groupMap) continue;
        for (const [group, raw] of Object.entries(groupMap)) {
            const bucket = asRecord(raw);
            if (!bucket) continue;
            rows.push({
                label,
                group,
                total: num(bucket.total) ?? 0,
                yes: num(bucket.yes) ?? 0,
                prevalence: num(bucket.prevalence) ?? 0,
                document_count: num(bucket.document_count) ?? 0,
            });
        }
    }
    return rows;
}

export default function ComparativeAnalysisView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();

    const [groupBy, setGroupBy] = useState<string>("country");
    const [provenanceMode, setProvenanceMode] = useState<string>("human_only");
    const [modelId, setModelId] = useState("");
    const [selectedLabelIds, setSelectedLabelIds] = useState<string[]>([]);
    const [organization, setOrganization] = useState("");
    const [region, setRegion] = useState("");
    const [language, setLanguage] = useState("");
    const [run, setRun] = useState<AnalysisRun | null>(null);

    const modelsQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listClassifiers(ctx.projectId, ctx.selectedCorpusId, undefined, signal),
        enabled: Boolean(ctx.projectId) && provenanceMode !== "human_only",
    });

    const labelOptions = ctx.labels;
    const effectiveLabelIds =
        selectedLabelIds.length > 0 ? selectedLabelIds : labelOptions.map((l) => l.id);

    const mutation = useMutation({
        mutationFn: () => {
            if (!ctx.selectedCorpusId) throw new Error("Select a corpus first.");
            if (!ctx.selectedCodebookId) throw new Error("Select a codebook first.");
            if (effectiveLabelIds.length === 0) throw new Error("Select at least one label.");
            return runComparativePrevalence(ctx.selectedCorpusId, {
                unit_type: ctx.unitType as UnitType,
                codebook_id: ctx.selectedCodebookId,
                label_ids: effectiveLabelIds,
                group_by: groupBy,
                provenance_mode: provenanceMode,
                model_id: modelId || undefined,
                organization: organization.trim() || undefined,
                region: region.trim() || undefined,
                language: language.trim() || undefined,
            });
        },
        onSuccess: (next) => {
            setRun(next);
            showToast({ message: "Comparative prevalence completed.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Comparative prevalence failed."),
                severity: "error",
            }),
    });

    const results = asRecord(run?.results);
    const metrics = asRecord(run?.metrics);
    const prevalence = asRecord(results?.prevalence);
    const rows = useMemo(() => flattenPrevalence(prevalence), [prevalence]);
    const chartItems = rows.slice(0, 24).map((row) => ({
        label: `${row.label} · ${row.group}`,
        value: row.prevalence,
    }));

    function toggleLabel(id: string) {
        setSelectedLabelIds((ids) =>
            ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
        );
    }

    function exportCsv() {
        const header = "label,group,yes,total,prevalence,document_count\n";
        const body = rows
            .map(
                (row) =>
                    `"${row.label.replaceAll('"', '""')}","${row.group.replaceAll('"', '""')}",${row.yes},${row.total},${row.prevalence},${row.document_count}`
            )
            .join("\n");
        const blob = new Blob([header + body], { type: "text/csv;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `comparative-prevalence-${run?.id ?? "export"}.csv`;
        anchor.click();
        URL.revokeObjectURL(url);
    }

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Comparative prevalence"
                description="Descriptive prevalence of coded labels across metadata groups. Does not support causal inference."
            >
                <Stack spacing={2}>
                    <Alert severity="warning">
                        These comparisons are descriptive. Differences across groups are not
                        causal estimates and should not be interpreted as treatment effects.
                    </Alert>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                        <TextField
                            select
                            size="small"
                            label="Group by"
                            value={groupBy}
                            onChange={(e) => setGroupBy(e.target.value)}
                            sx={{ minWidth: 200 }}
                        >
                            {GROUP_BY_OPTIONS.map((field) => (
                                <MenuItem key={field} value={field}>
                                    {field}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Provenance mode"
                            value={provenanceMode}
                            onChange={(e) => setProvenanceMode(e.target.value)}
                            sx={{ minWidth: 240 }}
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
                                label="Classifier / model"
                                value={modelId}
                                onChange={(e) => setModelId(e.target.value)}
                                sx={{ minWidth: 220 }}
                            >
                                <MenuItem value="">None</MenuItem>
                                {(modelsQuery.data ?? []).map((model) => (
                                    <MenuItem key={model.id} value={model.id}>
                                        {model.name ?? model.id.slice(0, 8)}
                                    </MenuItem>
                                ))}
                            </TextField>
                        ) : null}
                    </Stack>

                    <Box>
                        <Typography variant="subtitle2" gutterBottom>
                            Labels
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            {labelOptions.map((label) => (
                                <FormControlLabel
                                    key={label.id}
                                    control={
                                        <Checkbox
                                            size="small"
                                            checked={
                                                selectedLabelIds.length === 0 ||
                                                selectedLabelIds.includes(label.id)
                                            }
                                            onChange={() => toggleLabel(label.id)}
                                        />
                                    }
                                    label={label.name}
                                />
                            ))}
                        </Stack>
                        <Typography variant="caption" color="text.secondary">
                            Leave all unchecked to include every codebook label.
                        </Typography>
                    </Box>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                        <TextField
                            size="small"
                            label="Filter: organization"
                            value={organization}
                            onChange={(e) => setOrganization(e.target.value)}
                        />
                        <TextField
                            size="small"
                            label="Filter: region"
                            value={region}
                            onChange={(e) => setRegion(e.target.value)}
                        />
                        <TextField
                            size="small"
                            label="Filter: language"
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                        />
                    </Stack>

                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="contained"
                            startIcon={<RunIcon />}
                            disabled={
                                mutation.isPending ||
                                !ctx.selectedCorpusId ||
                                !ctx.selectedCodebookId
                            }
                            onClick={() => mutation.mutate()}
                        >
                            Run comparative prevalence
                        </Button>
                        {rows.length ? (
                            <Button variant="outlined" onClick={exportCsv}>
                                Export table CSV
                            </Button>
                        ) : null}
                    </Stack>
                </Stack>
            </SectionCard>

            {run ? (
                <SectionCard
                    title="Results"
                    description="Prevalence by group with denominators. Confidence depends on sample size per cell."
                    action={<RunStatusChip status={run.status} />}
                >
                    <QueryBoundary isLoading={mutation.isPending}>
                        <Stack spacing={2}>
                            <MetricCards
                                items={[
                                    {
                                        label: "Units",
                                        value: String(num(metrics?.unit_count) ?? "—"),
                                    },
                                    {
                                        label: "Human provenance",
                                        value: String(
                                            num(asRecord(metrics?.provenance_counts)?.human) ?? "—"
                                        ),
                                    },
                                    {
                                        label: "Model provenance",
                                        value: String(
                                            num(asRecord(metrics?.provenance_counts)?.model) ?? "—"
                                        ),
                                    },
                                    {
                                        label: "Group field",
                                        value: groupBy,
                                    },
                                ]}
                            />

                            {chartItems.length ? (
                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Prevalence (top cells)
                                    </Typography>
                                    <RankedBarChart
                                        items={chartItems}
                                        valueFormatter={(v) =>
                                            v == null ? "—" : `${(v * 100).toFixed(1)}%`
                                        }
                                    />
                                </Box>
                            ) : null}

                            <Box sx={{ overflowX: "auto" }}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Label</TableCell>
                                            <TableCell>Group</TableCell>
                                            <TableCell align="right">Yes</TableCell>
                                            <TableCell align="right">Total (denom.)</TableCell>
                                            <TableCell align="right">Prevalence</TableCell>
                                            <TableCell align="right">Documents</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {rows.map((row) => (
                                            <TableRow key={`${row.label}:${row.group}`}>
                                                <TableCell>{row.label}</TableCell>
                                                <TableCell>{row.group}</TableCell>
                                                <TableCell align="right">{row.yes}</TableCell>
                                                <TableCell align="right">{row.total}</TableCell>
                                                <TableCell align="right">
                                                    {(row.prevalence * 100).toFixed(1)}%
                                                </TableCell>
                                                <TableCell align="right">
                                                    {row.document_count}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>

                            <Alert severity="info">
                                Small denominators yield unstable prevalence estimates. Inspect
                                cell totals before interpreting group differences.
                            </Alert>

                            <ResultsInspector title="comparative results" data={run} />
                        </Stack>
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
