import { useState } from "react";
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
import { PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import { compareMeasurements } from "../../../api/textResearch";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MatrixHeatmap, MetricCards, RankedBarChart, ResultsInspector } from "../components/ResearchCharts";
import { RunStatusChip } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import type { AnalysisRun } from "../types";

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

/** Accept JSON array or newline/comma separated scalars. */
function parseSeries(text: string, label: string): unknown[] {
    const trimmed = text.trim();
    if (!trimmed) throw new Error(`${label} is empty.`);
    if (trimmed.startsWith("[")) {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!Array.isArray(parsed)) throw new Error(`${label} JSON must be an array.`);
        return parsed;
    }
    return trimmed
        .split(/[\n,]+/)
        .map((part) => part.trim())
        .filter((part) => part.length > 0)
        .map((part) => {
            const asNumber = Number(part);
            return Number.isFinite(asNumber) && part !== "" ? asNumber : part;
        });
}

function parseOptionalStringList(text: string): string[] | undefined {
    const trimmed = text.trim();
    if (!trimmed) return undefined;
    if (trimmed.startsWith("[")) {
        const parsed = JSON.parse(trimmed) as unknown;
        if (!Array.isArray(parsed)) throw new Error("Optional list must be a JSON array.");
        return parsed.map(String);
    }
    return trimmed
        .split(/[\n,]+/)
        .map((part) => part.trim())
        .filter(Boolean);
}

const SAMPLE_A = `["a","a","b","b","a","b"]`;
const SAMPLE_B = `["a","b","b","b","a","a"]`;

export default function MeasurementComparisonView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();

    const [sourceA, setSourceA] = useState("human");
    const [sourceB, setSourceB] = useState("classifier");
    const [valuesAText, setValuesAText] = useState(SAMPLE_A);
    const [valuesBText, setValuesBText] = useState(SAMPLE_B);
    const [idsText, setIdsText] = useState("");
    const [subgroupText, setSubgroupText] = useState("");
    const [valueKind, setValueKind] = useState<"categorical" | "continuous">("categorical");
    const [run, setRun] = useState<AnalysisRun | null>(null);

    const compareMutation = useMutation({
        mutationFn: () => {
            const valuesA = parseSeries(valuesAText, "Series A");
            const valuesB = parseSeries(valuesBText, "Series B");
            if (valuesA.length !== valuesB.length) {
                throw new Error("Series A and B must have equal length.");
            }
            return compareMeasurements(ctx.selectedCorpusId, {
                source_a: sourceA.trim() || "source_a",
                source_b: sourceB.trim() || "source_b",
                values_a: valuesA,
                values_b: valuesB,
                ids: parseOptionalStringList(idsText),
                value_kind: valueKind,
                subgroup: parseOptionalStringList(subgroupText),
            });
        },
        onSuccess: (next) => {
            setRun(next);
            showToast({ message: "Measurement comparison completed.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Measurement comparison failed."),
                severity: "error",
            }),
    });

    const results = asRecord(run?.results);
    const confusion = asRecord(results?.confusion);
    const confusionLabels = Array.isArray(confusion?.labels)
        ? confusion.labels.map(String)
        : [];
    const confusionMatrix = Array.isArray(confusion?.matrix)
        ? (confusion.matrix as number[][])
        : null;
    const prevalenceA = asRecord(results?.prevalence_a);
    const prevalenceB = asRecord(results?.prevalence_b);
    const correlation = asRecord(results?.correlation);
    const bySubgroup = asRecord(results?.by_subgroup);
    const notes = Array.isArray(results?.notes) ? results.notes.map(String) : [];

    return (
        <SectionCard
            title="Measurement comparison"
            description="Triangulate two user-aligned measurement series. The platform does not equate annotation, dictionary, classifier, or topic scores automatically."
        >
            <Stack spacing={2}>
                <Alert severity="info">
                    You assert that the series are comparable. Results report agreement or
                    correlation only — they do not merge conceptual frameworks.
                </Alert>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                    <TextField
                        size="small"
                        label="Source A name"
                        value={sourceA}
                        onChange={(e) => setSourceA(e.target.value)}
                        sx={{ minWidth: 160 }}
                    />
                    <TextField
                        size="small"
                        label="Source B name"
                        value={sourceB}
                        onChange={(e) => setSourceB(e.target.value)}
                        sx={{ minWidth: 160 }}
                    />
                    <TextField
                        select
                        size="small"
                        label="Value kind"
                        value={valueKind}
                        onChange={(e) =>
                            setValueKind(e.target.value as "categorical" | "continuous")
                        }
                        sx={{ minWidth: 180 }}
                    >
                        <MenuItem value="categorical">Categorical (agreement)</MenuItem>
                        <MenuItem value="continuous">Continuous (correlation)</MenuItem>
                    </TextField>
                </Stack>

                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                    <TextField
                        size="small"
                        label={`Values A (${sourceA || "A"})`}
                        value={valuesAText}
                        onChange={(e) => setValuesAText(e.target.value)}
                        multiline
                        minRows={5}
                        fullWidth
                        helperText="JSON array or comma/newline list"
                    />
                    <TextField
                        size="small"
                        label={`Values B (${sourceB || "B"})`}
                        value={valuesBText}
                        onChange={(e) => setValuesBText(e.target.value)}
                        multiline
                        minRows={5}
                        fullWidth
                        helperText="Must align 1:1 with series A"
                    />
                </Stack>

                <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                    <TextField
                        size="small"
                        label="Optional unit IDs"
                        value={idsText}
                        onChange={(e) => setIdsText(e.target.value)}
                        multiline
                        minRows={2}
                        fullWidth
                        helperText="Same length as series when set"
                    />
                    <TextField
                        size="small"
                        label="Optional subgroup labels"
                        value={subgroupText}
                        onChange={(e) => setSubgroupText(e.target.value)}
                        multiline
                        minRows={2}
                        fullWidth
                        helperText="Same length as original series for stratified comparison"
                    />
                </Stack>

                <Button
                    variant="contained"
                    startIcon={<RunIcon />}
                    onClick={() => compareMutation.mutate()}
                    disabled={!ctx.selectedCorpusId || compareMutation.isPending}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Compare measurements
                </Button>

                {run ? (
                    <Stack spacing={2}>
                        <Typography variant="body2">
                            Run {run.id} — <RunStatusChip status={run.status} />
                        </Typography>
                        {run.error_message ? (
                            <Alert severity="error">{run.error_message}</Alert>
                        ) : null}

                        <MetricCards
                            items={[
                                {
                                    label: "Paired n",
                                    value: formatMetric(num(results?.n_paired), 0),
                                },
                                {
                                    label: "Dropped missing",
                                    value: formatMetric(num(results?.n_dropped_missing), 0),
                                },
                                {
                                    label: "Agreement",
                                    value: formatMetric(num(results?.agreement_rate)),
                                },
                                {
                                    label: "Correlation r",
                                    value: formatMetric(num(correlation?.r)),
                                },
                                {
                                    label: "Mean A",
                                    value: formatMetric(num(results?.mean_a)),
                                },
                                {
                                    label: "Mean B",
                                    value: formatMetric(num(results?.mean_b)),
                                },
                            ]}
                        />

                        {confusionMatrix && confusionLabels.length ? (
                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    Confusion ({sourceA} rows × {sourceB} columns)
                                </Typography>
                                <MatrixHeatmap
                                    rowLabels={confusionLabels}
                                    colLabels={confusionLabels}
                                    values={confusionMatrix}
                                    formatCell={(value) => (value == null ? "—" : String(value))}
                                />
                            </Box>
                        ) : null}

                        {prevalenceA || prevalenceB ? (
                            <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                                {prevalenceA ? (
                                    <Box sx={{ flex: 1 }}>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Prevalence — {sourceA}
                                        </Typography>
                                        <RankedBarChart
                                            items={Object.entries(prevalenceA).map(([label, value]) => ({
                                                label,
                                                value: Number(value) || 0,
                                            }))}
                                            valueFormatter={(v) =>
                                                v == null ? "" : `${(Number(v) * 100).toFixed(1)}%`
                                            }
                                        />
                                    </Box>
                                ) : null}
                                {prevalenceB ? (
                                    <Box sx={{ flex: 1 }}>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Prevalence — {sourceB}
                                        </Typography>
                                        <RankedBarChart
                                            items={Object.entries(prevalenceB).map(([label, value]) => ({
                                                label,
                                                value: Number(value) || 0,
                                            }))}
                                            valueFormatter={(v) =>
                                                v == null ? "" : `${(Number(v) * 100).toFixed(1)}%`
                                            }
                                        />
                                    </Box>
                                ) : null}
                            </Stack>
                        ) : null}

                        {bySubgroup ? (
                            <Box sx={{ overflowX: "auto" }}>
                                <Typography variant="subtitle2" gutterBottom>
                                    By subgroup
                                </Typography>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Subgroup</TableCell>
                                            <TableCell align="right">n</TableCell>
                                            <TableCell align="right">Agreement / r</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {Object.entries(bySubgroup).map(([group, raw]) => {
                                            const row = asRecord(raw);
                                            const corr = asRecord(row?.correlation);
                                            return (
                                                <TableRow key={group}>
                                                    <TableCell>{group}</TableCell>
                                                    <TableCell align="right">
                                                        {formatMetric(num(row?.n_paired), 0)}
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {formatMetric(
                                                            num(row?.agreement_rate) ?? num(corr?.r)
                                                        )}
                                                    </TableCell>
                                                </TableRow>
                                            );
                                        })}
                                    </TableBody>
                                </Table>
                            </Box>
                        ) : null}

                        {notes.map((note) => (
                            <Alert key={note} severity="warning">
                                {note}
                            </Alert>
                        ))}

                        <ResultsInspector title="measurement comparison" data={run} />
                    </Stack>
                ) : null}
            </Stack>
        </SectionCard>
    );
}
