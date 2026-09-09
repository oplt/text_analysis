import { useMemo, useState } from "react";
import {
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
import { MetricCards, ScientificLineChart } from "./ResearchCharts";
import { ResearchResultsTable } from "./ResearchResults";
import type { TrainedModel } from "../types";

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

function reliabilityCurvePoints(
    metrics: Record<string, unknown> | null,
    results: Record<string, unknown> | null
): Array<{ x: number; y: number }> {
    const calibration =
        asRecord(metrics?.calibration) ??
        asRecord(results?.calibration) ??
        asRecord(asRecord(results?.calibration_summary)?.after) ??
        asRecord(asRecord(results?.calibration_summary)?.before) ??
        asRecord(metrics?.calibration_summary);
    const curve =
        (calibration?.reliability_curve as unknown) ??
        (asRecord(calibration)?.reliability_curve as unknown);
    if (!Array.isArray(curve)) return [];
    return curve
        .map((row) => {
            const item = asRecord(row);
            const x = num(item?.mean_predicted) ?? num(item?.confidence) ?? num(item?.x);
            const y = num(item?.fraction_positive) ?? num(item?.accuracy) ?? num(item?.y);
            return x == null || y == null ? null : { x, y };
        })
        .filter((point): point is { x: number; y: number } => point != null);
}

function rocPrSeries(metrics: Record<string, unknown> | null): {
    roc: Array<{ x: number; y: number }>;
    pr: Array<{ x: number; y: number }>;
    rocAuc: number | null;
    prAuc: number | null;
} {
    const rocCurve = Array.isArray(metrics?.roc_curve) ? metrics.roc_curve : null;
    const prCurve = Array.isArray(metrics?.pr_curve) ? metrics.pr_curve : null;
    const roc: Array<{ x: number; y: number }> = [];
    const pr: Array<{ x: number; y: number }> = [];
    if (Array.isArray(rocCurve)) {
        for (const row of rocCurve) {
            const item = asRecord(row);
            const x = num(item?.fpr) ?? num(item?.x);
            const y = num(item?.tpr) ?? num(item?.y);
            if (x != null && y != null) roc.push({ x, y });
        }
    }
    if (Array.isArray(prCurve)) {
        for (const row of prCurve) {
            const item = asRecord(row);
            const x = num(item?.recall) ?? num(item?.x);
            const y = num(item?.precision) ?? num(item?.y);
            if (x != null && y != null) pr.push({ x, y });
        }
    }
    // If only scalar AUCs exist, draw a simple diagonal reference for ROC.
    if (!roc.length && num(metrics?.roc_auc) != null) {
        roc.push({ x: 0, y: 0 }, { x: 1, y: 1 });
    }
    return {
        roc,
        pr,
        rocAuc: num(metrics?.roc_auc),
        prAuc: num(metrics?.pr_auc) ?? num(metrics?.average_precision),
    };
}

export function ClassificationCalibrationPanel({
    metrics,
    results,
}: {
    metrics: Record<string, unknown> | null;
    results: Record<string, unknown> | null;
}) {
    const points = reliabilityCurvePoints(metrics, results);
    const calibration =
        asRecord(metrics?.calibration) ??
        asRecord(results?.calibration) ??
        asRecord(results?.calibration_summary);
    const ece =
        num(calibration?.ece) ??
        num(asRecord(calibration?.after)?.ece) ??
        num(asRecord(calibration?.before)?.ece);

    if (!points.length && ece == null) {
        return (
            <Typography variant="body2" color="text.secondary">
                No calibration curve in this run (probabilities may be unavailable).
            </Typography>
        );
    }

    return (
        <Stack spacing={1}>
            {ece != null ? <MetricCards items={[{ label: "ECE", value: formatMetric(ece) }]} /> : null}
            {points.length ? (
                <ScientificLineChart
                    series={[
                        { label: "Reliability", points },
                        {
                            label: "Perfect",
                            points: [
                                { x: 0, y: 0 },
                                { x: 1, y: 1 },
                            ],
                        },
                    ]}
                    height={280}
                />
            ) : null}
        </Stack>
    );
}

export function ClassificationCurvePanel({
    metrics,
}: {
    metrics: Record<string, unknown> | null;
}) {
    const { roc, pr, rocAuc, prAuc } = rocPrSeries(metrics);
    if (!roc.length && !pr.length && rocAuc == null && prAuc == null) {
        return (
            <Typography variant="body2" color="text.secondary">
                ROC/PR curves are available for binary tasks with probability scores.
            </Typography>
        );
    }
    return (
        <Stack spacing={2}>
            <MetricCards
                items={[
                    ...(rocAuc != null ? [{ label: "ROC-AUC", value: formatMetric(rocAuc) }] : []),
                    ...(prAuc != null ? [{ label: "PR-AUC", value: formatMetric(prAuc) }] : []),
                ]}
            />
            {roc.length > 1 ? (
                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        ROC
                    </Typography>
                    <ScientificLineChart series={[{ label: "ROC", points: roc }]} height={260} />
                </Box>
            ) : null}
            {pr.length ? (
                <Box>
                    <Typography variant="subtitle2" gutterBottom>
                        Precision–Recall
                    </Typography>
                    <ScientificLineChart series={[{ label: "PR", points: pr }]} height={260} />
                </Box>
            ) : null}
        </Stack>
    );
}

export function ClassificationErrorBrowser({
    results,
}: {
    results: Record<string, unknown> | null;
}) {
    const errorAnalysis = asRecord(results?.error_analysis);
    const [sliceField, setSliceField] = useState("");
    const uncertain = Array.isArray(errorAnalysis?.most_uncertain_cases)
        ? errorAnalysis.most_uncertain_cases
        : [];
    const highConfErrors = Array.isArray(errorAnalysis?.high_confidence_errors)
        ? errorAnalysis.high_confidence_errors
        : [];
    const falsePositives = Array.isArray(errorAnalysis?.false_positives)
        ? errorAnalysis.false_positives
        : [];
    const falseNegatives = Array.isArray(errorAnalysis?.false_negatives)
        ? errorAnalysis.false_negatives
        : [];
    const bySlice = asRecord(errorAnalysis?.performance_by_metadata_slice);
    const sliceFields = Object.keys(bySlice ?? {});
    const activeSlice = sliceField || sliceFields[0] || "";
    const sliceRows = asRecord(bySlice?.[activeSlice]);

    if (!errorAnalysis) {
        return (
            <Typography variant="body2" color="text.secondary">
                No error-analysis payload on this run.
            </Typography>
        );
    }

    const caseRows: Array<Record<string, unknown> & { kind: string }> = [
        ...uncertain.map((row) => ({ ...(asRecord(row) ?? {}), kind: "uncertain" })),
        ...highConfErrors.map((row) => ({ ...(asRecord(row) ?? {}), kind: "high-confidence error" })),
    ];

    return (
        <Stack spacing={2}>
            <MetricCards
                items={[
                    { label: "False positives", value: falsePositives.length },
                    { label: "False negatives", value: falseNegatives.length },
                    { label: "Uncertain cases", value: uncertain.length },
                    { label: "High-confidence errors", value: highConfErrors.length },
                ]}
            />
            {sliceFields.length ? (
                <Stack spacing={1}>
                    <TextField
                        select
                        size="small"
                        label="Metadata slice"
                        value={activeSlice}
                        onChange={(e) => setSliceField(e.target.value)}
                        sx={{ maxWidth: 240 }}
                    >
                        {sliceFields.map((field) => (
                            <MenuItem key={field} value={field}>
                                {field}
                            </MenuItem>
                        ))}
                    </TextField>
                    {sliceRows ? (
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Value</TableCell>
                                        <TableCell align="right">Support</TableCell>
                                        <TableCell align="right">Accuracy / F1</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {Object.entries(sliceRows).map(([key, raw]) => {
                                        const row = asRecord(raw);
                                        return (
                                            <TableRow key={key}>
                                                <TableCell>{key}</TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(num(row?.support) ?? num(row?.n), 0)}
                                                </TableCell>
                                                <TableCell align="right">
                                                    {formatMetric(
                                                        num(row?.accuracy) ??
                                                            num(row?.f1) ??
                                                            num(row?.f1_macro)
                                                    )}
                                                </TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </Box>
                    ) : null}
                </Stack>
            ) : null}
            <ResearchResultsTable
                rows={caseRows.map((row, index) => ({
                    id: String(row.unit_id ?? row.text_unit_id ?? index),
                    kind: String(row.kind ?? ""),
                    unit: String(row.unit_id ?? row.text_unit_id ?? ""),
                    trueLabel: String(row.y_true ?? row.true_label ?? ""),
                    predLabel: String(row.y_pred ?? row.predicted_label ?? ""),
                    confidence: num(row.confidence) ?? num(row.uncertainty),
                }))}
                columns={[
                    { id: "kind", label: "Kind", value: (row) => row.kind },
                    { id: "unit", label: "Unit", value: (row) => row.unit },
                    { id: "true", label: "True", value: (row) => row.trueLabel },
                    { id: "pred", label: "Pred", value: (row) => row.predLabel },
                    {
                        id: "confidence",
                        label: "Confidence / uncertainty",
                        value: (row) => row.confidence,
                        align: "right",
                    },
                ]}
            />
        </Stack>
    );
}

export function ClassificationModelComparison({
    models,
    selectedIds,
    onToggle,
}: {
    models: TrainedModel[];
    selectedIds: string[];
    onToggle: (id: string) => void;
}) {
    const selected = useMemo(
        () => models.filter((model) => selectedIds.includes(model.id)),
        [models, selectedIds]
    );

    if (!models.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                Train at least two models to compare.
            </Typography>
        );
    }

    return (
        <Stack spacing={2}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {models.map((model) => {
                    const active = selectedIds.includes(model.id);
                    return (
                        <Button
                            key={model.id}
                            size="small"
                            variant={active ? "contained" : "outlined"}
                            onClick={() => onToggle(model.id)}
                        >
                            {model.name ?? model.id.slice(0, 8)}
                        </Button>
                    );
                })}
            </Stack>
            <Typography variant="caption" color="text.secondary">
                Select 2+ models to compare holdout metrics.
            </Typography>
            {selected.length >= 2 ? (
                <Box sx={{ overflowX: "auto" }}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Model</TableCell>
                                <TableCell>Family</TableCell>
                                <TableCell>Task</TableCell>
                                <TableCell align="right">Macro F1</TableCell>
                                <TableCell align="right">Micro F1</TableCell>
                                <TableCell align="right">Accuracy</TableCell>
                                <TableCell align="right">ROC-AUC</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {selected.map((model) => {
                                const metrics = asRecord(model.metrics);
                                return (
                                    <TableRow key={model.id}>
                                        <TableCell>{model.name ?? model.id.slice(0, 8)}</TableCell>
                                        <TableCell>{model.model_family}</TableCell>
                                        <TableCell>{model.task_type}</TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(metrics?.f1_macro))}
                                        </TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(metrics?.f1_micro))}
                                        </TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(metrics?.accuracy))}
                                        </TableCell>
                                        <TableCell align="right">
                                            {formatMetric(num(metrics?.roc_auc))}
                                        </TableCell>
                                    </TableRow>
                                );
                            })}
                        </TableBody>
                    </Table>
                </Box>
            ) : null}
        </Stack>
    );
}
