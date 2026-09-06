import { useState } from "react";
import {
    Box,
    Button,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import { BarChart } from "@mui/x-charts/BarChart";
import { JsonBlock } from "./ResearchShared";

export type RankedItem = {
    label: string;
    value: number;
};

export function RankedBarChart({
    items,
    height = 320,
    valueFormatter,
}: {
    items: RankedItem[];
    height?: number;
    valueFormatter?: (value: number | null) => string;
}) {
    const sliced = items.slice(0, 25);
    if (!sliced.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No chartable values.
            </Typography>
        );
    }
    return (
        <Box sx={{ width: "100%", minHeight: height }}>
            <BarChart
                layout="horizontal"
                height={Math.max(height, sliced.length * 28)}
                yAxis={[{ data: sliced.map((item) => item.label), width: 140 }]}
                series={[
                    {
                        data: sliced.map((item) => item.value),
                        valueFormatter: valueFormatter ?? ((v) => String(v ?? "")),
                    },
                ]}
                margin={{ left: 16, right: 16, top: 16, bottom: 16 }}
            />
        </Box>
    );
}

export function DivergingBarChart({
    items,
    height = 360,
}: {
    items: Array<{ label: string; value: number }>;
    height?: number;
}) {
    const sliced = items.slice(0, 30);
    if (!sliced.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No coefficient values.
            </Typography>
        );
    }
    return (
        <Box sx={{ width: "100%", minHeight: height }}>
            <BarChart
                layout="horizontal"
                height={Math.max(height, sliced.length * 26)}
                yAxis={[{ data: sliced.map((item) => item.label), width: 160 }]}
                series={[{ data: sliced.map((item) => item.value) }]}
                margin={{ left: 16, right: 16, top: 16, bottom: 16 }}
            />
        </Box>
    );
}

export function SimpleLineLikeBars({
    items,
    height = 280,
}: {
    items: RankedItem[];
    height?: number;
}) {
    if (!items.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No values to plot.
            </Typography>
        );
    }
    return (
        <Box sx={{ width: "100%" }}>
            <BarChart
                height={height}
                xAxis={[{ data: items.map((item) => item.label), scaleType: "band" }]}
                series={[{ data: items.map((item) => item.value) }]}
                margin={{ left: 40, right: 16, top: 16, bottom: 40 }}
            />
        </Box>
    );
}

function heatColor(value: number, max: number): string {
    if (max <= 0) return "transparent";
    const t = Math.max(0, Math.min(1, value / max));
    return `rgba(25, 118, 210, ${0.08 + t * 0.72})`;
}

export function MatrixHeatmap({
    rowLabels,
    colLabels,
    values,
    onCellClick,
    formatCell,
}: {
    rowLabels: string[];
    colLabels: string[];
    values: number[][];
    onCellClick?: (rowIndex: number, colIndex: number) => void;
    formatCell?: (value: number) => string;
}) {
    if (!rowLabels.length || !colLabels.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No matrix data.
            </Typography>
        );
    }
    const max = Math.max(0.0001, ...values.flatMap((row) => row));
    const fmt = formatCell ?? ((v: number) => v.toFixed(2));
    return (
        <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell />
                        {colLabels.map((label) => (
                            <TableCell key={label} align="center">
                                {label}
                            </TableCell>
                        ))}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {rowLabels.map((rowLabel, rowIndex) => (
                        <TableRow key={rowLabel}>
                            <TableCell sx={{ whiteSpace: "nowrap" }}>{rowLabel}</TableCell>
                            {colLabels.map((colLabel, colIndex) => {
                                const value = values[rowIndex]?.[colIndex] ?? 0;
                                return (
                                    <TableCell
                                        key={colLabel}
                                        align="center"
                                        sx={{ bgcolor: heatColor(value, max), p: 0.5 }}
                                    >
                                        {onCellClick ? (
                                            <Button
                                                size="small"
                                                onClick={() => onCellClick(rowIndex, colIndex)}
                                                sx={{ minWidth: 56 }}
                                            >
                                                {fmt(value)}
                                            </Button>
                                        ) : (
                                            fmt(value)
                                        )}
                                    </TableCell>
                                );
                            })}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </Box>
    );
}

export function ResultsInspector({
    title = "raw data",
    data,
}: {
    title?: string;
    data: unknown;
}) {
    const [open, setOpen] = useState(false);
    if (data == null) return null;
    return (
        <Stack spacing={1}>
            <Button size="small" variant="text" onClick={() => setOpen((v) => !v)}>
                {open ? "Hide" : "Inspect"} {title}
            </Button>
            {open ? <JsonBlock data={data} /> : null}
        </Stack>
    );
}

export function MetricCards({
    items,
}: {
    items: Array<{ label: string; value: string | number | null | undefined }>;
}) {
    return (
        <Box
            sx={{
                display: "grid",
                gap: 1,
                gridTemplateColumns: { xs: "1fr 1fr", md: "repeat(4, 1fr)" },
            }}
        >
            {items.map((item) => (
                <Box key={item.label} sx={{ p: 1.5, borderRadius: 1, bgcolor: "action.hover" }}>
                    <Typography variant="caption" color="text.secondary">
                        {item.label}
                    </Typography>
                    <Typography variant="h6">
                        {item.value == null || item.value === "" ? "—" : String(item.value)}
                    </Typography>
                </Box>
            ))}
        </Box>
    );
}
