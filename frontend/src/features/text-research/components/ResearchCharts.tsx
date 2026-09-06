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
    Tooltip,
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

export function ReliabilityComparisonChart({
    items,
    onSelect,
}: {
    items: Array<{ label: string; kappa: number | null; alpha: number | null }>;
    onSelect?: (label: string) => void;
}) {
    if (!items.length) return <Typography color="text.secondary">No evaluable reliability metrics to plot.</Typography>;
    return (
        <Box sx={{ width: "100%" }}>
            <BarChart
                height={Math.max(280, items.length * 46)}
                layout="horizontal"
                yAxis={[{ data: items.map((item) => item.label), scaleType: "band", width: 140 }]}
                xAxis={[{ min: -1, max: 1 }]}
                series={[
                    { data: items.map((item) => item.kappa), label: "Cohen's κ" },
                    { data: items.map((item) => item.alpha), label: "Krippendorff's α" },
                ]}
                onAxisClick={(_, detail) => {
                    const label = items[detail?.dataIndex ?? -1]?.label;
                    if (label) onSelect?.(label);
                }}
            />
        </Box>
    );
}

export function ScientificLineChart({
    series,
    height = 280,
}: {
    series: Array<{ label: string; points: Array<{ x: number; y: number }> }>;
    height?: number;
}) {
    const points = series.flatMap((item) => item.points);
    if (!points.length) return <Typography color="text.secondary">No time-series values to plot.</Typography>;
    const width = 720;
    const padding = 42;
    const xMin = Math.min(...points.map((point) => point.x));
    const xMax = Math.max(...points.map((point) => point.x));
    const yMin = Math.min(0, ...points.map((point) => point.y));
    const yMax = Math.max(...points.map((point) => point.y));
    const x = (value: number) => padding + ((value - xMin) / Math.max(1, xMax - xMin)) * (width - padding * 2);
    const y = (value: number) => height - padding - ((value - yMin) / Math.max(0.0001, yMax - yMin)) * (height - padding * 2);
    const colors = ["#2d6a4f", "#4361ee", "#b45309", "#9d174d", "#0f766e"];
    return (
        <Box sx={{ overflowX: "auto" }}>
            <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Time-series chart" style={{ width: "100%", minWidth: 500, height }}>
                <line x1={padding} x2={padding} y1={padding} y2={height - padding} stroke="currentColor" opacity="0.35" />
                <line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} stroke="currentColor" opacity="0.35" />
                {series.map((item, index) => (
                    <g key={item.label}>
                        <polyline fill="none" stroke={colors[index % colors.length]} strokeWidth="2" points={item.points.map((point) => `${x(point.x)},${y(point.y)}`).join(" ")} />
                        {item.points.map((point) => <circle key={`${point.x}-${point.y}`} cx={x(point.x)} cy={y(point.y)} r="3" fill={colors[index % colors.length]}><title>{`${item.label}: ${point.x}, ${point.y.toFixed(3)}`}</title></circle>)}
                    </g>
                ))}
                <text x={padding} y={height - 10} fontSize="12">{xMin}</text>
                <text x={width - padding - 28} y={height - 10} fontSize="12">{xMax}</text>
            </svg>
        </Box>
    );
}

export function ScientificScatterPlot({
    points,
    height = 300,
}: {
    points: Array<{ x: number; y: number; label: string; detail: string }>;
    height?: number;
}) {
    if (!points.length) return <Typography color="text.secondary">No joined observations to plot.</Typography>;
    const width = 720;
    const padding = 42;
    const xMin = Math.min(...points.map((point) => point.x));
    const xMax = Math.max(...points.map((point) => point.x));
    const yMin = Math.min(0, ...points.map((point) => point.y));
    const yMax = Math.max(...points.map((point) => point.y));
    const x = (value: number) => padding + ((value - xMin) / Math.max(0.0001, xMax - xMin)) * (width - padding * 2);
    const y = (value: number) => height - padding - ((value - yMin) / Math.max(0.0001, yMax - yMin)) * (height - padding * 2);
    return <Box sx={{ overflowX: "auto" }}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Contextual indicator and prevalence scatter plot" style={{ width: "100%", minWidth: 500, height }}>
        <line x1={padding} x2={padding} y1={padding} y2={height - padding} stroke="currentColor" opacity="0.35" /><line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} stroke="currentColor" opacity="0.35" />
        {points.map((point) => <circle key={`${point.label}:${point.x}:${point.y}`} cx={x(point.x)} cy={y(point.y)} r="5" fill="#2d6a4f"><title>{`${point.label}\n${point.detail}`}</title></circle>)}
    </svg></Box>;
}

export function CooccurrenceNetwork({
    edges,
    onNodeClick,
}: {
    edges: Array<{ termA: string; termB: string; count: number; association: number }>;
    onNodeClick?: (term: string) => void;
}) {
    const terms = Array.from(new Set(edges.flatMap((edge) => [edge.termA, edge.termB])));
    if (!edges.length) return <Typography color="text.secondary">No co-occurrence edges meet the selected threshold.</Typography>;
    const width = 720;
    const height = 420;
    const centerX = width / 2;
    const centerY = height / 2;
    const positions = new Map(terms.map((term, index) => [term, { x: centerX + 150 * Math.cos((Math.PI * 2 * index) / terms.length), y: centerY + 150 * Math.sin((Math.PI * 2 * index) / terms.length) }]));
    const degree = new Map<string, number>();
    edges.forEach((edge) => { degree.set(edge.termA, (degree.get(edge.termA) ?? 0) + edge.count); degree.set(edge.termB, (degree.get(edge.termB) ?? 0) + edge.count); });
    const maxCount = Math.max(...edges.map((edge) => edge.count));
    const maxDegree = Math.max(...degree.values());
    return <Box sx={{ overflowX: "auto" }}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Co-occurrence network" style={{ width: "100%", minWidth: 500, height }}>
        {edges.map((edge) => { const a = positions.get(edge.termA)!; const b = positions.get(edge.termB)!; return <line key={`${edge.termA}:${edge.termB}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#4361ee" opacity="0.45" strokeWidth={1 + (edge.count / maxCount) * 5}><title>{`${edge.termA} × ${edge.termB}: ${edge.count}, association ${edge.association.toFixed(3)}`}</title></line>; })}
        {terms.map((term) => { const point = positions.get(term)!; return <g key={term} onClick={() => onNodeClick?.(term)} style={{ cursor: onNodeClick ? "pointer" : "default" }}><circle cx={point.x} cy={point.y} r={8 + ((degree.get(term) ?? 0) / maxDegree) * 16} fill="#2d6a4f" /><text x={point.x} y={point.y + 4} textAnchor="middle" fill="#f8fafc" fontSize="10">{term.slice(0, 8)}</text><title>{`${term}: weighted degree ${degree.get(term)}`}</title></g>; })}
    </svg></Box>;
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
    cellDetails,
}: {
    rowLabels: string[];
    colLabels: string[];
    values: Array<Array<number | null>>;
    onCellClick?: (rowIndex: number, colIndex: number) => void;
    formatCell?: (value: number | null) => string;
    cellDetails?: (rowIndex: number, colIndex: number) => string | undefined;
}) {
    if (!rowLabels.length || !colLabels.length) {
        return (
            <Typography variant="body2" color="text.secondary">
                No matrix data.
            </Typography>
        );
    }
    const numericValues = values.flatMap((row) => row.filter((value): value is number => value !== null));
    const max = Math.max(0.0001, ...numericValues);
    const fmt = formatCell ?? ((v: number | null) => (v === null ? "—" : v.toFixed(2)));
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
                                const value = values[rowIndex]?.[colIndex] ?? null;
                                const content = onCellClick && value !== null ? (
                                    <Button
                                        size="small"
                                        onClick={() => onCellClick(rowIndex, colIndex)}
                                        sx={{ minWidth: 56 }}
                                    >
                                        {fmt(value)}
                                    </Button>
                                ) : (
                                    fmt(value)
                                );
                                return (
                                    <TableCell
                                        key={colLabel}
                                        align="center"
                                        sx={{
                                            bgcolor: value === null ? "action.disabledBackground" : heatColor(value, max),
                                            color: value === null ? "text.secondary" : undefined,
                                            p: 0.5,
                                        }}
                                    >
                                        {cellDetails ? (
                                            <Tooltip title={cellDetails(rowIndex, colIndex) ?? ""} arrow>
                                                <Box component="span">{content}</Box>
                                            </Tooltip>
                                        ) : content}
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
