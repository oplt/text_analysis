import { useMemo, useState, type ReactNode } from "react";
import {
    Box,
    Button,
    Drawer,
    IconButton,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    ToggleButton,
    ToggleButtonGroup,
    Tooltip,
    Typography,
} from "@mui/material";
import { Download as DownloadIcon, InfoOutlined as InfoIcon } from "@mui/icons-material";
import { researchExportUrl } from "../../../api/textResearch";
import type { AnalysisRun } from "../types";
import { JsonBlock, RunStatusChip } from "./ResearchShared";
import { MetricCards, ResultsInspector } from "./ResearchCharts";
import {
    formatProvenanceValue,
    getRunResultProvenance,
} from "./resultProvenance";

export type ResearchResultsColumn<Row> = {
    id: string;
    label: string;
    value: (row: Row) => string | number | null | undefined;
    align?: "left" | "right" | "center";
};

export function ChartTableToggle({
    value,
    onChange,
}: {
    value: "chart" | "table" | "both";
    onChange: (value: "chart" | "table" | "both") => void;
}) {
    return (
        <ToggleButtonGroup
            size="small"
            exclusive
            value={value}
            onChange={(_, next) => next && onChange(next)}
            aria-label="Result display"
        >
            <ToggleButton value="chart">Chart</ToggleButton>
            <ToggleButton value="table">Table</ToggleButton>
            <ToggleButton value="both">Both</ToggleButton>
        </ToggleButtonGroup>
    );
}

export function ResearchResultsTable<Row extends { id?: string | number }>({
    columns,
    rows,
    emptyMessage = "No rows were returned for this analysis.",
    pageSize = 25,
}: {
    columns: Array<ResearchResultsColumn<Row>>;
    rows: Row[];
    emptyMessage?: string;
    pageSize?: number;
}) {
    const [sort, setSort] = useState<{ id: string; direction: "asc" | "desc" } | null>(null);
    const [page, setPage] = useState(0);
    const sortedRows = useMemo(() => {
        if (!sort) return rows;
        const column = columns.find((item) => item.id === sort.id);
        if (!column) return rows;
        return [...rows].sort((a, b) => {
            const left = column.value(a);
            const right = column.value(b);
            const comparison = typeof left === "number" && typeof right === "number"
                ? left - right
                : String(left ?? "").localeCompare(String(right ?? ""), undefined, { numeric: true });
            return sort.direction === "asc" ? comparison : -comparison;
        });
    }, [columns, rows, sort]);
    const pageCount = Math.max(1, Math.ceil(sortedRows.length / pageSize));
    const currentPage = Math.min(page, pageCount - 1);
    const visibleRows = sortedRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

    function toggleSort(id: string) {
        setPage(0);
        setSort((current) =>
            current?.id === id
                ? { id, direction: current.direction === "asc" ? "desc" : "asc" }
                : { id, direction: "asc" }
        );
    }

    if (!rows.length) return <Typography color="text.secondary">{emptyMessage}</Typography>;
    return (
        <Stack spacing={1}>
            <Box sx={{ overflowX: "auto" }}>
                <Table size="small" aria-label="Analysis result table">
                    <TableHead>
                        <TableRow>
                            {columns.map((column) => (
                                <TableCell key={column.id} align={column.align} sortDirection={sort?.id === column.id ? sort.direction : false}>
                                    <Button size="small" color="inherit" onClick={() => toggleSort(column.id)}>
                                        {column.label}{sort?.id === column.id ? (sort.direction === "asc" ? " ↑" : " ↓") : ""}
                                    </Button>
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {visibleRows.map((row, index) => (
                            <TableRow key={row.id ?? `${currentPage}-${index}`} hover>
                                {columns.map((column) => {
                                    const value = column.value(row);
                                    return <TableCell key={column.id} align={column.align}>{value == null || value === "" ? "—" : String(value)}</TableCell>;
                                })}
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </Box>
            {pageCount > 1 ? (
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="flex-end">
                    <Button size="small" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</Button>
                    <Typography variant="caption">Page {currentPage + 1} of {pageCount}</Typography>
                    <Button size="small" disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)}>Next</Button>
                </Stack>
            ) : null}
        </Stack>
    );
}

export function MethodsAndProvenanceDrawer({ run }: { run: AnalysisRun }) {
    const [open, setOpen] = useState(false);
    const parameters = run.parameters ?? {};
    const resultProvenance = getRunResultProvenance(run);
    const section = (title: string, keys: string[]) => {
        const values = keys
            .filter((key) => parameters[key] != null)
            .map((key) => `${key.replace(/_/g, " ")}: ${typeof parameters[key] === "object" ? JSON.stringify(parameters[key]) : String(parameters[key])}`);
        return values.length ? <Box><Typography variant="subtitle2">{title}</Typography>{values.map((value) => <Typography key={value} variant="body2" color="text.secondary">{value}</Typography>)}</Box> : null;
    };
    return (
        <>
            <Tooltip title="Methods and provenance"><IconButton aria-label="Methods and provenance" onClick={() => setOpen(true)}><InfoIcon /></IconButton></Tooltip>
            <Drawer anchor="right" open={open} onClose={() => setOpen(false)}>
                <Stack spacing={2} sx={{ width: { xs: "min(100vw, 360px)", sm: 420 }, p: 3 }}>
                    <Typography variant="h6">Methods and provenance</Typography>
                    {section("Data", ["corpus_id", "document_ids", "unit_type", "filters"])}
                    {section("Preprocessing", ["preprocessing_profile_id", "preprocessing_config", "ngram_range", "min_df", "max_df"])}
                    {section("Measurement", ["codebook_id", "codebook_version", "annotation_source", "minimum_agreement", "provenance_mode"])}
                    {section("Model", ["algorithm", "dataset_snapshot_id", "random_seed", "test_size", "class_weight", "C", "grouped_split"])}
                    {resultProvenance.runtime ? (
                        <Box>
                            <Typography variant="subtitle2">Runtime</Typography>
                            {Object.entries(resultProvenance.runtime).map(([key, value]) => (
                                <Typography key={key} variant="body2" color="text.secondary">
                                    {key.replace(/_/g, " ")}: {formatProvenanceValue(value)}
                                </Typography>
                            ))}
                        </Box>
                    ) : null}
                    {resultProvenance.identity ? (
                        <Box>
                            <Typography variant="subtitle2">Result identity</Typography>
                            {Object.entries(resultProvenance.identity).map(([key, value]) => (
                                <Typography key={key} variant="body2" color="text.secondary">
                                    {key.replace(/_/g, " ")}: {formatProvenanceValue(value)}
                                </Typography>
                            ))}
                        </Box>
                    ) : null}
                    {resultProvenance.artifacts.length ? (
                        <Box>
                            <Typography variant="subtitle2">Artifacts</Typography>
                            {resultProvenance.artifacts.map((artifact, index) => (
                                <Typography key={index} variant="body2" color="text.secondary">
                                    {formatProvenanceValue(artifact)}
                                </Typography>
                            ))}
                        </Box>
                    ) : null}
                    <Box>
                        <Typography variant="subtitle2">Reproducibility</Typography>
                        <Typography variant="body2" color="text.secondary">AnalysisRun ID: {run.id}</Typography>
                        <Typography variant="body2" color="text.secondary">Created: {new Date(run.created_at).toLocaleString()}</Typography>
                        <Typography variant="body2" color="text.secondary">Completed: {run.completed_at ? new Date(run.completed_at).toLocaleString() : "—"}</Typography>
                        <Typography variant="body2" color="text.secondary">Artifact: {run.artifact_path ?? "—"}</Typography>
                        <ResearchExportActions runId={run.id} />
                    </Box>
                    <JsonBlock data={{
                        parameters,
                        random_seed: run.random_seed,
                        artifact_path: run.artifact_path,
                        result_provenance: resultProvenance,
                    }} />
                </Stack>
            </Drawer>
        </>
    );
}

export function ResearchExportActions({ runId }: { runId: string }) {
    return <Button size="small" startIcon={<DownloadIcon />} href={researchExportUrl(`/research/runs/${runId}/export.json`)} target="_blank" rel="noopener">Export JSON</Button>;
}

export function ResearchResultPanel({
    run,
    title,
    metricItems,
    children,
}: {
    run: AnalysisRun;
    title: string;
    metricItems?: Array<{ label: string; value: string | number | null | undefined }>;
    children: ReactNode;
}) {
    const resultProvenance = getRunResultProvenance(run);
    const runtimeLabel = [
        resultProvenance.runtime?.engine,
        resultProvenance.runtime?.implementation,
    ].filter((part): part is string => typeof part === "string" && Boolean(part)).join(" / ");
    return (
        <Stack spacing={2}>
            <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                <Box>
                    <Typography variant="h6">{title}</Typography>
                    <Typography variant="caption" color="text.secondary">Run {run.id} · {new Date(run.created_at).toLocaleString()}</Typography>
                    {runtimeLabel ? (
                        <Typography variant="caption" display="block" color="text.secondary">
                            Runtime: {runtimeLabel}
                        </Typography>
                    ) : null}
                </Box>
                <Stack direction="row" spacing={0.5} alignItems="center">
                    <RunStatusChip status={run.status} />
                    <MethodsAndProvenanceDrawer run={run} />
                    <ResearchExportActions runId={run.id} />
                </Stack>
            </Stack>
            {metricItems?.length ? <MetricCards items={metricItems} /> : null}
            {children}
            <ResultsInspector title="raw JSON" data={{ metrics: run.metrics, results: run.results }} />
        </Stack>
    );
}
