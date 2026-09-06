import { useState } from "react";
import {
    Alert,
    Button,
    MenuItem,
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
import {
    Link as LinkIcon,
    UploadFile as UploadIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    createContextualDataset,
    deleteContextualDataset,
    getContextualDataset,
    importContextualCsv,
    linkContextualDiscourse,
    listContextualDatasets,
    listContextualObservations,
    type ContextualLinkResult,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { MetricCards, ResultsInspector, ScientificLineChart, ScientificScatterPlot } from "../components/ResearchCharts";
import { NoCorpusEmptyState } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

export default function ContextualView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const { showToast } = useSnackbar();

    const [selectedDatasetId, setSelectedDatasetId] = useState("");
    const [name, setName] = useState("Country-year indicators");
    const [description, setDescription] = useState("");
    const [indicatorKey, setIndicatorKey] = useState("");
    const [groupBy, setGroupBy] = useState<"country" | "publication_year">("country");
    const [selectedLabelId, setSelectedLabelId] = useState("");
    const [linkResult, setLinkResult] = useState<ContextualLinkResult | null>(null);
    const [observationPage, setObservationPage] = useState(0);
    const observationPageSize = 50;

    const datasetsQuery = useQuery({
        queryKey: queryKeys.textResearch.contextualDatasets(ctx.projectId),
        queryFn: () => listContextualDatasets(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const datasets = datasetsQuery.data ?? [];
    const activeDatasetId = selectedDatasetId || datasets[0]?.id || "";

    const detailQuery = useQuery({
        queryKey: queryKeys.textResearch.contextualDataset(activeDatasetId),
        queryFn: () => getContextualDataset(activeDatasetId),
        enabled: Boolean(activeDatasetId),
    });

    const observationsQuery = useQuery({
        queryKey: ["text-research", "contextual-observations", activeDatasetId, observationPage],
        queryFn: () =>
            listContextualObservations(activeDatasetId, {
                limit: observationPageSize,
                offset: observationPage * observationPageSize,
            }),
        enabled: Boolean(activeDatasetId),
    });

    const detail = detailQuery.data;
    const indicatorKeys = detail?.indicator_keys ?? [];
    const activeIndicator = indicatorKey || indicatorKeys[0] || "";
    const activeLabelId = selectedLabelId || ctx.labels[0]?.id || "";

    const createMutation = useMutation({
        mutationFn: () =>
            createContextualDataset(ctx.projectId, {
                name: name.trim(),
                description: description.trim() || undefined,
            }),
        onSuccess: (dataset) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.contextualDatasets(ctx.projectId),
            });
            setSelectedDatasetId(dataset.id);
            showToast({ message: "Contextual dataset created.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create dataset."),
                severity: "error",
            }),
    });

    const importMutation = useMutation({
        mutationFn: (file: File) => importContextualCsv(activeDatasetId, file, true),
        onSuccess: (result) => {
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.contextualDatasets(ctx.projectId),
            });
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.contextualDataset(activeDatasetId),
            });
            void client.invalidateQueries({
                queryKey: ["text-research", "contextual-observations", activeDatasetId],
            });
            setObservationPage(0);
            if (result.indicator_keys[0]) setIndicatorKey(result.indicator_keys[0]);
            showToast({
                message: `Imported ${result.imported} observations (${result.skipped} skipped).`,
                severity: "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "CSV import failed."),
                severity: "error",
            }),
    });

    const deleteMutation = useMutation({
        mutationFn: () => deleteContextualDataset(activeDatasetId),
        onSuccess: () => {
            setSelectedDatasetId("");
            setLinkResult(null);
            void client.invalidateQueries({
                queryKey: queryKeys.textResearch.contextualDatasets(ctx.projectId),
            });
            showToast({ message: "Contextual dataset deleted.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to delete dataset."),
                severity: "error",
            }),
    });

    const linkMutation = useMutation({
        mutationFn: () =>
            linkContextualDiscourse(activeDatasetId, {
                corpus_id: ctx.selectedCorpusId,
                codebook_id: ctx.selectedCodebookId,
                label_ids: activeLabelId ? [activeLabelId] : ctx.labels.map((l) => l.id),
                indicator_key: activeIndicator,
                unit_type: ctx.unitType,
                group_by: groupBy,
            }),
        onSuccess: (result) => {
            setLinkResult(result);
            showToast({ message: "Exploratory join completed.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to link discourse."),
                severity: "error",
            }),
    });

    const activeLabelName = ctx.labels.find((l) => l.id === activeLabelId)?.name;

    if (!ctx.corporaLoading && ctx.corpora.length === 0) {
        return <NoCorpusEmptyState />;
    }

    const canLink =
        Boolean(activeDatasetId && activeIndicator && ctx.selectedCorpusId && ctx.selectedCodebookId) &&
        ctx.labels.length > 0;

    return (
        <Stack spacing={2}>
            <Alert severity="info">
                Contextual joins are exploratory. Associations between discourse prevalence and
                indicators (trust, well-being, polarization, etc.) are descriptive only and are not
                causal estimates.
            </Alert>

            <SectionCard
                title="Contextual datasets"
                description="Import country/year indicator tables, then join them to discourse prevalence."
            >
                <Stack spacing={2}>
                    <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                        <TextField
                            size="small"
                            label="Dataset name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            sx={{ minWidth: 220 }}
                        />
                        <TextField
                            size="small"
                            label="Description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            sx={{ flex: 1 }}
                        />
                        <Button
                            variant="contained"
                            onClick={() => createMutation.mutate()}
                            disabled={!name.trim() || createMutation.isPending}
                        >
                            Create dataset
                        </Button>
                    </Stack>

                    <QueryBoundary
                        isLoading={datasetsQuery.isLoading}
                        isError={datasetsQuery.isError}
                        error={datasetsQuery.error}
                        onRetry={() => void datasetsQuery.refetch()}
                        variant="inline"
                    >
                        {!datasets.length ? (
                            <EmptyState
                                icon={<UploadIcon fontSize="large" />}
                                title="No contextual datasets yet"
                                description="Create a dataset, then upload a CSV with country and/or year plus indicator columns."
                            />
                        ) : (
                            <TextField
                                select
                                size="small"
                                label="Active dataset"
                                value={activeDatasetId}
                                onChange={(e) => {
                                    setSelectedDatasetId(e.target.value);
                                    setObservationPage(0);
                                    setLinkResult(null);
                                }}
                                sx={{ minWidth: 280 }}
                            >
                                {datasets.map((dataset) => (
                                    <MenuItem key={dataset.id} value={dataset.id}>
                                        {dataset.name} · {dataset.observation_count} rows
                                    </MenuItem>
                                ))}
                            </TextField>
                        )}
                    </QueryBoundary>
                </Stack>
            </SectionCard>

            {activeDatasetId ? (
                <SectionCard
                    title="Import indicators"
                    description="CSV headers should include country and/or year, plus one or more numeric indicator columns."
                    action={
                        <Stack direction="row" spacing={1}>
                            <Button
                                component="label"
                                variant="contained"
                                startIcon={<UploadIcon />}
                                disabled={importMutation.isPending}
                            >
                                Upload CSV
                                <input
                                    hidden
                                    type="file"
                                    accept=".csv,text/csv"
                                    onChange={(event) => {
                                        const file = event.target.files?.[0];
                                        if (file) importMutation.mutate(file);
                                        event.target.value = "";
                                    }}
                                />
                            </Button>
                            <Button
                                color="error"
                                variant="outlined"
                                onClick={() => deleteMutation.mutate()}
                                disabled={deleteMutation.isPending}
                            >
                                Delete
                            </Button>
                        </Stack>
                    }
                >
                    <QueryBoundary
                        isLoading={detailQuery.isLoading}
                        isError={detailQuery.isError}
                        error={detailQuery.error}
                        onRetry={() => void detailQuery.refetch()}
                        variant="inline"
                    >
                        {detail ? (
                            <Stack spacing={1.5}>
                                <MetricCards
                                    items={[
                                        { label: "Observations", value: detail.observation_count },
                                        {
                                            label: "Indicators",
                                            value: detail.indicator_keys.length,
                                        },
                                        { label: "Displayed", value: observationsQuery.data?.items.length ?? 0 },
                                    ]}
                                />
                                <Typography variant="body2" color="text.secondary">
                                    Indicators:{" "}
                                    {detail.indicator_keys.length
                                        ? detail.indicator_keys.join(" · ")
                                        : "none yet — upload a CSV"}
                                </Typography>
                                <QueryBoundary
                                    isLoading={observationsQuery.isLoading}
                                    isError={observationsQuery.isError}
                                    error={observationsQuery.error}
                                    onRetry={() => void observationsQuery.refetch()}
                                    variant="inline"
                                >
                                    {observationsQuery.data?.items.length ? (
                                        <>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Country</TableCell>
                                                        <TableCell>Year</TableCell>
                                                        <TableCell>Values</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {observationsQuery.data.items.map((row) => (
                                                        <TableRow key={row.id}>
                                                            <TableCell>{row.country ?? "—"}</TableCell>
                                                            <TableCell>{row.year ?? "—"}</TableCell>
                                                            <TableCell>
                                                                {Object.entries(row.values)
                                                                    .slice(0, 4)
                                                                    .map(([key, value]) => `${key}=${value}`)
                                                                    .join(" · ")}
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
                                            <TablePagination
                                                component="div"
                                                count={observationsQuery.data.total}
                                                page={observationPage}
                                                rowsPerPage={observationPageSize}
                                                rowsPerPageOptions={[observationPageSize]}
                                                onPageChange={(_, nextPage) => setObservationPage(nextPage)}
                                            />
                                        </>
                                    ) : (
                                        <Typography variant="body2" color="text.secondary">
                                            No observations yet — upload a CSV to add indicators.
                                        </Typography>
                                    )}
                                </QueryBoundary>
                            </Stack>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}

            <SectionCard
                title="Link discourse prevalence"
                description="Join codebook label prevalence by country or year with a contextual indicator."
            >
                <Stack spacing={2}>
                    <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                        <TextField
                            select
                            size="small"
                            label="Indicator"
                            value={activeIndicator}
                            onChange={(e) => setIndicatorKey(e.target.value)}
                            disabled={!indicatorKeys.length}
                            sx={{ minWidth: 200 }}
                        >
                            {indicatorKeys.map((key) => (
                                <MenuItem key={key} value={key}>
                                    {key}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Group discourse by"
                            value={groupBy}
                            onChange={(e) =>
                                setGroupBy(e.target.value as "country" | "publication_year")
                            }
                            sx={{ minWidth: 200 }}
                        >
                            <MenuItem value="country">Country</MenuItem>
                            <MenuItem value="publication_year">Publication year</MenuItem>
                        </TextField>
                        <TextField
                            select
                            size="small"
                            label="Discourse label"
                            value={activeLabelId}
                            onChange={(e) => setSelectedLabelId(e.target.value)}
                            disabled={!ctx.labels.length}
                            sx={{ minWidth: 200 }}
                        >
                            {ctx.labels.map((label) => (
                                <MenuItem key={label.id} value={label.id}>
                                    {label.name}
                                </MenuItem>
                            ))}
                        </TextField>
                        <Button
                            variant="contained"
                            startIcon={<LinkIcon />}
                            onClick={() => linkMutation.mutate()}
                            disabled={!canLink || linkMutation.isPending}
                        >
                            Run exploratory join
                        </Button>
                    </Stack>

                    {!ctx.selectedCorpusId || !ctx.selectedCodebookId ? (
                        <Typography variant="body2" color="text.secondary">
                            Select a corpus and codebook in the workspace context bar first.
                        </Typography>
                    ) : null}

                    {linkResult ? (
                        <Stack spacing={2}>
                            <Alert severity="warning">{linkResult.disclaimer}</Alert>
                            <MetricCards
                                items={[
                                    { label: "Joined points", value: linkResult.point_count },
                                    {
                                        label: "Unmatched groups",
                                        value: linkResult.unmatched_groups,
                                    },
                                    {
                                        label: "Indicator",
                                        value: linkResult.indicator_key,
                                    },
                                ]}
                            />
                            {linkResult.group_by === "publication_year" ? (
                                <>
                                    <Typography variant="subtitle2">Discourse prevalence over time</Typography>
                                    <ScientificLineChart
                                        series={Object.entries(
                                            linkResult.points
                                                .filter((point) => !activeLabelName || point.label === activeLabelName)
                                                .reduce<Record<string, Array<{ x: number; y: number }>>>((acc, point) => {
                                                    const year = Number(point.group);
                                                    if (Number.isFinite(year)) (acc[point.label] ??= []).push({ x: year, y: point.prevalence });
                                                    return acc;
                                                }, {})
                                        ).map(([label, points]) => ({ label, points: points.sort((a, b) => a.x - b.x) }))}
                                    />
                                </>
                            ) : (
                                <>
                                    <Typography variant="subtitle2">Contextual indicator and discourse prevalence</Typography>
                                    <ScientificScatterPlot
                                        points={linkResult.points
                                            .filter((point) => !activeLabelName || point.label === activeLabelName)
                                            .map((point) => ({
                                                x: point.indicator_value,
                                                y: point.prevalence,
                                                label: point.group,
                                                detail: `${point.group_by}: ${point.group}; ${point.indicator_key}: ${point.indicator_value}; prevalence: ${point.prevalence.toFixed(3)}; yes: ${point.yes}; total: ${point.total}`,
                                            }))}
                                    />
                                </>
                            )}
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Label</TableCell>
                                        <TableCell>n</TableCell>
                                        <TableCell>Pearson r</TableCell>
                                        <TableCell>Status</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {Object.entries(linkResult.correlations).map(([label, row]) => (
                                        <TableRow key={label}>
                                            <TableCell>{label}</TableCell>
                                            <TableCell>{row.n}</TableCell>
                                            <TableCell>
                                                {row.pearson_r == null
                                                    ? "—"
                                                    : row.pearson_r.toFixed(3)}
                                            </TableCell>
                                            <TableCell>{row.status}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                            <ResultsInspector title="joined points" data={linkResult} />
                        </Stack>
                    ) : null}
                </Stack>
            </SectionCard>
        </Stack>
    );
}
