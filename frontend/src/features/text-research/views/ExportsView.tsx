import { useCallback } from "react";
import {
    Alert,
    Button,
    Link,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import { Download as DownloadIcon } from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { useSnackbar } from "../../../app/snackbarContext";
import {
    getExportManifest,
    getQuantedaScript,
    listClassifiers,
    listPreprocessingProfiles,
    listRuns,
    researchExportUrl,
} from "../../../api/textResearch";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { useTabQueryParam } from "../../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

const EXPORT_TABS = ["data", "models", "analysis", "reproducibility"] as const;
type ExportTab = (typeof EXPORT_TABS)[number];

const EXPORT_TAB_ITEMS: Array<{ value: ExportTab; label: string }> = [
    { value: "data", label: "Data" },
    { value: "models", label: "Models" },
    { value: "analysis", label: "Analysis" },
    { value: "reproducibility", label: "Reproducibility" },
];

async function downloadAuthenticated(path: string, filename: string) {
    const apiBase = import.meta.env.VITE_API_BASE ?? "/api/v1";
    const response = await fetch(`${apiBase}${path}`, { credentials: "include" });
    if (!response.ok) {
        throw new Error("Download failed");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

export default function ExportsView() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [tab, setTab] = useTabQueryParam(EXPORT_TABS, "data");

    const manifestQuery = useQuery({
        queryKey: queryKeys.textResearch.exportManifest(ctx.selectedCorpusId),
        queryFn: ({ signal }) => getExportManifest(ctx.selectedCorpusId, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const scriptQuery = useQuery({
        queryKey: ["text-research", "quanteda-script", ctx.selectedCorpusId],
        queryFn: () => getQuantedaScript(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId) && tab === "reproducibility",
    });

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: ({ signal }) => listPreprocessingProfiles(ctx.projectId, signal),
        enabled: Boolean(ctx.projectId) && tab === "models",
    });

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listClassifiers(ctx.projectId, ctx.selectedCorpusId, undefined, signal),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId) && tab === "models",
    });
    const runsQuery = useQuery({
        queryKey: queryKeys.textResearch.runs(ctx.projectId, ctx.selectedCorpusId),
        queryFn: ({ signal }) => listRuns(ctx.projectId, { corpus_id: ctx.selectedCorpusId, limit: 100 }, signal),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId) && tab === "analysis",
    });

    const handleDownload = useCallback(
        async (path: string, filename: string) => {
            try {
                await downloadAuthenticated(path, filename);
            } catch (error) {
                showToast({
                    message: getQueryErrorMessage(error, "Download failed"),
                    severity: "error",
                });
            }
        },
        [showToast]
    );

    if (!ctx.selectedCorpusId) {
        return (
            <SectionCard title="Exports">
                <Typography color="text.secondary">Select a corpus to view export options.</Typography>
            </SectionCard>
        );
    }

    const unitsPath = `/research/corpora/${ctx.selectedCorpusId}/export/units.csv?unit_type=${ctx.unitType}`;
    const annotationsPath = ctx.selectedCodebookId
        ? `/research/corpora/${ctx.selectedCorpusId}/export/annotations.csv?codebook_id=${ctx.selectedCodebookId}`
        : null;

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={EXPORT_TAB_ITEMS}
                ariaLabel="Export categories"
            />

            {tab === "data" ? (
                <SectionCard
                    title="Data exports"
                    description="Corpus manifest and tabular research downloads."
                >
                    <Stack spacing={2}>
                        <QueryBoundary
                            isLoading={manifestQuery.isLoading}
                            isError={manifestQuery.isError}
                            error={manifestQuery.error}
                            onRetry={() => void manifestQuery.refetch()}
                        >
                            {manifestQuery.data ? (
                                <JsonBlock data={manifestQuery.data.manifest} />
                            ) : null}
                        </QueryBoundary>

                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Export</TableCell>
                                    <TableCell>Description</TableCell>
                                    <TableCell align="right">Action</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                <TableRow>
                                    <TableCell>Text units</TableCell>
                                    <TableCell>Segmented units ({ctx.unitType})</TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    unitsPath,
                                                    `units-${ctx.unitType}.csv`
                                                )
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell>Annotations</TableCell>
                                    <TableCell>
                                        Annotations for selected codebook
                                        {!ctx.selectedCodebookId ? " (select a codebook)" : ""}
                                    </TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            disabled={!annotationsPath}
                                            onClick={() =>
                                                annotationsPath
                                                    ? void handleDownload(
                                                          annotationsPath,
                                                          "annotations.csv"
                                                      )
                                                    : undefined
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                        <Alert severity="info">
                            Direct links (requires session cookie):{" "}
                            <Link href={researchExportUrl(unitsPath)} target="_blank" rel="noopener">
                                units.csv
                            </Link>
                            {annotationsPath ? (
                                <>
                                    {" · "}
                                    <Link
                                        href={researchExportUrl(annotationsPath)}
                                        target="_blank"
                                        rel="noopener"
                                    >
                                        annotations.csv
                                    </Link>
                                </>
                            ) : null}
                        </Alert>
                    </Stack>
                </SectionCard>
            ) : null}

            {tab === "models" ? (
                <SectionCard
                    title="Measurement & models"
                    description="Export codebooks, preprocessing profiles, model metrics, and predictions."
                >
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Export</TableCell>
                                <TableCell>Description</TableCell>
                                <TableCell align="right">Action</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {ctx.selectedCodebookId ? (
                                <TableRow>
                                    <TableCell>Codebook</TableCell>
                                    <TableCell>Selected codebook with labels and version</TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    `/research/codebooks/${ctx.selectedCodebookId}/export.json`,
                                                    "codebook.json"
                                                )
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ) : null}
                            {(profilesQuery.data ?? []).map((profile) => (
                                <TableRow key={profile.id}>
                                    <TableCell>Preprocessing: {profile.name}</TableCell>
                                    <TableCell>Versioned preprocessing configuration</TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    `/research/preprocessing-profiles/${profile.id}/export.json`,
                                                    `preprocessing-${profile.id}.json`
                                                )
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {(classifiersQuery.data ?? []).map((model) => (
                                <TableRow key={model.id}>
                                    <TableCell>
                                        Model metrics: {model.name || `v${model.version}`}
                                    </TableCell>
                                    <TableCell>
                                        Classifier metrics, features, and training configuration
                                    </TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    `/research/classifiers/${model.id}/export/metrics.json`,
                                                    `model-${model.id}-metrics.json`
                                                )
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {(classifiersQuery.data ?? []).map((model) => (
                                <TableRow key={`${model.id}:predictions`}>
                                    <TableCell>
                                        Predictions: {model.name || `v${model.version}`}
                                    </TableCell>
                                    <TableCell>
                                        Predicted labels, scores, uncertainty, and model provenance
                                    </TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    `/research/classifiers/${model.id}/export/predictions.csv`,
                                                    `model-${model.id}-predictions.csv`
                                                )
                                            }
                                        >
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                    <Alert severity="info" sx={{ mt: 2 }}>
                        Export a robustness result or any other analysis from its Run detail page.
                    </Alert>
                </SectionCard>
            ) : null}

            {tab === "analysis" ? (
                <SectionCard title="Analysis" description="Completed analysis runs in portable JSON.">
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Run</TableCell>
                                <TableCell>Status</TableCell>
                                <TableCell align="right">Action</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {(runsQuery.data?.items ?? []).map((run) => (
                                <TableRow key={run.id}>
                                    <TableCell>
                                        {run.run_type} · {run.id.slice(0, 8)}
                                    </TableCell>
                                    <TableCell>{run.status}</TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            onClick={() =>
                                                void handleDownload(
                                                    `/research/runs/${run.id}/export.json`,
                                                    `run-${run.id}.json`
                                                )
                                            }
                                        >
                                            Download JSON
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </SectionCard>
            ) : null}

            {tab === "reproducibility" ? (
                <SectionCard
                    title="Reproducibility"
                    description="R script for reproducing corpus construction."
                >
                    <QueryBoundary
                        isLoading={scriptQuery.isLoading}
                        isError={scriptQuery.isError}
                        error={scriptQuery.error}
                        onRetry={() => void scriptQuery.refetch()}
                    >
                        {scriptQuery.data ? (
                            <Typography
                                component="pre"
                                sx={{
                                    p: 2,
                                    borderRadius: 2,
                                    bgcolor: "action.hover",
                                    overflow: "auto",
                                    fontSize: 12,
                                    whiteSpace: "pre-wrap",
                                }}
                            >
                                {scriptQuery.data.script}
                            </Typography>
                        ) : null}
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
