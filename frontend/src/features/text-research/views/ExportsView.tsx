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
    researchExportUrl,
} from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { JsonBlock } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";

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

    const manifestQuery = useQuery({
        queryKey: queryKeys.textResearch.exportManifest(ctx.selectedCorpusId),
        queryFn: () => getExportManifest(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const scriptQuery = useQuery({
        queryKey: ["text-research", "quanteda-script", ctx.selectedCorpusId],
        queryFn: () => getQuantedaScript(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const profilesQuery = useQuery({
        queryKey: queryKeys.textResearch.preprocessingProfiles(ctx.projectId),
        queryFn: () => listPreprocessingProfiles(ctx.projectId),
        enabled: Boolean(ctx.projectId),
    });

    const classifiersQuery = useQuery({
        queryKey: queryKeys.textResearch.classifiers(ctx.projectId, ctx.selectedCorpusId),
        queryFn: () => listClassifiers(ctx.projectId, ctx.selectedCorpusId),
        enabled: Boolean(ctx.projectId && ctx.selectedCorpusId),
    });

    const handleDownload = useCallback(
        async (path: string, filename: string) => {
            try {
                await downloadAuthenticated(path, filename);
                showToast({ message: `Downloaded ${filename}.`, severity: "success" });
            } catch (error) {
                showToast({
                    message: getQueryErrorMessage(error, "Download failed."),
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
            <SectionCard title="Export manifest" description="Overview of exportable artifacts for this corpus.">
                <QueryBoundary
                    isLoading={manifestQuery.isLoading}
                    isError={manifestQuery.isError}
                    error={manifestQuery.error}
                    onRetry={() => void manifestQuery.refetch()}
                >
                    {manifestQuery.data ? <JsonBlock data={manifestQuery.data.manifest} /> : null}
                </QueryBoundary>
            </SectionCard>

            <SectionCard title="CSV downloads" description="Download tabular exports for offline analysis.">
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
                                        void handleDownload(unitsPath, `units-${ctx.unitType}.csv`)
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
                <Alert severity="info" sx={{ mt: 2 }}>
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
            </SectionCard>

            <SectionCard
                title="Reproducibility JSON"
                description="Export codebooks, preprocessing profiles, model metrics, and individual robustness or analysis runs."
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
                                <TableCell>Model metrics: {model.name || `v${model.version}`}</TableCell>
                                <TableCell>Classifier metrics, features, and training configuration</TableCell>
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
                    </TableBody>
                </Table>
                <Alert severity="info" sx={{ mt: 2 }}>
                    Export a robustness result or any other analysis from its Run detail page.
                </Alert>
            </SectionCard>

            <SectionCard title="Quanteda script" description="R script for reproducing corpus construction.">
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
        </Stack>
    );
}
