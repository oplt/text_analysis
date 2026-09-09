import { useState } from "react";
import {
    Alert,
    Button,
    Chip,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import { FactCheck as QaIcon } from "@mui/icons-material";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
    getDocumentIngestionQa,
    runIngestionQa,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { useSnackbar } from "../../../app/snackbarContext";
import { useResearchContext } from "../hooks/useResearchContext";
import type { IngestionQaDocument } from "../types";

function severityColor(severity: string): "error" | "warning" | "info" | "default" {
    return severity === "error" || severity === "warning" || severity === "info"
        ? severity
        : "default";
}

export function IngestionQaPanel() {
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [runId, setRunId] = useState<string | null>(null);
    const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

    const qaMutation = useMutation({
        mutationFn: () => runIngestionQa(ctx.selectedCorpusId),
        onSuccess: (run) => {
            setRunId(run.id);
            setSelectedDocumentId(null);
            showToast({ message: "Ingestion QA completed.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Ingestion QA failed."),
                severity: "error",
            }),
    });

    const findingsQuery = useQuery({
        queryKey: ["text-research", "ingestion-qa", runId, selectedDocumentId],
        queryFn: () => getDocumentIngestionQa(selectedDocumentId!, runId ?? undefined),
        enabled: Boolean(runId && selectedDocumentId),
    });

    const runDocuments = ((qaMutation.data?.results as Record<string, unknown> | null)?.documents ?? []) as IngestionQaDocument[];
    const counts = (qaMutation.data?.metrics as Record<string, unknown> | null)?.finding_counts as
        | Record<string, number>
        | undefined;

    return (
        <Stack spacing={2}>
            <SectionCard
                title="Ingestion quality assurance"
                description="Inspect extraction quality without changing raw or canonical text."
                action={
                    <Button
                        variant="contained"
                        startIcon={<QaIcon />}
                        onClick={() => qaMutation.mutate()}
                        disabled={qaMutation.isPending || !ctx.selectedCorpusId}
                    >
                        {qaMutation.isPending ? "Checking corpus…" : "Run QA"}
                    </Button>
                }
            >
                <Stack spacing={1.5}>
                    <Typography variant="body2" color="text.secondary">
                        Checks include missing source text, extraction failures, duplicate content, and unusual document lengths.
                    </Typography>
                    {counts ? (
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            {Object.entries(counts).map(([severity, count]) => (
                                <Chip
                                    key={severity}
                                    size="small"
                                    color={severityColor(severity)}
                                    label={`${count} ${severity}${count === 1 ? "" : "s"}`}
                                />
                            ))}
                        </Stack>
                    ) : null}
                    {qaMutation.data?.error_message ? (
                        <Alert severity="error">{qaMutation.data.error_message}</Alert>
                    ) : null}
                </Stack>
            </SectionCard>

            {!qaMutation.data ? (
                <EmptyState
                    icon={<QaIcon fontSize="large" />}
                    title="No QA results yet"
                    description="Run QA after documents have been uploaded to inspect each document’s findings."
                />
            ) : (
                <SectionCard title="Document findings" description="Select a document to inspect its complete QA record.">
                    {runDocuments.length ? (
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Document</TableCell>
                                    <TableCell>Language</TableCell>
                                    <TableCell>Findings</TableCell>
                                    <TableCell align="right" />
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {runDocuments.map((document) => (
                                    <TableRow key={document.document_id} selected={selectedDocumentId === document.document_id}>
                                        <TableCell>{document.title ?? document.document_id}</TableCell>
                                        <TableCell>{document.language ?? "—"}</TableCell>
                                        <TableCell>
                                            {document.findings.length ? (
                                                <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                                                    {document.findings.map((finding) => (
                                                        <Chip
                                                            key={`${document.document_id}-${finding.code}`}
                                                            size="small"
                                                            color={severityColor(finding.severity)}
                                                            label={finding.code.replaceAll("_", " ")}
                                                        />
                                                    ))}
                                                </Stack>
                                            ) : (
                                                "No findings"
                                            )}
                                        </TableCell>
                                        <TableCell align="right">
                                            <Button size="small" onClick={() => setSelectedDocumentId(document.document_id)}>
                                                Inspect
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    ) : (
                        <Typography variant="body2" color="text.secondary">No documents were available for QA.</Typography>
                    )}
                </SectionCard>
            )}

            {selectedDocumentId ? (
                <SectionCard title="Selected document findings">
                    <QueryBoundary
                        isLoading={findingsQuery.isLoading}
                        isError={findingsQuery.isError}
                        error={findingsQuery.error}
                        onRetry={() => void findingsQuery.refetch()}
                        variant="inline"
                    >
                        <Stack spacing={1}>
                            {findingsQuery.data?.findings.length ? findingsQuery.data.findings.map((finding) => (
                                <Alert key={finding.code} severity={finding.severity === "error" || finding.severity === "warning" ? finding.severity : "info"}>
                                    <strong>{finding.code.replaceAll("_", " ")}: </strong>{finding.message}
                                </Alert>
                            )) : <Alert severity="success">No findings for this document.</Alert>}
                        </Stack>
                    </QueryBoundary>
                </SectionCard>
            ) : null}
        </Stack>
    );
}
