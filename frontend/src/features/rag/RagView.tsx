import { useEffect, useMemo, useRef, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    MenuItem,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TablePagination,
    TextField,
    Typography,
} from "@mui/material";
import {
    CloudUpload as UploadIcon,
    Refresh as RetryIcon,
    Search as RetrieveIcon,
    QuestionAnswer as AskIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../app/snackbarContext";
import { listProjects } from "../../api/projects";
import {
    askRag,
    deleteRagDocument,
    getRagIngestionJob,
    indexRagDocument,
    listRagChunks,
    listRagDocuments,
    listRagQueries,
    retrieveRagChunks,
    uploadRagDocument,
    type RagAskResult,
    type RagRetrieveResult,
} from "../../api/rag";
import { EmptyState } from "../../components/ui/EmptyState";
import { PageShell } from "../../components/ui/PageShell";
import { QueryBoundary, QueryErrorAlert } from "../../components/ui/QueryBoundary";
import { SectionCard } from "../../components/ui/SectionCard";
import { SettingsTabs } from "../../components/layout/SettingsTabs";
import { queryKeys } from "../../config/queryKeys";
import { getQueryErrorMessage } from "../../utils/queryErrors";

const PAGE_SIZE = 20;
const TERMINAL_JOB = new Set(["completed", "failed", "cancelled"]);

function statusColor(
    status: string
): "default" | "success" | "warning" | "error" | "info" {
    if (status === "indexed" || status === "completed") return "success";
    if (status === "failed") return "error";
    if (status === "pending" || status === "processing" || status === "uploaded") return "warning";
    return "info";
}

export default function RagView() {
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [projectId, setProjectId] = useState("");
    const [docOffset, setDocOffset] = useState(0);
    const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
    const [chunkOffset, setChunkOffset] = useState(0);
    const [pollJobId, setPollJobId] = useState<string | null>(null);
    const [retrieveQuery, setRetrieveQuery] = useState("");
    const [askQuery, setAskQuery] = useState("");
    const [retrieveResult, setRetrieveResult] = useState<RagRetrieveResult | null>(null);
    const [askResult, setAskResult] = useState<RagAskResult | null>(null);
    const [historyOffset, setHistoryOffset] = useState(0);

    const projectsQuery = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });

    const documentsQuery = useQuery({
        queryKey: queryKeys.rag.documents(projectId || undefined, docOffset),
        queryFn: () =>
            listRagDocuments({
                projectId: projectId || undefined,
                limit: PAGE_SIZE,
                offset: docOffset,
            }),
    });

    const chunksQuery = useQuery({
        queryKey: queryKeys.rag.chunks(selectedDocumentId ?? "", chunkOffset),
        queryFn: () =>
            listRagChunks(selectedDocumentId!, {
                limit: PAGE_SIZE,
                offset: chunkOffset,
                contentMode: "snippet",
            }),
        enabled: Boolean(selectedDocumentId),
    });

    const jobQuery = useQuery({
        queryKey: queryKeys.rag.job(pollJobId ?? ""),
        queryFn: () => getRagIngestionJob(pollJobId!),
        enabled: Boolean(pollJobId),
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            if (status && TERMINAL_JOB.has(status)) return false;
            return 2000;
        },
    });

    const historyQuery = useQuery({
        queryKey: queryKeys.rag.queries(historyOffset),
        queryFn: () => listRagQueries({ limit: PAGE_SIZE, offset: historyOffset }),
    });

    useEffect(() => {
        if (!jobQuery.data || !TERMINAL_JOB.has(jobQuery.data.status)) return;
        void client.invalidateQueries({ queryKey: queryKeys.rag.all });
        if (jobQuery.data.status === "completed") {
            showToast({ message: "Indexing completed.", severity: "success" });
        } else if (jobQuery.data.status === "failed") {
            showToast({
                message: jobQuery.data.error_message || "Indexing failed.",
                severity: "error",
            });
        }
        const clearPoll = window.setTimeout(() => setPollJobId(null), 0);
        return () => window.clearTimeout(clearPoll);
    }, [client, jobQuery.data, showToast]);

    const uploadMutation = useMutation({
        mutationFn: (file: File) => uploadRagDocument(file, projectId || undefined),
        onSuccess: async (result) => {
            setPollJobId(result.ingestion_job.id);
            setSelectedDocumentId(result.document.id);
            await client.invalidateQueries({ queryKey: queryKeys.rag.documents(projectId || undefined, docOffset) });
            showToast({ message: "Upload accepted; indexing queued.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Upload failed."),
                severity: "error",
            }),
    });

    const indexMutation = useMutation({
        mutationFn: (documentId: string) => indexRagDocument(documentId),
        onSuccess: (job) => {
            setPollJobId(job.id);
            showToast({ message: "Re-index queued.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Re-index failed."),
                severity: "error",
            }),
    });

    const deleteMutation = useMutation({
        mutationFn: (documentId: string) => deleteRagDocument(documentId),
        onSuccess: async () => {
            setSelectedDocumentId(null);
            await client.invalidateQueries({ queryKey: queryKeys.rag.all });
            showToast({ message: "Document deleted.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Delete failed."),
                severity: "error",
            }),
    });

    const retrieveMutation = useMutation({
        mutationFn: () =>
            retrieveRagChunks({
                query: retrieveQuery.trim(),
                project_id: projectId || undefined,
                document_ids: selectedDocumentId ? [selectedDocumentId] : undefined,
            }),
        onSuccess: (result) => setRetrieveResult(result),
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Retrieve failed."),
                severity: "error",
            }),
    });

    const askMutation = useMutation({
        mutationFn: () =>
            askRag({
                query: askQuery.trim(),
                project_id: projectId || undefined,
                document_ids: selectedDocumentId ? [selectedDocumentId] : undefined,
            }),
        onSuccess: async (result) => {
            setAskResult(result);
            await client.invalidateQueries({ queryKey: ["rag", "queries"] });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Ask failed."),
                severity: "error",
            }),
    });

    const documents = useMemo(() => documentsQuery.data?.items ?? [], [documentsQuery.data?.items]);
    const selectedDocument = useMemo(
        () => documents.find((d) => d.id === selectedDocumentId) ?? null,
        [documents, selectedDocumentId]
    );

    return (
        <PageShell maxWidth="xl">
            <SettingsTabs />

            <Stack spacing={2}>
                <SectionCard title="Project filter" description="Scope documents and retrieval to a project.">
                    <TextField
                        select
                        size="small"
                        label="Project"
                        value={projectId}
                        onChange={(e) => {
                            setProjectId(e.target.value);
                            setDocOffset(0);
                            setSelectedDocumentId(null);
                        }}
                        sx={{ minWidth: 280 }}
                    >
                        <MenuItem value="">All projects</MenuItem>
                        {(projectsQuery.data ?? []).map((project) => (
                            <MenuItem key={project.id} value={project.id}>
                                {project.name}
                            </MenuItem>
                        ))}
                    </TextField>
                </SectionCard>

                <SectionCard
                    title="Documents"
                    description="Upload, monitor indexing status, retry failed jobs."
                    action={
                        <>
                            <input
                                ref={fileInputRef}
                                type="file"
                                hidden
                                onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) uploadMutation.mutate(file);
                                    e.target.value = "";
                                }}
                            />
                            <Button
                                size="small"
                                variant="contained"
                                startIcon={<UploadIcon />}
                                disabled={uploadMutation.isPending}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                Upload
                            </Button>
                        </>
                    }
                >
                    {documentsQuery.isError ? (
                        <QueryErrorAlert
                            error={documentsQuery.error}
                            fallback="Failed to load RAG documents. RAG may be disabled."
                            onRetry={() => void documentsQuery.refetch()}
                        />
                    ) : (
                        <QueryBoundary
                            isLoading={documentsQuery.isLoading}
                            isError={false}
                        >
                            {documents.length === 0 ? (
                                <EmptyState
                                    icon={<UploadIcon />}
                                    title="No RAG documents"
                                    description="Upload a file to start ingestion and indexing."
                                />
                            ) : (
                                <>
                                    <Box sx={{ overflowX: "auto" }}>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Filename</TableCell>
                                                    <TableCell>Status</TableCell>
                                                    <TableCell>Updated</TableCell>
                                                    <TableCell align="right">Actions</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {documents.map((doc) => (
                                                    <TableRow
                                                        key={doc.id}
                                                        selected={selectedDocumentId === doc.id}
                                                        hover
                                                        onClick={() => {
                                                            setSelectedDocumentId(doc.id);
                                                            setChunkOffset(0);
                                                        }}
                                                        sx={{ cursor: "pointer" }}
                                                    >
                                                        <TableCell>
                                                            {doc.original_filename || doc.filename}
                                                        </TableCell>
                                                        <TableCell>
                                                            <Chip
                                                                size="small"
                                                                color={statusColor(doc.status)}
                                                                label={doc.status}
                                                            />
                                                        </TableCell>
                                                        <TableCell>
                                                            {new Date(doc.updated_at).toLocaleString()}
                                                        </TableCell>
                                                        <TableCell align="right">
                                                            <Button
                                                                size="small"
                                                                startIcon={<RetryIcon />}
                                                                disabled={indexMutation.isPending}
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    indexMutation.mutate(doc.id);
                                                                }}
                                                            >
                                                                Re-index
                                                            </Button>
                                                            <Button
                                                                size="small"
                                                                color="error"
                                                                disabled={deleteMutation.isPending}
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    if (
                                                                        window.confirm(
                                                                            "Delete this RAG document?"
                                                                        )
                                                                    ) {
                                                                        deleteMutation.mutate(doc.id);
                                                                    }
                                                                }}
                                                            >
                                                                Delete
                                                            </Button>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </Box>
                                    <TablePagination
                                        component="div"
                                        count={documentsQuery.data?.total ?? 0}
                                        page={Math.floor(docOffset / PAGE_SIZE)}
                                        onPageChange={(_, page) => setDocOffset(page * PAGE_SIZE)}
                                        rowsPerPage={PAGE_SIZE}
                                        rowsPerPageOptions={[PAGE_SIZE]}
                                    />
                                </>
                            )}
                        </QueryBoundary>
                    )}
                    {pollJobId && jobQuery.data ? (
                        <Alert
                            severity={
                                jobQuery.data.status === "failed"
                                    ? "error"
                                    : jobQuery.data.status === "completed"
                                      ? "success"
                                      : "info"
                            }
                            sx={{ mt: 1 }}
                        >
                            Ingestion job {jobQuery.data.status}
                            {jobQuery.data.error_message
                                ? `: ${jobQuery.data.error_message}`
                                : ""}
                        </Alert>
                    ) : null}
                </SectionCard>

                {selectedDocument ? (
                    <SectionCard
                        title="Chunks"
                        description={`Snippet browser for ${selectedDocument.original_filename || selectedDocument.filename}`}
                    >
                        <QueryBoundary
                            isLoading={chunksQuery.isLoading}
                            isError={chunksQuery.isError}
                            error={chunksQuery.error}
                            onRetry={() => void chunksQuery.refetch()}
                        >
                            {(chunksQuery.data?.items ?? []).map((chunk) => (
                                <Box
                                    key={chunk.id}
                                    sx={{ py: 1, borderBottom: 1, borderColor: "divider" }}
                                >
                                    <Typography variant="caption" color="text.secondary">
                                        #{chunk.chunk_index} · {chunk.token_count} tokens
                                    </Typography>
                                    <Typography variant="body2">{chunk.content}</Typography>
                                </Box>
                            ))}
                            <TablePagination
                                component="div"
                                count={chunksQuery.data?.total ?? 0}
                                page={Math.floor(chunkOffset / PAGE_SIZE)}
                                onPageChange={(_, page) => setChunkOffset(page * PAGE_SIZE)}
                                rowsPerPage={PAGE_SIZE}
                                rowsPerPageOptions={[PAGE_SIZE]}
                            />
                        </QueryBoundary>
                    </SectionCard>
                ) : null}

                <SectionCard title="Retrieval explorer" description="Run a retrieve-only query.">
                    <Stack spacing={1.5}>
                        <TextField
                            size="small"
                            label="Query"
                            value={retrieveQuery}
                            onChange={(e) => setRetrieveQuery(e.target.value)}
                            fullWidth
                        />
                        <Button
                            variant="contained"
                            startIcon={<RetrieveIcon />}
                            disabled={!retrieveQuery.trim() || retrieveMutation.isPending}
                            onClick={() => retrieveMutation.mutate()}
                        >
                            Retrieve
                        </Button>
                        {retrieveResult ? (
                            <Stack spacing={1}>
                                {retrieveResult.degraded || retrieveResult.no_matches ? (
                                    <Alert severity="warning">
                                        {retrieveResult.degraded
                                            ? `Degraded: ${retrieveResult.degradation_reason ?? "unknown"}`
                                            : "No matches"}
                                        {retrieveResult.injection_chunks_filtered
                                            ? ` · filtered ${retrieveResult.injection_chunks_filtered} injection chunk(s)`
                                            : ""}
                                    </Alert>
                                ) : null}
                                {retrieveResult.chunks.map((chunk) => (
                                    <Box key={chunk.chunk_id} sx={{ p: 1.5, bgcolor: "action.hover", borderRadius: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                            {chunk.filename} · score {chunk.score.toFixed(3)}
                                        </Typography>
                                        <Typography variant="body2">{chunk.content}</Typography>
                                    </Box>
                                ))}
                            </Stack>
                        ) : null}
                    </Stack>
                </SectionCard>

                <SectionCard title="Ask" description="Grounded answer with citations.">
                    <Stack spacing={1.5}>
                        <TextField
                            size="small"
                            label="Question"
                            value={askQuery}
                            onChange={(e) => setAskQuery(e.target.value)}
                            fullWidth
                            multiline
                            minRows={2}
                        />
                        <Button
                            variant="contained"
                            startIcon={<AskIcon />}
                            disabled={!askQuery.trim() || askMutation.isPending}
                            onClick={() => askMutation.mutate()}
                        >
                            Ask
                        </Button>
                        {askResult ? (
                            <Stack spacing={1}>
                                {(askResult.retrieval_degraded ||
                                    askResult.memory_degraded ||
                                    askResult.no_context_found) && (
                                    <Alert severity="warning">
                                        {askResult.no_context_found
                                            ? "No context found. "
                                            : ""}
                                        {askResult.degradation_reason ??
                                            (askResult.retrieval_degraded
                                                ? "Retrieval degraded."
                                                : askResult.memory_degraded
                                                  ? "Memory degraded."
                                                  : "")}
                                    </Alert>
                                )}
                                <Typography variant="body1">{askResult.answer}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {askResult.model_name} · {askResult.latency_ms} ms
                                </Typography>
                                <Typography variant="subtitle2">Citations</Typography>
                                {askResult.citations.map((citation) => (
                                    <Box
                                        key={`${citation.document_id}:${citation.chunk_id}`}
                                        sx={{ p: 1, border: 1, borderColor: "divider", borderRadius: 1 }}
                                    >
                                        <Typography variant="caption" color="text.secondary">
                                            {citation.filename} · score {citation.score.toFixed(3)}
                                            {citation.page_number != null
                                                ? ` · p.${citation.page_number}`
                                                : ""}
                                        </Typography>
                                        <Typography variant="body2">{citation.snippet}</Typography>
                                    </Box>
                                ))}
                            </Stack>
                        ) : null}
                    </Stack>
                </SectionCard>

                <SectionCard title="Query history" description="Recent RAG asks for this user.">
                    <QueryBoundary
                        isLoading={historyQuery.isLoading}
                        isError={historyQuery.isError}
                        error={historyQuery.error}
                        onRetry={() => void historyQuery.refetch()}
                    >
                        <Box sx={{ overflowX: "auto" }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Query</TableCell>
                                        <TableCell>Model</TableCell>
                                        <TableCell>Latency</TableCell>
                                        <TableCell>When</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {(historyQuery.data?.items ?? []).map((item) => (
                                        <TableRow key={item.id}>
                                            <TableCell>{item.query}</TableCell>
                                            <TableCell>{item.model_name}</TableCell>
                                            <TableCell>{item.latency_ms} ms</TableCell>
                                            <TableCell>
                                                {new Date(item.created_at).toLocaleString()}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                        <TablePagination
                            component="div"
                            count={historyQuery.data?.total ?? 0}
                            page={Math.floor(historyOffset / PAGE_SIZE)}
                            onPageChange={(_, page) => setHistoryOffset(page * PAGE_SIZE)}
                            rowsPerPage={PAGE_SIZE}
                            rowsPerPageOptions={[PAGE_SIZE]}
                        />
                    </QueryBoundary>
                </SectionCard>
            </Stack>
        </PageShell>
    );
}
