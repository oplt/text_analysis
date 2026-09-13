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
import { HelpFieldLabel, HelpTooltip } from "../../components/ui/HelpTooltip";
import { KeyValueList } from "../../components/ui/KeyValueList";
import { MetricGrid } from "../../components/ui/MetricGrid";
import { PageHeader } from "../../components/ui/PageHeader";
import { PageShell } from "../../components/ui/PageShell";
import { PageTabs } from "../../components/ui/PageTabs";
import { QueryBoundary, QueryErrorAlert } from "../../components/ui/QueryBoundary";
import { DataTable } from "../../components/ui/DataTable";
import { RunStatusChip } from "../../components/ui/RunStatusChip";
import { RunStatusPanel } from "../../components/ui/RunStatusPanel";
import { SectionCard } from "../../components/ui/SectionCard";
import { SettingsTabs } from "../../components/layout/SettingsTabs";
import { queryKeys } from "../../config/queryKeys";
import { useTabQueryParam } from "../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../utils/queryErrors";
import {
    documentPipelineLabel,
    formatCoverage,
    indexingParamsFromChunkMeta,
    summarizeRagAsk,
} from "./ragEvaluationSummary";

const PAGE_SIZE = 20;
const TERMINAL_JOB = new Set(["completed", "failed", "cancelled"]);
const RAG_TABS = ["documents", "indexing", "retrieval", "evaluation", "runs"] as const;

function MetricTile({
    label,
    value,
    helpTermId,
    description,
}: {
    label: string;
    value: string | number;
    helpTermId?: string;
    description?: string;
}) {
    return (
        <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: "action.hover", height: "100%" }}>
            <Typography variant="caption" color="text.secondary" component="div">
                {helpTermId ? (
                    <HelpTooltip termId={helpTermId} variant="label">
                        {label}
                    </HelpTooltip>
                ) : (
                    label
                )}
            </Typography>
            <Typography variant="h6">{value}</Typography>
            {description ? (
                <Typography variant="caption" color="text.secondary">
                    {description}
                </Typography>
            ) : null}
        </Box>
    );
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
    const [topK, setTopK] = useState(5);
    const [askQuery, setAskQuery] = useState("");
    const [retrieveResult, setRetrieveResult] = useState<RagRetrieveResult | null>(null);
    const [askResult, setAskResult] = useState<RagAskResult | null>(null);
    const [historyOffset, setHistoryOffset] = useState(0);
    const [tab, setTab] = useTabQueryParam(RAG_TABS, "documents");

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
            await client.invalidateQueries({
                queryKey: queryKeys.rag.documents(projectId || undefined, docOffset),
            });
            showToast({ message: "Upload accepted; indexing queued.", severity: "success" });
            setTab("indexing");
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
            setTab("indexing");
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
                top_k: topK,
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
    const indexingParams = useMemo(
        () => indexingParamsFromChunkMeta(chunksQuery.data?.items?.[0]?.metadata),
        [chunksQuery.data?.items]
    );
    const askEval = useMemo(
        () => (askResult ? summarizeRagAsk(askResult) : null),
        [askResult]
    );
    const projectNameById = useMemo(() => {
        const map = new Map<string, string>();
        for (const project of projectsQuery.data ?? []) {
            map.set(project.id, project.name);
        }
        return map;
    }, [projectsQuery.data]);

    return (
        <PageShell maxWidth="xl">
            <PageHeader
                title="RAG workspace"
                description="Follow the retrieval pipeline: Documents → Indexing → Retrieval → Evaluation → Runs. Researchers can also use Ask Corpus inside Text Research."
            />
            <SettingsTabs />

            <Stack spacing={2} sx={{ mt: 2 }}>
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1.5}
                    alignItems={{ sm: "center" }}
                    justifyContent="space-between"
                >
                    <Typography variant="body2" color="text.secondary">
                        Optionally limit documents and retrieval to one project (index / corpus scope).
                    </Typography>
                    <TextField
                        select
                        size="small"
                        label="Project / index scope"
                        value={projectId}
                        onChange={(e) => {
                            setProjectId(e.target.value);
                            setDocOffset(0);
                            setSelectedDocumentId(null);
                        }}
                        sx={{ minWidth: { xs: "100%", sm: 280 } }}
                    >
                        <MenuItem value="">All projects</MenuItem>
                        {(projectsQuery.data ?? []).map((project) => (
                            <MenuItem key={project.id} value={project.id}>
                                {project.name}
                            </MenuItem>
                        ))}
                    </TextField>
                </Stack>

                <PageTabs
                    value={tab}
                    onChange={setTab}
                    tabs={[
                        { value: "documents", label: "Documents" },
                        { value: "indexing", label: "Indexing" },
                        { value: "retrieval", label: "Retrieval" },
                        { value: "evaluation", label: "Evaluation" },
                        { value: "runs", label: "Runs" },
                    ]}
                    ariaLabel="RAG workspace"
                />

                {tab === "documents" ? (
                    <SectionCard
                        title="Documents"
                        description="Upload sources and monitor parsing + indexing state before retrieval."
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
                            <QueryBoundary isLoading={documentsQuery.isLoading} isError={false}>
                                <DataTable
                                    ariaLabel="RAG documents"
                                    columns={[
                                        {
                                            id: "document",
                                            label: "Document",
                                            sticky: "left",
                                            truncate: 48,
                                            sortable: true,
                                            getSortValue: (doc) =>
                                                doc.original_filename || doc.filename,
                                            render: (doc) =>
                                                doc.original_filename || doc.filename,
                                        },
                                        {
                                            id: "parsing",
                                            label: (
                                                <HelpTooltip
                                                    termId="parsing_state"
                                                    variant="label"
                                                >
                                                    Parsing
                                                </HelpTooltip>
                                            ),
                                            render: (doc) =>
                                                documentPipelineLabel(doc.status).parsing,
                                        },
                                        {
                                            id: "indexing",
                                            label: (
                                                <HelpTooltip
                                                    termId="indexing_state"
                                                    variant="label"
                                                >
                                                    Indexing
                                                </HelpTooltip>
                                            ),
                                            render: (doc) =>
                                                documentPipelineLabel(doc.status).indexing,
                                        },
                                        {
                                            id: "status",
                                            label: "Status",
                                            sortable: true,
                                            getSortValue: (doc) => doc.status,
                                            render: (doc) => (
                                                <RunStatusChip status={doc.status} showRaw />
                                            ),
                                        },
                                        {
                                            id: "updated",
                                            label: "Updated",
                                            sortable: true,
                                            hideable: true,
                                            getSortValue: (doc) => Date.parse(doc.updated_at),
                                            render: (doc) =>
                                                new Date(doc.updated_at).toLocaleString(),
                                        },
                                        {
                                            id: "actions",
                                            label: "Actions",
                                            align: "right",
                                            sticky: "right",
                                            hideable: false,
                                            render: (doc) => (
                                                <Stack
                                                    direction="row"
                                                    spacing={0.5}
                                                    justifyContent="flex-end"
                                                    onClick={(e) => e.stopPropagation()}
                                                >
                                                    <Button
                                                        size="small"
                                                        startIcon={<RetryIcon />}
                                                        disabled={indexMutation.isPending}
                                                        onClick={() =>
                                                            indexMutation.mutate(doc.id)
                                                        }
                                                    >
                                                        Re-index
                                                    </Button>
                                                    <Button
                                                        size="small"
                                                        color="error"
                                                        disabled={deleteMutation.isPending}
                                                        onClick={() => {
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
                                                </Stack>
                                            ),
                                        },
                                    ]}
                                    rows={documents}
                                    getRowId={(doc) => doc.id}
                                    density="compact"
                                    showDensityToggle
                                    showColumnVisibility
                                    stickyHeader
                                    stickyFirstColumn
                                    clientSort
                                    page={Math.floor(docOffset / PAGE_SIZE)}
                                    pageSize={PAGE_SIZE}
                                    totalCount={documentsQuery.data?.total ?? 0}
                                    onPageChange={(page) => setDocOffset(page * PAGE_SIZE)}
                                    selectedRowId={selectedDocumentId}
                                    onRowClick={(doc) => {
                                        setSelectedDocumentId(doc.id);
                                        setChunkOffset(0);
                                        setTab("indexing");
                                    }}
                                    emptyIcon={<UploadIcon />}
                                    emptyTitle="No RAG documents"
                                    emptyDescription="Upload a file to start parsing, chunking, and indexing."
                                />
                            </QueryBoundary>
                        )}
                        {pollJobId && jobQuery.data ? (
                            <Box sx={{ mt: 1 }}>
                                <RunStatusPanel
                                    dense
                                    title="Ingestion job"
                                    status={jobQuery.data.status}
                                    runId={jobQuery.data.id}
                                    stage="parse → chunk → embed"
                                    startedAt={jobQuery.data.started_at}
                                    completedAt={jobQuery.data.finished_at}
                                    createdAt={jobQuery.data.created_at}
                                    errorMessage={jobQuery.data.error_message}
                                    onRetry={
                                        selectedDocumentId
                                            ? () => indexMutation.mutate(selectedDocumentId)
                                            : undefined
                                    }
                                    retryLabel="Re-index"
                                    retryDisabled={indexMutation.isPending}
                                />
                            </Box>
                        ) : null}
                    </SectionCard>
                ) : null}

                {tab === "indexing" ? (
                    <>
                        {!selectedDocument ? (
                            <SectionCard
                                title="Indexing"
                                description="Select a document to inspect embedding/chunk settings, progress, and chunks."
                            >
                                <EmptyState
                                    icon={<RetryIcon />}
                                    title="No document selected"
                                    description="Choose a document from Documents, then return here to browse chunks and indexing progress."
                                    action={
                                        <Button
                                            variant="contained"
                                            onClick={() => setTab("documents")}
                                        >
                                            Open Documents
                                        </Button>
                                    }
                                />
                            </SectionCard>
                        ) : (
                            <>
                                <SectionCard
                                    title="Indexing configuration"
                                    description={`Settings observed for ${
                                        selectedDocument.original_filename ||
                                        selectedDocument.filename
                                    }`}
                                    action={
                                        <Button
                                            size="small"
                                            startIcon={<RetryIcon />}
                                            disabled={indexMutation.isPending}
                                            onClick={() =>
                                                indexMutation.mutate(selectedDocument.id)
                                            }
                                        >
                                            Re-index
                                        </Button>
                                    }
                                >
                                    <Stack spacing={2}>
                                        <MetricGrid columns={4}>
                                            <MetricTile
                                                label="Document status"
                                                value={selectedDocument.status}
                                                helpTermId="indexing_state"
                                            />
                                            <MetricTile
                                                label="Chunks"
                                                value={chunksQuery.data?.total ?? "—"}
                                                helpTermId="chunk_size"
                                                description="Indexed segments"
                                            />
                                            <MetricTile
                                                label="Parsing"
                                                value={
                                                    documentPipelineLabel(selectedDocument.status)
                                                        .parsing
                                                }
                                                helpTermId="parsing_state"
                                            />
                                            <MetricTile
                                                label="Job"
                                                value={jobQuery.data?.status ?? "idle"}
                                                description={
                                                    pollJobId ? "Live ingestion poll" : "No active job"
                                                }
                                            />
                                        </MetricGrid>

                                        {pollJobId && jobQuery.data ? (
                                            <RunStatusPanel
                                                title="Indexing progress"
                                                status={jobQuery.data.status}
                                                runId={jobQuery.data.id}
                                                stage="parse → chunk → embed"
                                                startedAt={jobQuery.data.started_at}
                                                completedAt={jobQuery.data.finished_at}
                                                createdAt={jobQuery.data.created_at}
                                                errorMessage={jobQuery.data.error_message}
                                                onRetry={() =>
                                                    indexMutation.mutate(selectedDocument.id)
                                                }
                                                retryLabel="Re-index"
                                                retryDisabled={indexMutation.isPending}
                                            />
                                        ) : null}

                                        {indexingParams.length ? (
                                            <KeyValueList
                                                showHelp
                                                items={indexingParams.map((row) => ({
                                                    key: row.key,
                                                    label: row.label,
                                                    value: row.value,
                                                    helpTermId: row.helpTermId,
                                                }))}
                                            />
                                        ) : (
                                            <Alert severity="info" icon={false}>
                                                Embedding model,{" "}
                                                <HelpTooltip termId="chunk_size" variant="label">
                                                    chunk size
                                                </HelpTooltip>
                                                , and{" "}
                                                <HelpTooltip termId="chunk_overlap" variant="label">
                                                    overlap
                                                </HelpTooltip>{" "}
                                                appear here once chunks are available (from chunk
                                                provenance metadata).
                                            </Alert>
                                        )}
                                    </Stack>
                                </SectionCard>

                                <SectionCard
                                    title="Chunks"
                                    description="Browse indexed segments for this document."
                                >
                                    <QueryBoundary
                                        isLoading={chunksQuery.isLoading}
                                        isError={chunksQuery.isError}
                                        error={chunksQuery.error}
                                        onRetry={() => void chunksQuery.refetch()}
                                    >
                                        {(chunksQuery.data?.items ?? []).length === 0 ? (
                                            <Typography variant="body2" color="text.secondary">
                                                No chunks yet — wait for indexing or re-index.
                                            </Typography>
                                        ) : (
                                            (chunksQuery.data?.items ?? []).map((chunk) => (
                                                <Box
                                                    key={chunk.id}
                                                    sx={{
                                                        py: 1,
                                                        borderBottom: 1,
                                                        borderColor: "divider",
                                                    }}
                                                >
                                                    <Typography
                                                        variant="caption"
                                                        color="text.secondary"
                                                    >
                                                        #{chunk.chunk_index} · {chunk.token_count}{" "}
                                                        tokens
                                                    </Typography>
                                                    <Typography variant="body2">
                                                        {chunk.content}
                                                    </Typography>
                                                </Box>
                                            ))
                                        )}
                                        <TablePagination
                                            component="div"
                                            count={chunksQuery.data?.total ?? 0}
                                            page={Math.floor(chunkOffset / PAGE_SIZE)}
                                            onPageChange={(_, page) =>
                                                setChunkOffset(page * PAGE_SIZE)
                                            }
                                            rowsPerPage={PAGE_SIZE}
                                            rowsPerPageOptions={[PAGE_SIZE]}
                                        />
                                    </QueryBoundary>
                                </SectionCard>
                            </>
                        )}
                    </>
                ) : null}

                {tab === "retrieval" ? (
                    <SectionCard
                        title="Retrieval"
                        description="Run a retrieve-only query and inspect ranked chunks with similarity scores and sources."
                    >
                        <Stack spacing={1.5}>
                            <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={2}
                                alignItems={{ sm: "flex-start" }}
                            >
                                <TextField
                                    size="small"
                                    label="Query"
                                    value={retrieveQuery}
                                    onChange={(e) => setRetrieveQuery(e.target.value)}
                                    fullWidth
                                />
                                <TextField
                                    size="small"
                                    type="number"
                                    label={
                                        <HelpFieldLabel termId="top_k_retrieval">
                                            Top-k
                                        </HelpFieldLabel>
                                    }
                                    value={topK}
                                    onChange={(e) =>
                                        setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 1)))
                                    }
                                    inputProps={{ min: 1, max: 50, step: 1 }}
                                    sx={{ width: 120 }}
                                />
                            </Stack>
                            {selectedDocumentId ? (
                                <Typography variant="caption" color="text.secondary">
                                    Scoped to selected document. Clear selection on Documents to
                                    search the full project scope.
                                </Typography>
                            ) : null}
                            <Button
                                variant="contained"
                                startIcon={<RetrieveIcon />}
                                disabled={!retrieveQuery.trim() || retrieveMutation.isPending}
                                onClick={() => retrieveMutation.mutate()}
                                sx={{ alignSelf: "flex-start" }}
                            >
                                Retrieve
                            </Button>
                            {retrieveResult ? (
                                <Stack spacing={1}>
                                    {retrieveResult.degraded || retrieveResult.no_matches ? (
                                        <Alert severity="warning">
                                            {retrieveResult.degraded
                                                ? `Degraded: ${
                                                      retrieveResult.degradation_reason ?? "unknown"
                                                  }`
                                                : "No matches"}
                                            {retrieveResult.injection_chunks_filtered
                                                ? ` · filtered ${retrieveResult.injection_chunks_filtered} injection chunk(s)`
                                                : ""}
                                        </Alert>
                                    ) : null}
                                    <Typography variant="subtitle2">
                                        Retrieved chunks ({retrieveResult.chunks.length})
                                    </Typography>
                                    <Box sx={{ overflowX: "auto" }}>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Rank</TableCell>
                                                    <TableCell>Document source</TableCell>
                                                    <TableCell>Page</TableCell>
                                                    <TableCell>
                                                        <HelpTooltip
                                                            termId="similarity_threshold"
                                                            variant="label"
                                                        >
                                                            Similarity
                                                        </HelpTooltip>
                                                    </TableCell>
                                                    <TableCell>Chunk</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {retrieveResult.chunks.map((chunk, index) => (
                                                    <TableRow key={chunk.chunk_id}>
                                                        <TableCell>{index + 1}</TableCell>
                                                        <TableCell>{chunk.filename}</TableCell>
                                                        <TableCell>
                                                            {chunk.page_number ?? "—"}
                                                        </TableCell>
                                                        <TableCell>
                                                            {chunk.score.toFixed(3)}
                                                        </TableCell>
                                                        <TableCell sx={{ maxWidth: 420 }}>
                                                            <Typography variant="body2">
                                                                {chunk.content.slice(0, 280)}
                                                                {chunk.content.length > 280
                                                                    ? "…"
                                                                    : ""}
                                                            </Typography>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </Box>
                                </Stack>
                            ) : null}
                        </Stack>
                    </SectionCard>
                ) : null}

                {tab === "evaluation" ? (
                    <SectionCard
                        title="Evaluation"
                        description="Grounded ask with citation coverage and retrieval-relevance signals for this response."
                    >
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
                                sx={{ alignSelf: "flex-start" }}
                            >
                                Ask (grounded)
                            </Button>
                            {askResult && askEval ? (
                                <Stack spacing={1.5}>
                                    <MetricGrid columns={4}>
                                        <MetricTile
                                            label="Retrieved chunks"
                                            value={askEval.retrievedCount}
                                            helpTermId="top_k_retrieval"
                                        />
                                        <MetricTile
                                            label="Citations"
                                            value={askEval.citationCount}
                                            helpTermId="grounding"
                                        />
                                        <MetricTile
                                            label="Citation coverage"
                                            value={formatCoverage(askEval.citationCoverage)}
                                            helpTermId="citation_coverage"
                                            description="Cited ÷ retrieved"
                                        />
                                        <MetricTile
                                            label="Documents cited"
                                            value={askEval.distinctDocuments}
                                            description={
                                                askEval.meanCitationScore != null
                                                    ? `Mean score ${askEval.meanCitationScore.toFixed(3)}`
                                                    : undefined
                                            }
                                        />
                                    </MetricGrid>

                                    {askEval.groundingNotes.map((note) => (
                                        <Alert key={note} severity="warning">
                                            {note}
                                        </Alert>
                                    ))}

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

                                    <Typography variant="subtitle2">Answer</Typography>
                                    <Typography variant="body1">{askResult.answer}</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {askResult.model_name} · {askResult.latency_ms} ms
                                    </Typography>

                                    <Typography variant="subtitle2">
                                        <HelpTooltip termId="grounding" variant="label">
                                            Citations (evidence)
                                        </HelpTooltip>
                                    </Typography>
                                    {askResult.citations.map((citation) => (
                                        <Box
                                            key={`${citation.document_id}:${citation.chunk_id}`}
                                            sx={{
                                                p: 1,
                                                border: 1,
                                                borderColor: "divider",
                                                borderRadius: 1,
                                            }}
                                        >
                                            <Typography variant="caption" color="text.secondary">
                                                {citation.filename} · score{" "}
                                                {citation.score.toFixed(3)}
                                                {citation.page_number != null
                                                    ? ` · p.${citation.page_number}`
                                                    : ""}
                                            </Typography>
                                            <Typography variant="body2">
                                                {citation.snippet}
                                            </Typography>
                                        </Box>
                                    ))}
                                </Stack>
                            ) : null}
                        </Stack>
                    </SectionCard>
                ) : null}

                {tab === "runs" ? (
                    <SectionCard
                        title="Runs"
                        description="Recent RAG asks with timestamp, model, scope, status, and duration."
                    >
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
                                            <TableCell>Timestamp</TableCell>
                                            <TableCell>Query</TableCell>
                                            <TableCell>Model</TableCell>
                                            <TableCell>Project / index</TableCell>
                                            <TableCell>Status</TableCell>
                                            <TableCell>Duration</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {(historyQuery.data?.items ?? []).map((item) => (
                                            <TableRow key={item.id}>
                                                <TableCell>
                                                    {new Date(item.created_at).toLocaleString()}
                                                </TableCell>
                                                <TableCell sx={{ maxWidth: 320 }}>
                                                    {item.query}
                                                </TableCell>
                                                <TableCell>{item.model_name}</TableCell>
                                                <TableCell>
                                                    {item.project_id
                                                        ? projectNameById.get(item.project_id) ??
                                                          item.project_id.slice(0, 8)
                                                        : "All / default"}
                                                </TableCell>
                                                <TableCell>
                                                    <Chip
                                                        size="small"
                                                        color={
                                                            item.answer?.trim()
                                                                ? "success"
                                                                : "warning"
                                                        }
                                                        label={
                                                            item.answer?.trim()
                                                                ? "completed"
                                                                : "empty answer"
                                                        }
                                                    />
                                                </TableCell>
                                                <TableCell>{item.latency_ms} ms</TableCell>
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
                ) : null}
            </Stack>
        </PageShell>
    );
}
