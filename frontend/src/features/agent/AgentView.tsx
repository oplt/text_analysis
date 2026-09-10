import { useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import { SmartToy as AgentIcon, PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../app/snackbarContext";
import { createAgentRun, getAgentRun, listAgentRuns } from "../../api/agent";
import { getAiOverview } from "../../api/ai";
import { listProjects } from "../../api/projects";
import { listRagDocuments } from "../../api/rag";
import { EmptyState } from "../../components/ui/EmptyState";
import { PageHeader } from "../../components/ui/PageHeader";
import { PageShell } from "../../components/ui/PageShell";
import { QueryBoundary } from "../../components/ui/QueryBoundary";
import { SectionCard } from "../../components/ui/SectionCard";
import { SettingsTabs } from "../../components/layout/SettingsTabs";
import { queryKeys } from "../../config/queryKeys";
import { getQueryErrorMessage } from "../../utils/queryErrors";
import { AgentRunDetail } from "./AgentRunDetail";
import { invalidateAgentRunQueries } from "./agentQueryUtils";

export default function AgentView() {
    const { showToast } = useSnackbar();
    const queryClient = useQueryClient();

    const [agentId, setAgentId] = useState("default");
    const [projectId, setProjectId] = useState("");
    const [userMessage, setUserMessage] = useState("");
    const [promptTemplateKey, setPromptTemplateKey] = useState("");
    const [retrievalQuery, setRetrievalQuery] = useState("");
    const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
    const [topK, setTopK] = useState(4);
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

    const overviewQuery = useQuery({
        queryKey: queryKeys.ai.overview,
        queryFn: getAiOverview,
    });

    const projectsQuery = useQuery({
        queryKey: queryKeys.projects.all,
        queryFn: listProjects,
    });

    const documentsQuery = useQuery({
        queryKey: queryKeys.rag.documents(projectId || undefined, 0),
        queryFn: () =>
            listRagDocuments({
                projectId: projectId || undefined,
                limit: 50,
                offset: 0,
            }),
    });

    const runsQuery = useQuery({
        queryKey: queryKeys.agent.runs,
        queryFn: () => listAgentRuns({ limit: 20, offset: 0 }),
    });

    const selectedRunQuery = useQuery({
        queryKey: queryKeys.agent.run(selectedRunId ?? ""),
        queryFn: () => getAgentRun(selectedRunId!),
        enabled: selectedRunId !== null,
    });

    const templates = overviewQuery.data?.prompt_templates ?? [];

    const runMutation = useMutation({
        mutationFn: () => {
            if (!userMessage.trim() && !promptTemplateKey) {
                throw new Error("Provide a user message or prompt template key.");
            }
            return createAgentRun({
                agent_id: agentId.trim() || "default",
                project_id: projectId || undefined,
                user_message: userMessage.trim() || undefined,
                prompt_template_key: promptTemplateKey || undefined,
                retrieval_query: retrievalQuery.trim() || userMessage.trim() || undefined,
                document_ids: selectedDocumentIds,
                top_k: topK,
            });
        },
        onSuccess: async (run) => {
            setSelectedRunId(run.id);
            await invalidateAgentRunQueries(queryClient, run.id);
            showToast({ message: "Agent run completed.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Agent run failed."),
                severity: "error",
            }),
    });

    const history = runsQuery.data?.items ?? [];
    const selectedRun = selectedRunQuery.data ?? null;

    function toggleDocument(id: string) {
        setSelectedDocumentIds((ids) =>
            ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
        );
    }

    return (
        <PageShell maxWidth="xl">
            <SettingsTabs />
            <PageHeader
                title="Agent"
                description="Run an agent prompt with optional RAG retrieval and memory. Long runs use the sync API with loading/error handling."
            />

            <Stack spacing={2} sx={{ mt: 2 }}>
                <SectionCard
                    title="New agent run"
                    description="Configure agent ID, project, message, retrieval options, and selected RAG documents."
                >
                    <Stack spacing={2}>
                        <Alert severity="info">
                            Execution is synchronous today (HTTP 201). The request shows a
                            loading state until the backend responds or times out.
                        </Alert>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} flexWrap="wrap" useFlexGap>
                            <TextField
                                size="small"
                                label="Agent ID"
                                value={agentId}
                                onChange={(e) => setAgentId(e.target.value)}
                                sx={{ minWidth: 160 }}
                            />
                            <TextField
                                select
                                size="small"
                                label="Project"
                                value={projectId}
                                onChange={(e) => {
                                    setProjectId(e.target.value);
                                    setSelectedDocumentIds([]);
                                }}
                                sx={{ minWidth: 220 }}
                            >
                                <MenuItem value="">None</MenuItem>
                                {(projectsQuery.data ?? []).map((project) => (
                                    <MenuItem key={project.id} value={project.id}>
                                        {project.name}
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                select
                                size="small"
                                label="Prompt template"
                                value={promptTemplateKey}
                                onChange={(e) => setPromptTemplateKey(e.target.value)}
                                sx={{ minWidth: 220 }}
                            >
                                <MenuItem value="">None (message only)</MenuItem>
                                {templates.map((template) => (
                                    <MenuItem key={template.id} value={template.key}>
                                        {template.name} ({template.key})
                                    </MenuItem>
                                ))}
                            </TextField>
                            <TextField
                                size="small"
                                type="number"
                                label="Top K"
                                value={topK}
                                onChange={(e) => setTopK(Number(e.target.value) || 4)}
                                inputProps={{ min: 1, max: 20 }}
                                sx={{ width: 100 }}
                            />
                        </Stack>
                        <TextField
                            size="small"
                            label="User message"
                            value={userMessage}
                            onChange={(e) => setUserMessage(e.target.value)}
                            multiline
                            minRows={3}
                            fullWidth
                        />
                        <TextField
                            size="small"
                            label="Retrieval query (optional)"
                            value={retrievalQuery}
                            onChange={(e) => setRetrievalQuery(e.target.value)}
                            fullWidth
                            helperText="Defaults to the user message when empty."
                        />
                        <Box>
                            <Typography variant="subtitle2" gutterBottom>
                                RAG documents (optional)
                            </Typography>
                            <QueryBoundary
                                isLoading={documentsQuery.isLoading}
                                isError={documentsQuery.isError}
                                error={documentsQuery.error}
                                onRetry={() => void documentsQuery.refetch()}
                            >
                                {(documentsQuery.data?.items ?? []).length === 0 ? (
                                    <Typography variant="body2" color="text.secondary">
                                        No RAG documents for this filter.
                                    </Typography>
                                ) : (
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        {(documentsQuery.data?.items ?? []).map((doc) => {
                                            const selected = selectedDocumentIds.includes(doc.id);
                                            return (
                                                <Chip
                                                    key={doc.id}
                                                    label={doc.original_filename || doc.filename}
                                                    color={selected ? "primary" : "default"}
                                                    variant={selected ? "filled" : "outlined"}
                                                    onClick={() => toggleDocument(doc.id)}
                                                />
                                            );
                                        })}
                                    </Stack>
                                )}
                            </QueryBoundary>
                        </Box>
                        <Button
                            variant="contained"
                            startIcon={<RunIcon />}
                            disabled={runMutation.isPending}
                            onClick={() => runMutation.mutate()}
                        >
                            {runMutation.isPending ? "Running…" : "Run agent"}
                        </Button>
                        {runMutation.isError ? (
                            <Alert severity="error">
                                {getQueryErrorMessage(runMutation.error, "Agent run failed.")}
                            </Alert>
                        ) : null}
                    </Stack>
                </SectionCard>

                <SectionCard
                    title="Run history"
                    description="Persisted runs for your account, including runs from earlier sessions."
                >
                    <Stack spacing={1.5}>
                        <Button
                            variant="text"
                            size="small"
                            onClick={() => void runsQuery.refetch()}
                            disabled={runsQuery.isFetching}
                            sx={{ alignSelf: "flex-start" }}
                        >
                            {runsQuery.isFetching ? "Refreshing…" : "Refresh history"}
                        </Button>
                        <QueryBoundary
                            isLoading={runsQuery.isLoading}
                            isError={runsQuery.isError}
                            error={runsQuery.error}
                            onRetry={() => void runsQuery.refetch()}
                        >
                            {history.length === 0 ? (
                                <EmptyState
                                    icon={<AgentIcon />}
                                    title="No agent runs yet"
                                    description="Submit a run to inspect response, citations, cost, and degradation."
                                />
                            ) : (
                                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                    {history.map((run) => (
                                        <Chip
                                            key={run.id}
                                            label={`${run.id.slice(0, 8)}… · ${run.status}`}
                                            color={selectedRunId === run.id ? "primary" : "default"}
                                            onClick={() => setSelectedRunId(run.id)}
                                        />
                                    ))}
                                </Stack>
                            )}
                        </QueryBoundary>
                    </Stack>
                </SectionCard>

                {selectedRunId ? (
                    <QueryBoundary
                        isLoading={selectedRunQuery.isLoading}
                        isError={selectedRunQuery.isError}
                        error={selectedRunQuery.error}
                        onRetry={() => void selectedRunQuery.refetch()}
                    >
                        {selectedRun ? <AgentRunDetail run={selectedRun} /> : null}
                    </QueryBoundary>
                ) : null}
            </Stack>
        </PageShell>
    );
}
