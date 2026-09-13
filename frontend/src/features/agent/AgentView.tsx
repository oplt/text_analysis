import { useEffect, useMemo, useState } from "react";
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
    TextField,
    Typography,
} from "@mui/material";
import { SmartToy as AgentIcon, PlayArrow as RunIcon } from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSnackbar } from "../../app/snackbarContext";
import { createAgentRun, getAgentRun, listAgentRuns, formatAgentCostMicros } from "../../api/agent";
import { getAiOverview } from "../../api/ai";
import { listProjects } from "../../api/projects";
import { listRagDocuments } from "../../api/rag";
import { AdvancedSettings } from "../../components/ui/AdvancedSettings";
import { DisabledWithReason } from "../../components/ui/DisabledWithReason";
import { EmptyState } from "../../components/ui/EmptyState";
import { HelpFieldLabel, HelpTooltip } from "../../components/ui/HelpTooltip";
import { JsonBlock } from "../../components/ui/JsonBlock";
import { KeyValueList } from "../../components/ui/KeyValueList";
import { PageHeader } from "../../components/ui/PageHeader";
import { PageShell } from "../../components/ui/PageShell";
import { PageTabs } from "../../components/ui/PageTabs";
import { QueryBoundary } from "../../components/ui/QueryBoundary";
import { RunStatusChip } from "../../components/ui/RunStatusChip";
import { RunStatusPanel } from "../../components/ui/RunStatusPanel";
import { SectionCard } from "../../components/ui/SectionCard";
import { recordToKeyValueItems } from "../../components/ui/jsonDisplay";
import { agentRunDisabledReason } from "../text-research/actionDisabledReasons";
import { SettingsTabs } from "../../components/layout/SettingsTabs";
import { queryKeys } from "../../config/queryKeys";
import { useTabQueryParam } from "../../hooks/useTabQueryParam";
import { getQueryErrorMessage } from "../../utils/queryErrors";
import { invalidateAgentRunQueries } from "./agentQueryUtils";
import {
    agentRunElapsedMs,
    buildAgentTraceSteps,
    formatElapsed,
    type AgentTraceCategory,
} from "./agentTraceModel";

const AGENT_TABS = ["configure", "run", "trace", "sources", "output"] as const;

function categoryLabel(category: AgentTraceCategory): string {
    switch (category) {
        case "plan":
            return "Plan";
        case "tool":
            return "Tool call";
        case "result":
            return "Result";
        case "error":
            return "Error";
        case "meta":
            return "Meta";
        default:
            return category;
    }
}

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
    const [tab, setTab] = useTabQueryParam(AGENT_TABS, "configure");
    const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
    const [elapsedMs, setElapsedMs] = useState(0);

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

    const templates = overviewQuery.data?.prompt_templates ?? [];
    const providers = overviewQuery.data?.providers ?? [];
    const selectedTemplate = templates.find((t) => t.key === promptTemplateKey) ?? null;

    const runsQuery = useQuery({
        queryKey: queryKeys.agent.runs,
        queryFn: () => listAgentRuns({ limit: 30 }),
    });

    const selectedRunQuery = useQuery({
        queryKey: queryKeys.agent.run(selectedRunId ?? ""),
        queryFn: () => getAgentRun(selectedRunId!),
        enabled: Boolean(selectedRunId),
    });

    const runMutation = useMutation({
        mutationFn: () =>
            createAgentRun({
                agent_id: agentId.trim() || "default",
                project_id: projectId || undefined,
                user_message: userMessage.trim() || undefined,
                prompt_template_key: promptTemplateKey || undefined,
                retrieval_query: retrievalQuery.trim() || userMessage.trim() || undefined,
                document_ids: selectedDocumentIds,
                top_k: topK,
            }),
        onMutate: () => {
            setRunStartedAt(Date.now());
            setElapsedMs(0);
            setTab("run");
        },
        onSuccess: async (run) => {
            setSelectedRunId(run.id);
            setRunStartedAt(null);
            setTab("output");
            await invalidateAgentRunQueries(queryClient, run.id);
            showToast({ message: "Agent run completed.", severity: "success" });
        },
        onError: (error) => {
            setRunStartedAt(null);
            showToast({
                message: getQueryErrorMessage(error, "Agent run failed."),
                severity: "error",
            });
        },
    });

    useEffect(() => {
        if (!runMutation.isPending || runStartedAt == null) return;
        const tick = window.setInterval(() => {
            setElapsedMs(Date.now() - runStartedAt);
        }, 200);
        return () => window.clearInterval(tick);
    }, [runMutation.isPending, runStartedAt]);

    const history = runsQuery.data?.items ?? [];
    const selectedRun = selectedRunQuery.data ?? null;
    const traceSteps = useMemo(
        () => (selectedRun ? buildAgentTraceSteps(selectedRun) : []),
        [selectedRun]
    );

    const enabledTools = useMemo(() => {
        const tools = [
            { id: "generate", label: "Generate", enabled: true },
            {
                id: "retrieve",
                label: "RAG retrieve",
                enabled: Boolean(retrievalQuery.trim() || userMessage.trim() || selectedDocumentIds.length),
            },
            { id: "memory", label: "Memory link", enabled: true },
        ];
        return tools;
    }, [retrievalQuery, userMessage, selectedDocumentIds.length]);

    function toggleDocument(id: string) {
        setSelectedDocumentIds((ids) =>
            ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
        );
    }

    return (
        <PageShell maxWidth="xl">
            <PageHeader
                title="AI Agent"
                description="Transparent agent runs: configure tools and retrieval, watch progress, inspect structured trace and sources, then read the final output separately."
            />
            <SettingsTabs />

            <Stack spacing={2} sx={{ mt: 2 }}>
                <PageTabs
                    value={tab}
                    onChange={setTab}
                    tabs={[
                        { value: "configure", label: "Configure" },
                        { value: "run", label: "Run" },
                        { value: "trace", label: "Trace" },
                        { value: "sources", label: "Sources" },
                        { value: "output", label: "Output" },
                    ]}
                    ariaLabel="Agent workflow"
                />

                {tab === "configure" ? (
                    <SectionCard
                        title="Configure"
                        description="Model, tools, prompt, limits, and retrieval — then continue to Run."
                    >
                        <Stack spacing={2}>
                            <Alert severity="info">
                                Execution is synchronous today (HTTP 201). The Run tab shows live
                                status until the backend responds or times out.
                            </Alert>

                            <Typography variant="subtitle2">
                                <HelpTooltip termId="model_family" variant="label">
                                    Model
                                </HelpTooltip>
                            </Typography>
                            <KeyValueList
                                showHelp
                                items={[
                                    {
                                        key: "agent_id",
                                        label: "Agent ID",
                                        value: agentId || "default",
                                    },
                                    {
                                        key: "providers",
                                        label: "Available providers",
                                        value: providers.length
                                            ? providers
                                                  .map((p) => `${p.label} (${p.key})`)
                                                  .join(", ")
                                            : "—",
                                        helpTermId: "model_family",
                                    },
                                    {
                                        key: "prompt_template",
                                        label: "Prompt template",
                                        value: selectedTemplate
                                            ? `${selectedTemplate.name} (${selectedTemplate.key})`
                                            : "Message only (no template)",
                                    },
                                ]}
                            />

                            <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={2}
                                flexWrap="wrap"
                                useFlexGap
                            >
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
                            </Stack>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    <HelpTooltip termId="agent_tools" variant="label">
                                        Tools
                                    </HelpTooltip>
                                </Typography>
                                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                    {enabledTools.map((tool) => (
                                        <Chip
                                            key={tool.id}
                                            size="small"
                                            color={tool.enabled ? "primary" : "default"}
                                            variant={tool.enabled ? "filled" : "outlined"}
                                            label={tool.label}
                                        />
                                    ))}
                                </Stack>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    display="block"
                                    sx={{ mt: 0.5 }}
                                >
                                    Enabled tools appear as structured steps on the Trace tab after a
                                    run.
                                </Typography>
                            </Box>

                            <Typography variant="subtitle2">Prompt</Typography>
                            <TextField
                                size="small"
                                label="User message"
                                value={userMessage}
                                onChange={(e) => setUserMessage(e.target.value)}
                                multiline
                                minRows={3}
                                fullWidth
                            />

                            <Typography variant="subtitle2">Limits & retrieval</Typography>
                            <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={2}
                                flexWrap="wrap"
                                useFlexGap
                            >
                                <TextField
                                    size="small"
                                    type="number"
                                    label={
                                        <HelpFieldLabel termId="top_k_retrieval">
                                            Top-k
                                        </HelpFieldLabel>
                                    }
                                    value={topK}
                                    onChange={(e) => setTopK(Number(e.target.value) || 4)}
                                    inputProps={{ min: 1, max: 20 }}
                                    sx={{ width: 120 }}
                                />
                                <TextField
                                    size="small"
                                    label="Retrieval query (optional)"
                                    value={retrievalQuery}
                                    onChange={(e) => setRetrievalQuery(e.target.value)}
                                    sx={{ minWidth: 280, flex: 1 }}
                                    helperText="Defaults to the user message when empty."
                                />
                            </Stack>

                            <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                    RAG documents (optional scope)
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
                                        <Stack
                                            direction="row"
                                            spacing={1}
                                            flexWrap="wrap"
                                            useFlexGap
                                        >
                                            {(documentsQuery.data?.items ?? []).map((doc) => {
                                                const selected = selectedDocumentIds.includes(
                                                    doc.id
                                                );
                                                return (
                                                    <Chip
                                                        key={doc.id}
                                                        label={
                                                            doc.original_filename || doc.filename
                                                        }
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
                                onClick={() => setTab("run")}
                                disabled={!userMessage.trim()}
                            >
                                Continue to Run
                            </Button>
                        </Stack>
                    </SectionCard>
                ) : null}

                {tab === "run" ? (
                    <>
                        <SectionCard
                            title="Run"
                            description="Current task, step, progress, elapsed time, and status."
                        >
                            <Stack spacing={2}>
                                <RunStatusPanel
                                    title="Agent run"
                                    status={
                                        runMutation.isPending
                                            ? "running"
                                            : runMutation.isError
                                              ? "failed"
                                              : selectedRun?.status ?? "queued"
                                    }
                                    runId={selectedRun?.id}
                                    stage={
                                        runMutation.isPending
                                            ? "Calling agent (retrieve → generate)…"
                                            : selectedRun
                                              ? `Last run ${selectedRun.status}`
                                              : userMessage.trim()
                                                ? "Idle — ready to run"
                                                : "Idle — enter a task on Configure"
                                    }
                                    startedAt={selectedRun?.created_at}
                                    completedAt={selectedRun?.completed_at}
                                    latencyMs={
                                        runMutation.isPending
                                            ? elapsedMs
                                            : selectedRun
                                              ? agentRunElapsedMs(selectedRun)
                                              : null
                                    }
                                    errorMessage={
                                        runMutation.isError
                                            ? getQueryErrorMessage(
                                                  runMutation.error,
                                                  "Agent run failed."
                                              )
                                            : selectedRun?.error_message
                                    }
                                    onRetry={
                                        userMessage.trim()
                                            ? () => runMutation.mutate()
                                            : undefined
                                    }
                                    retryLabel="Run agent"
                                    retryDisabled={runMutation.isPending || !userMessage.trim()}
                                />

                                <KeyValueList
                                    dense
                                    items={[
                                        {
                                            key: "task",
                                            label: "Current task",
                                            value: userMessage.trim()
                                                ? userMessage.trim().slice(0, 160) +
                                                  (userMessage.trim().length > 160 ? "…" : "")
                                                : "—",
                                        },
                                    ]}
                                />

                                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                                    <DisabledWithReason
                                        reason={agentRunDisabledReason({
                                            message: userMessage,
                                            pending: runMutation.isPending,
                                        })}
                                    >
                                        <Button
                                            variant="contained"
                                            startIcon={<RunIcon />}
                                            disabled={
                                                runMutation.isPending || !userMessage.trim()
                                            }
                                            onClick={() => runMutation.mutate()}
                                        >
                                            {runMutation.isPending ? "Running…" : "Run agent"}
                                        </Button>
                                    </DisabledWithReason>
                                    <Button
                                        size="small"
                                        variant="text"
                                        onClick={() => setTab("configure")}
                                    >
                                        Edit configuration
                                    </Button>
                                </Stack>
                            </Stack>
                        </SectionCard>

                        <SectionCard
                            title="Run history"
                            description="Persisted runs for your account. Select one to inspect Trace, Sources, and Output."
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
                                        <Box sx={{ overflowX: "auto" }}>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Run</TableCell>
                                                        <TableCell>Status</TableCell>
                                                        <TableCell>Model</TableCell>
                                                        <TableCell>Duration</TableCell>
                                                        <TableCell>When</TableCell>
                                                        <TableCell align="right">Open</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {history.map((run) => (
                                                        <TableRow
                                                            key={run.id}
                                                            selected={selectedRunId === run.id}
                                                            hover
                                                        >
                                                            <TableCell>
                                                                {run.id.slice(0, 8)}…
                                                                {run.agent_id
                                                                    ? ` · ${run.agent_id}`
                                                                    : ""}
                                                            </TableCell>
                                                            <TableCell>
                                                                <RunStatusChip status={run.status} />
                                                            </TableCell>
                                                            <TableCell>
                                                                {run.provider_key}/
                                                                {run.model_name}
                                                            </TableCell>
                                                            <TableCell>
                                                                {run.latency_ms != null
                                                                    ? formatElapsed(run.latency_ms)
                                                                    : "—"}
                                                            </TableCell>
                                                            <TableCell>
                                                                {new Date(
                                                                    run.created_at
                                                                ).toLocaleString()}
                                                            </TableCell>
                                                            <TableCell align="right">
                                                                <Button
                                                                    size="small"
                                                                    variant={
                                                                        selectedRunId === run.id
                                                                            ? "contained"
                                                                            : "outlined"
                                                                    }
                                                                    onClick={() => {
                                                                        setSelectedRunId(run.id);
                                                                        setTab("output");
                                                                    }}
                                                                >
                                                                    {`${run.id.slice(0, 8)}… · ${run.status}`}
                                                                </Button>
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
                                        </Box>
                                    )}
                                </QueryBoundary>
                            </Stack>
                        </SectionCard>
                    </>
                ) : null}

                {tab === "trace" || tab === "sources" || tab === "output" ? (
                    !selectedRunId ? (
                        <SectionCard
                            title={
                                tab === "trace" ? "Trace" : tab === "sources" ? "Sources" : "Output"
                            }
                            description="Select or create a run first."
                        >
                            <EmptyState
                                icon={<AgentIcon />}
                                title="No run selected"
                                description="Run an agent or pick a history item on the Run tab."
                                action={
                                    <Button variant="contained" onClick={() => setTab("run")}>
                                        Open Run
                                    </Button>
                                }
                            />
                        </SectionCard>
                    ) : (
                        <QueryBoundary
                            isLoading={selectedRunQuery.isLoading}
                            isError={selectedRunQuery.isError}
                            error={selectedRunQuery.error}
                            onRetry={() => void selectedRunQuery.refetch()}
                        >
                            {selectedRun && tab === "trace" ? (
                                <SectionCard
                                    title="Trace"
                                    description="Structured steps (plan, tool calls, results, errors) — not a raw log dump."
                                >
                                    <Stack spacing={1.5}>
                                        <Stack
                                            direction="row"
                                            spacing={1}
                                            flexWrap="wrap"
                                            useFlexGap
                                        >
                                            <Chip size="small" label={`run ${selectedRun.id}`} />
                                            <RunStatusChip status={selectedRun.status} />
                                            <HelpTooltip termId="agent_trace" />
                                        </Stack>

                                        {(selectedRun.retrieval_degraded ||
                                            selectedRun.memory_degraded ||
                                            selectedRun.error_message) && (
                                            <Alert
                                                severity={
                                                    selectedRun.error_message ? "error" : "warning"
                                                }
                                            >
                                                {selectedRun.error_message ??
                                                    selectedRun.degradation_reason ??
                                                    (selectedRun.retrieval_degraded
                                                        ? "Retrieval degraded"
                                                        : "Memory degraded")}
                                            </Alert>
                                        )}

                                        <Stack spacing={1}>
                                            {traceSteps.map((step) => (
                                                <Box
                                                    key={step.id}
                                                    sx={{
                                                        p: 1.5,
                                                        borderRadius: 1,
                                                        bgcolor: "action.hover",
                                                        borderLeft: 3,
                                                        borderColor:
                                                            step.status === "error"
                                                                ? "error.main"
                                                                : step.status === "warn"
                                                                  ? "warning.main"
                                                                  : "primary.main",
                                                    }}
                                                >
                                                    <Stack
                                                        direction={{ xs: "column", sm: "row" }}
                                                        spacing={1}
                                                        justifyContent="space-between"
                                                    >
                                                        <Typography variant="subtitle2">
                                                            {step.title}
                                                        </Typography>
                                                        <Stack
                                                            direction="row"
                                                            spacing={0.75}
                                                            alignItems="center"
                                                        >
                                                            <Chip
                                                                size="small"
                                                                variant="outlined"
                                                                label={categoryLabel(step.category)}
                                                            />
                                                            {step.durationMs != null ? (
                                                                <Typography
                                                                    variant="caption"
                                                                    color="text.secondary"
                                                                >
                                                                    {formatElapsed(step.durationMs)}
                                                                </Typography>
                                                            ) : null}
                                                        </Stack>
                                                    </Stack>
                                                    {step.detail ? (
                                                        <Typography
                                                            variant="body2"
                                                            color="text.secondary"
                                                            sx={{ mt: 0.5, whiteSpace: "pre-wrap" }}
                                                        >
                                                            {step.detail}
                                                        </Typography>
                                                    ) : null}
                                                </Box>
                                            ))}
                                        </Stack>

                                        <Typography variant="body2" color="text.secondary">
                                            Tokens: {selectedRun.total_tokens} (in{" "}
                                            {selectedRun.input_tokens} / out{" "}
                                            {selectedRun.output_tokens}) · Est. cost:{" "}
                                            {formatAgentCostMicros(
                                                selectedRun.estimated_cost_micros
                                            )}
                                        </Typography>

                                        <AdvancedSettings title="Raw run identifiers">
                                            <KeyValueList
                                                items={[
                                                    {
                                                        key: "memory_run_id",
                                                        label: "Memory run",
                                                        value: selectedRun.memory_run_id,
                                                    },
                                                    {
                                                        key: "provider",
                                                        label: "Provider / model",
                                                        value: `${selectedRun.provider_key}/${selectedRun.model_name}`,
                                                    },
                                                ]}
                                            />
                                        </AdvancedSettings>
                                    </Stack>
                                </SectionCard>
                            ) : null}

                            {selectedRun && tab === "sources" ? (
                                <SectionCard
                                    title="Sources"
                                    description="Citations and retrieved evidence attached to this run."
                                >
                                    <Stack spacing={1.5}>
                                        {selectedRun.retrieval_query ? (
                                            <Typography variant="body2" color="text.secondary">
                                                Retrieval query: {selectedRun.retrieval_query}
                                            </Typography>
                                        ) : null}

                                        {selectedRun.retrieved_chunk_ids.length === 0 ? (
                                            <Typography variant="body2" color="text.secondary">
                                                No retrieved evidence for this run.
                                            </Typography>
                                        ) : (
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>#</TableCell>
                                                        <TableCell>Chunk ID</TableCell>
                                                        <TableCell>Role</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {selectedRun.retrieved_chunk_ids.map(
                                                        (chunkId, index) => (
                                                            <TableRow key={chunkId}>
                                                                <TableCell>{index + 1}</TableCell>
                                                                <TableCell
                                                                    sx={{
                                                                        fontFamily: "monospace",
                                                                        fontSize: 12,
                                                                    }}
                                                                >
                                                                    {chunkId}
                                                                </TableCell>
                                                                <TableCell>
                                                                    Retrieved evidence
                                                                </TableCell>
                                                            </TableRow>
                                                        )
                                                    )}
                                                </TableBody>
                                            </Table>
                                        )}

                                        {selectedRun.injection_chunks_filtered > 0 ? (
                                            <Alert severity="info" icon={false}>
                                                Filtered {selectedRun.injection_chunks_filtered}{" "}
                                                injection chunk(s) before generation.
                                            </Alert>
                                        ) : null}

                                        {(selectedRun.retrieval_degraded ||
                                            selectedRun.memory_degraded) && (
                                            <Alert severity="warning">
                                                {selectedRun.degradation_reason ??
                                                    (selectedRun.retrieval_degraded
                                                        ? "Retrieval degraded — treat sources cautiously."
                                                        : "Memory degraded.")}
                                            </Alert>
                                        )}
                                    </Stack>
                                </SectionCard>
                            ) : null}

                            {selectedRun && tab === "output" ? (
                                <SectionCard
                                    title="Output"
                                    description="Final generated artifact — separate from Trace."
                                >
                                    <Stack spacing={1.5}>
                                        <Stack
                                            direction="row"
                                            spacing={1}
                                            flexWrap="wrap"
                                            useFlexGap
                                        >
                                            <RunStatusChip status={selectedRun.status} />
                                            <Chip
                                                size="small"
                                                variant="outlined"
                                                label={`${selectedRun.provider_key}/${selectedRun.model_name}`}
                                            />
                                            <Button
                                                size="small"
                                                variant="text"
                                                onClick={() => setTab("trace")}
                                            >
                                                View trace
                                            </Button>
                                            <Button
                                                size="small"
                                                variant="text"
                                                onClick={() => setTab("sources")}
                                            >
                                                View sources
                                            </Button>
                                        </Stack>

                                        <Typography
                                            variant="body1"
                                            sx={{ whiteSpace: "pre-wrap" }}
                                        >
                                            {selectedRun.output_text?.trim()
                                                ? selectedRun.output_text
                                                : selectedRun.output_json
                                                  ? "Structured JSON output (see Raw below)."
                                                  : "—"}
                                        </Typography>

                                        {selectedRun.output_json ? (
                                            <Stack spacing={1}>
                                                {recordToKeyValueItems(selectedRun.output_json)
                                                    .length ? (
                                                    <KeyValueList
                                                        dense
                                                        items={recordToKeyValueItems(
                                                            selectedRun.output_json
                                                        )}
                                                    />
                                                ) : null}
                                                <AdvancedSettings title="Raw JSON artifact">
                                                    <JsonBlock data={selectedRun.output_json} />
                                                </AdvancedSettings>
                                            </Stack>
                                        ) : null}
                                    </Stack>
                                </SectionCard>
                            ) : null}
                        </QueryBoundary>
                    )
                ) : null}
            </Stack>
        </PageShell>
    );
}
