import { useEffect, useState } from "react";
import {
    Alert,
    Box,
    Button,
    Chip,
    CircularProgress,
    Divider,
    Link,
    Stack,
    Tab,
    Tabs,
    TextField,
    Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "../../../../config/queryKeys";
import {
    createAssistantThread,
    getAssistantConversation,
    getAssistantScope,
    listAssistantThreads,
    postAssistantMessage,
    type AssistantCitation,
    type AssistantMessageResult,
    type AssistantThread,
} from "../../../../api/textResearch";
import { getQueryErrorMessage } from "../../../../utils/queryErrors";
import { ResearchContextBar } from "../ResearchShared";
import { useResearchContext } from "../../hooks/useResearchContext";

type PanelMode = "context" | "ask" | "evidence";

const SUGGESTIONS = [
    "Compare documents",
    "Find supporting evidence",
    "Find counter-evidence",
    "Summarize positions",
    "Find representative passages",
];

type Props = {
    onOpenCitation?: (citation: AssistantCitation) => void;
};

export function CorpusAssistantPanel({ onOpenCitation }: Props) {
    const queryClient = useQueryClient();
    const [mode, setMode] = useState<PanelMode>("ask");
    const [question, setQuestion] = useState("");
    const [threadId, setThreadId] = useState<string | null>(null);
    const [lastResult, setLastResult] = useState<AssistantMessageResult | null>(null);
    const [evidenceFilter, setEvidenceFilter] = useState<"all" | "used" | "unused">("all");
    const [defaultIntent, setDefaultIntent] = useState("evidence");

    // Corpus from research workspace context
    const ctx = useResearchContext();
    const corpusId = ctx.selectedCorpus?.id ?? "";

    const scopeQuery = useQuery({
        queryKey: queryKeys.textResearch.assistantScope(corpusId),
        queryFn: () => getAssistantScope(corpusId),
        enabled: Boolean(corpusId),
    });

    const threadsQuery = useQuery({
        queryKey: queryKeys.textResearch.assistantThreads(corpusId),
        queryFn: () => listAssistantThreads(corpusId),
        enabled: Boolean(corpusId),
    });

    const conversationQuery = useQuery({
        queryKey: queryKeys.textResearch.assistantConversation(threadId ?? ""),
        queryFn: () => getAssistantConversation(threadId!),
        enabled: Boolean(threadId),
    });

    useEffect(() => {
        setThreadId(null);
        setLastResult(null);
        setQuestion("");
    }, [corpusId]);

    const askMutation = useMutation({
        mutationFn: (payload: { query: string; intent?: string }) =>
            postAssistantMessage(corpusId, {
                query: payload.query,
                thread_id: threadId,
                intent: payload.intent ?? "evidence",
            }),
        onSuccess: (result: AssistantMessageResult) => {
            setThreadId(result.thread_id);
            setLastResult(result);
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.assistantThreads(corpusId),
            });
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.assistantConversation(result.thread_id),
            });
            setMode("ask");
        },
    });

    useEffect(() => {
        if (!ctx.pendingAsk) return;
        const pending = ctx.pendingAsk;
        setMode("ask");
        setQuestion(pending.question);
        setDefaultIntent(pending.intent ?? "evidence");
        ctx.clearPendingAsk();
        if (pending.autoSubmit !== false && corpusId && pending.question.trim()) {
            askMutation.mutate({
                query: pending.question.trim(),
                intent: pending.intent ?? "evidence",
            });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- consume once per pendingAsk.nonce
    }, [ctx.pendingAsk?.nonce]);

    const newThreadMutation = useMutation({
        mutationFn: () => createAssistantThread(corpusId),
        onSuccess: (thread: AssistantThread) => {
            setThreadId(thread.id);
            setLastResult(null);
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.assistantThreads(corpusId),
            });
        },
    });

    if (!corpusId) {
        return (
            <Stack spacing={1.5}>
                <Tabs
                    value={mode}
                    onChange={(_, value: PanelMode) => setMode(value)}
                    variant="fullWidth"
                >
                    <Tab value="context" label="Context" />
                    <Tab value="ask" label="Ask" />
                    <Tab value="evidence" label="Evidence" />
                </Tabs>
                {mode === "context" ? <ResearchContextBar /> : null}
                {mode !== "context" ? (
                    <Alert severity="info">Select a corpus to use Ask Corpus.</Alert>
                ) : null}
            </Stack>
        );
    }

    const scope = scopeQuery.data ?? lastResult?.scope;
    const liveHash = scopeQuery.data?.scope_hash;
    const answerHash = lastResult?.scope?.scope_hash;
    const corpusChangedSinceAnswer = Boolean(
        liveHash && answerHash && liveHash !== answerHash
    );
    const citations = lastResult?.citations ?? [];
    const filteredCitations = citations.filter((c) => {
        if (evidenceFilter === "used") return c.used_in_answer;
        if (evidenceFilter === "unused") return !c.used_in_answer;
        return true;
    });

    return (
        <Stack spacing={1.5} sx={{ minHeight: 0 }}>
            <Tabs
                value={mode}
                onChange={(_, value: PanelMode) => setMode(value)}
                variant="fullWidth"
                aria-label="Research side panel modes"
            >
                <Tab value="context" label="Context" />
                <Tab value="ask" label="Ask" />
                <Tab value="evidence" label="Evidence" />
            </Tabs>

            {mode === "context" ? <ResearchContextBar /> : null}

            {mode === "ask" ? (
                <Stack spacing={1.5}>
                    <Typography variant="subtitle1" fontWeight={600}>
                        Ask Corpus
                    </Typography>
                    {scope ? (
                        <Box>
                            <Typography variant="body2">{scope.corpus_name}</Typography>
                            <Typography variant="caption" color="text.secondary">
                                {scope.total_documents} documents · {scope.indexed_count} indexed
                                {scope.unavailable_count
                                    ? ` · ${scope.unavailable_count} unavailable`
                                    : ""}
                            </Typography>
                            {scope.warnings.map((warning: string) => (
                                <Alert key={warning} severity="warning" sx={{ mt: 1 }}>
                                    {warning}
                                </Alert>
                            ))}
                        </Box>
                    ) : null}
                    {corpusChangedSinceAnswer ? (
                        <Alert
                            severity="warning"
                            action={
                                <Button
                                    color="inherit"
                                    size="small"
                                    onClick={() =>
                                        askMutation.mutate({
                                            query: lastResult?.query || question.trim(),
                                            intent: defaultIntent,
                                        })
                                    }
                                    disabled={
                                        askMutation.isPending ||
                                        !(lastResult?.query || question.trim())
                                    }
                                >
                                    Re-ask
                                </Button>
                            }
                        >
                            Corpus indexing or membership changed since this answer. Historical
                            evidence still reflects the original scope snapshot; re-ask to use the
                            current corpus.
                        </Alert>
                    ) : null}

                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {SUGGESTIONS.map((label) => (
                            <Chip
                                key={label}
                                size="small"
                                label={label}
                                onClick={() => {
                                    setQuestion(label);
                                    const intent =
                                        label === "Find counter-evidence"
                                            ? "contradiction"
                                            : label === "Compare documents"
                                              ? "comparison"
                                              : label === "Summarize positions"
                                                ? "synthesis"
                                                : "evidence";
                                    askMutation.mutate({ query: label, intent });
                                }}
                                disabled={askMutation.isPending || scope?.indexed_count === 0}
                            />
                        ))}
                    </Stack>

                    <TextField
                        label="Question"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        multiline
                        minRows={3}
                        fullWidth
                        size="small"
                    />
                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="contained"
                            disabled={
                                !question.trim() ||
                                askMutation.isPending ||
                                (scope?.indexed_count ?? 0) === 0
                            }
                            onClick={() =>
                                askMutation.mutate({
                                    query: question.trim(),
                                    intent: defaultIntent,
                                })
                            }
                        >
                            Ask
                        </Button>
                        <Button
                            variant="outlined"
                            size="small"
                            onClick={() => newThreadMutation.mutate()}
                            disabled={newThreadMutation.isPending}
                        >
                            New thread
                        </Button>
                    </Stack>

                    {askMutation.isPending ? (
                        <Stack direction="row" spacing={1} alignItems="center">
                            <CircularProgress size={18} />
                            <Typography variant="body2">Retrieving evidence…</Typography>
                        </Stack>
                    ) : null}
                    {askMutation.isError ? (
                        <Alert severity="error">
                            {getQueryErrorMessage(askMutation.error, "Ask Corpus failed.")}
                        </Alert>
                    ) : null}

                    {lastResult ? (
                        <Stack spacing={1}>
                            {lastResult.no_context_found ? (
                                <Alert severity="info">No relevant evidence found in this corpus.</Alert>
                            ) : null}
                            {lastResult.retrieval_degraded ? (
                                <Alert severity="warning">
                                    Retrieval degraded
                                    {lastResult.degradation_reason
                                        ? `: ${lastResult.degradation_reason}`
                                        : ""}
                                </Alert>
                            ) : null}
                            {lastResult.citation_validation_failed ? (
                                <Alert severity="warning">
                                    Some model citations were discarded during validation.
                                </Alert>
                            ) : null}
                            {lastResult.coverage.documents_in_scope > 0 ? (
                                <Typography variant="caption" color="text.secondary">
                                    Evidence retrieved from{" "}
                                    {lastResult.coverage.documents_with_retrieved_evidence} of{" "}
                                    {lastResult.coverage.documents_in_scope} documents
                                </Typography>
                            ) : null}
                            <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                                {lastResult.answer}
                            </Typography>
                            <Divider />
                            <Typography variant="subtitle2">Sources</Typography>
                            {lastResult.citations
                                .filter((c) => c.used_in_answer)
                                .map((citation) => (
                                    <Box key={citation.chunk_id}>
                                        <Link
                                            component="button"
                                            type="button"
                                            variant="body2"
                                            onClick={() => onOpenCitation?.(citation)}
                                            aria-label={`Open source ${citation.citation_number ?? ""} ${citation.filename}`}
                                        >
                                            [{citation.citation_number}] {citation.filename}
                                            {citation.page_number != null
                                                ? ` · p.${citation.page_number}`
                                                : ""}
                                        </Link>
                                        <Typography variant="caption" display="block" color="text.secondary">
                                            {citation.snippet}
                                        </Typography>
                                    </Box>
                                ))}
                        </Stack>
                    ) : null}

                    {(threadsQuery.data?.length ?? 0) > 0 ? (
                        <Box>
                            <Typography variant="caption" color="text.secondary">
                                Recent threads
                            </Typography>
                            <Stack spacing={0.5} mt={0.5}>
                                {threadsQuery.data?.slice(0, 5).map((thread) => (
                                    <Button
                                        key={thread.id}
                                        size="small"
                                        variant={thread.id === threadId ? "contained" : "text"}
                                        onClick={() => setThreadId(thread.id)}
                                        sx={{ justifyContent: "flex-start" }}
                                    >
                                        {thread.title}
                                    </Button>
                                ))}
                            </Stack>
                        </Box>
                    ) : null}

                    {conversationQuery.data && !lastResult ? (
                        <Alert severity="info">
                            Thread loaded ({conversationQuery.data.messages.length} messages). Ask a
                            follow-up to continue.
                        </Alert>
                    ) : null}
                </Stack>
            ) : null}

            {mode === "evidence" ? (
                <Stack spacing={1.5}>
                    <Typography variant="subtitle1" fontWeight={600}>
                        Retrieved evidence
                    </Typography>
                    {!lastResult ? (
                        <Alert severity="info">Ask a question to populate evidence.</Alert>
                    ) : (
                        <>
                            <Typography variant="body2">
                                {lastResult.coverage.retrieved_passage_count} passages ·{" "}
                                {lastResult.coverage.documents_with_retrieved_evidence} documents
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Coverage: {lastResult.coverage.documents_with_retrieved_evidence} /{" "}
                                {lastResult.coverage.documents_in_scope} documents
                            </Typography>
                            <Stack direction="row" spacing={1}>
                                {(["all", "used", "unused"] as const).map((value) => (
                                    <Chip
                                        key={value}
                                        size="small"
                                        label={
                                            value === "all"
                                                ? "All"
                                                : value === "used"
                                                  ? "Used in answer"
                                                  : "Unused retrieved"
                                        }
                                        color={evidenceFilter === value ? "primary" : "default"}
                                        onClick={() => setEvidenceFilter(value)}
                                    />
                                ))}
                            </Stack>
                            {filteredCitations.map((citation) => (
                                <Box
                                    key={citation.chunk_id}
                                    sx={{ borderBottom: 1, borderColor: "divider", pb: 1 }}
                                >
                                    <Link
                                        component="button"
                                        type="button"
                                        variant="body2"
                                        onClick={() => onOpenCitation?.(citation)}
                                    >
                                        {citation.filename}
                                        {citation.page_number != null
                                            ? ` · p.${citation.page_number}`
                                            : ""}
                                    </Link>
                                    <Typography variant="caption" display="block">
                                        {citation.used_in_answer ? "Used in answer" : "Retrieved only"}
                                        {citation.citation_number != null
                                            ? ` · [${citation.citation_number}]`
                                            : ""}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        {citation.snippet}
                                    </Typography>
                                </Box>
                            ))}
                        </>
                    )}
                </Stack>
            ) : null}
        </Stack>
    );
}

export default CorpusAssistantPanel;
