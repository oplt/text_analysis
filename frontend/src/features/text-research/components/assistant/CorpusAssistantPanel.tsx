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
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "../../../../config/queryKeys";
import { researchRunStaleTime } from "../../../../config/queryTiming";
import {
    createAssistantThread,
    getAssistantConversation,
    getAssistantScope,
    getRun,
    listAssistantThreads,
    postAssistantMessageStream,
    postAssistantSynthesize,
    createResearchMemo,
    saveAssistantAsResearchMemo,
    type AssistantCitation,
    type AssistantClaim,
    type AssistantMessageResult,
    type AssistantScope,
    type AssistantScopeEvent,
    type AssistantSynthesizeResult,
    type AssistantSynthesisProvenance,
    type AssistantThread,
} from "../../../../api/textResearch";
import { getQueryErrorMessage } from "../../../../utils/queryErrors";
import { ResearchContextBar } from "../ResearchShared";
import { useResearchContext } from "../../hooks/useResearchContext";
import { useRunEvents } from "../../hooks/useRunEvents";
import { activeRunRefetchInterval } from "../../runPolling";
import { AssistantScopeControls } from "./AssistantScopeControls";
import { EvidenceProvenance } from "./EvidenceProvenance";

type PanelMode = "context" | "ask" | "evidence";

type ResearcherMode = "quick" | "evidence_search" | "synthesis";

const RESEARCHER_MODE_INTENT: Record<ResearcherMode, string> = {
    quick: "evidence",
    evidence_search: "semantic_search",
    synthesis: "synthesis",
};

const RESEARCHER_MODE_LABEL: Record<ResearcherMode, string> = {
    quick: "Quick answer",
    evidence_search: "Evidence search",
    synthesis: "Corpus synthesis",
};

const SUGGESTIONS = [
    "Compare documents",
    "Find supporting evidence",
    "Find counter-evidence",
    "Summarize positions",
    "Find representative passages",
];

type ConversationMessage = {
    id?: string;
    role?: string;
    content?: string;
    citations?: AssistantCitation[];
    claims?: AssistantClaim[];
    evidence_revision_hash?: string | null;
    metadata?: Record<string, unknown>;
    created_at?: string;
};

type Props = {
    onOpenCitation?: (citation: AssistantCitation) => void;
};

function emptyCoverage() {
    return {
        documents_in_scope: 0,
        documents_with_retrieved_evidence: 0,
        retrieved_passage_count: 0,
        coverage_ratio: 0,
    };
}

function emptyScope(partial?: Partial<AssistantScope> | null): AssistantScope {
    return {
        corpus_id: "",
        project_id: "",
        corpus_name: "",
        rag_document_ids: [],
        corpus_document_ids: [],
        indexed_rag_document_ids: [],
        unavailable_rag_document_ids: [],
        scope_hash: "",
        evidence_revision_hash: null,
        index_version: null,
        retrieval_version: null,
        total_documents: 0,
        indexed_count: 0,
        unavailable_count: 0,
        unavailable_corpus_document_ids: [],
        unavailable_reasons: {},
        documents: [],
        document_bindings: [],
        warnings: [],
        scope_mode: "fixed",
        ...partial,
    };
}

function citationValidationWarning(status: string | undefined | null): string | null {
    if (!status || status === "valid" || status === "no_evidence") return null;
    if (status === "unstructured") {
        return "Citation validation: answer citations were unstructured and could not be fully verified.";
    }
    if (status === "invalid") {
        return "Citation validation: some citations were invalid and discarded.";
    }
    if (status === "partial") {
        return "Citation validation: some citations were only partially validated.";
    }
    if (status === "missing_claims") {
        return "Citation validation: the answer did not contain any validated claims.";
    }
    if (status === "incomplete") {
        return "Citation validation: one or more claims were structurally incomplete.";
    }
    if (status === "answer_mismatch") {
        return "Citation validation: the answer contained statements not represented by its claims.";
    }
    if (status === "empty_answer") {
        return "Citation validation: the model returned an empty answer.";
    }
    return `Citation validation status: ${status}.`;
}

function messageToResult(
    msg: ConversationMessage,
    thread: AssistantThread,
    scope: AssistantScope | null | undefined
): AssistantMessageResult {
    const meta = msg.metadata ?? {};
    const coverage =
        (meta.coverage as AssistantMessageResult["coverage"] | undefined) ?? emptyCoverage();
    return {
        thread_id: thread.id,
        conversation_id: thread.rag_conversation_id,
        message_id: msg.id ?? "",
        retrieval_trace_id: (meta.retrieval_trace_id as string | null | undefined) ?? null,
        query: String(meta.original_query ?? meta.resolved_retrieval_query ?? ""),
        original_query: (meta.original_query as string | null | undefined) ?? null,
        resolved_retrieval_query:
            (meta.resolved_retrieval_query as string | null | undefined) ?? null,
        answer: msg.content ?? "",
        citations: Array.isArray(msg.citations) ? msg.citations : [],
        claims: Array.isArray(msg.claims) ? msg.claims : [],
        retrieved_chunk_ids: [],
        model_name: "",
        latency_ms: 0,
        no_context_found:
            Boolean(meta.no_context_found) || meta.citation_validation_status === "no_evidence",
        retrieval_degraded: Boolean(meta.retrieval_degraded),
        degradation_reason: (meta.degradation_reason as string | null | undefined) ?? null,
        citation_validation_failed: Boolean(meta.citation_validation_failed),
        citation_validation_status:
            (meta.citation_validation_status as string | undefined) ?? "valid",
        evidence_revision_hash:
            msg.evidence_revision_hash ??
            (meta.evidence_revision_hash as string | null | undefined) ??
            scope?.evidence_revision_hash ??
            null,
        injection_chunks_filtered: 0,
        scope: emptyScope(scope),
        coverage,
        context_message_ids: Array.isArray(meta.context_message_ids)
            ? (meta.context_message_ids as string[])
            : [],
        ai_run_id: (meta.ai_run_id as string | null | undefined) ?? null,
        fusion_method: (meta.fusion_method as string | null | undefined) ?? null,
    };
}

function synthesizeSyncToResult(
    result: Extract<AssistantSynthesizeResult, { mode: "sync" }>,
    query: string
): AssistantMessageResult {
    return {
        thread_id: "",
        conversation_id: "",
        message_id: "",
        retrieval_trace_id: result.retrieval_trace_id ?? null,
        query,
        original_query: query,
        resolved_retrieval_query: query,
        answer: result.answer,
        citations: result.citations ?? [],
        claims: result.claims ?? [],
        retrieved_chunk_ids: [],
        model_name: "",
        latency_ms: 0,
        no_context_found: !result.answer || result.citation_validation_status === "no_evidence",
        retrieval_degraded: false,
        degradation_reason: null,
        citation_validation_failed: Boolean(
            result.citation_validation_status &&
                result.citation_validation_status !== "valid" &&
                result.citation_validation_status !== "no_evidence"
        ),
        citation_validation_status: result.citation_validation_status ?? "valid",
        evidence_revision_hash:
            result.evidence_revision_hash ?? result.scope.evidence_revision_hash ?? null,
        synthesis_provenance: result.synthesis_provenance,
        injection_chunks_filtered: 0,
        scope: result.scope,
        coverage: result.coverage,
        context_message_ids: [],
        ai_run_id: null,
        fusion_method: "deterministic_map_reduce",
    };
}

function ClaimsWithCitations({
    claims,
    citations,
    fallbackAnswer,
    citationValidationFailed,
    noContextFound,
    onOpenCitation,
}: {
    claims: AssistantClaim[];
    citations: AssistantCitation[];
    fallbackAnswer: string;
    citationValidationFailed: boolean;
    noContextFound: boolean;
    onOpenCitation?: (citation: AssistantCitation) => void;
}) {
    if (!claims.length) {
        if (citationValidationFailed && !noContextFound) {
            return (
                <Alert severity="warning">
                    A fully cited answer could not be generated. Retrieved evidence is shown
                    separately below.
                </Alert>
            );
        }
        return (
            <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                {fallbackAnswer}
            </Typography>
        );
    }

    const byNumber = new Map<number, AssistantCitation>();
    for (const citation of citations) {
        if (citation.citation_number != null) {
            byNumber.set(citation.citation_number, citation);
        }
    }

    return (
        <Stack spacing={1}>
            {claims.map((claim, index) => (
                <Typography key={`${index}-${claim.text.slice(0, 24)}`} variant="body2">
                    {claim.text}{" "}
                    {claim.citation_numbers.map((num) => {
                        const citation = byNumber.get(num);
                        return (
                            <Link
                                key={`${index}-${num}`}
                                component="button"
                                type="button"
                                variant="body2"
                                sx={{ mx: 0.25, verticalAlign: "baseline" }}
                                onClick={() => {
                                    if (citation) onOpenCitation?.(citation);
                                }}
                                aria-label={`Open citation ${num}`}
                            >
                                [{num}]
                            </Link>
                        );
                    })}
                </Typography>
            ))}
        </Stack>
    );
}

function ScopeEventAudit({ events }: { events: AssistantScopeEvent[] }) {
    return (
        <Box>
            <Typography variant="subtitle2">Scope history</Typography>
            {events.length ? (
                <Stack spacing={0.5} mt={0.5}>
                    {events.map((event) => (
                        <Typography key={event.id} variant="caption" color="text.secondary">
                            {new Date(event.created_at).toLocaleString()} · {event.action.replaceAll("_", " ")}
                            {" · "}
                            {event.scope_mode === "live" ? "live corpus" : "fixed snapshot"}
                            {" · evidence "}
                            {event.evidence_revision_hash?.slice(0, 12) ?? "—"}
                            {event.reason ? ` · ${event.reason}` : ""}
                        </Typography>
                    ))}
                </Stack>
            ) : (
                <Typography variant="caption" color="text.secondary">
                    Scope changes will appear here.
                </Typography>
            )}
        </Box>
    );
}

export function CorpusAssistantPanel({ onOpenCitation }: Props) {
    const queryClient = useQueryClient();
    const [mode, setMode] = useState<PanelMode>("ask");
    const [researcherMode, setResearcherMode] = useState<ResearcherMode>("quick");
    const [question, setQuestion] = useState("");
    const [threadId, setThreadId] = useState<string | null>(null);
    const [lastResult, setLastResult] = useState<AssistantMessageResult | null>(null);
    const [evidenceFilter, setEvidenceFilter] = useState<"all" | "used" | "unused">("all");
    const [synthesisRunId, setSynthesisRunId] = useState<string | null>(null);
    const [synthesisQuery, setSynthesisQuery] = useState("");
    const [synthesisError, setSynthesisError] = useState<string | null>(null);
    const [lastSynthesisRunId, setLastSynthesisRunId] = useState<string | null>(null);
    const [assistantStage, setAssistantStage] = useState<string | null>(null);
    const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
    const [memoSavedMessageId, setMemoSavedMessageId] = useState<string | null>(null);
    const [threadDisplayLimit, setThreadDisplayLimit] = useState(5);

    const ctx = useResearchContext();
    const corpusId = ctx.selectedCorpus?.id ?? "";
    const projectId = ctx.projectId ?? "";

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

    const synthesisSseConnected = useRunEvents(synthesisRunId, projectId);
    const synthesisRunQuery = useQuery({
        queryKey: queryKeys.textResearch.run(synthesisRunId ?? ""),
        queryFn: ({ signal }) => getRun(synthesisRunId!, signal),
        enabled: Boolean(synthesisRunId),
        staleTime: (query) => researchRunStaleTime(query.state.data?.status),
        refetchInterval: (query) => activeRunRefetchInterval(query, synthesisSseConnected),
    });

    useEffect(() => {
        setThreadId(null);
        setLastResult(null);
        setQuestion("");
        setSynthesisRunId(null);
        setSynthesisQuery("");
        setSynthesisError(null);
        setLastSynthesisRunId(null);
        setThreadDisplayLimit(5);
    }, [corpusId]);

    useEffect(() => {
        const data = conversationQuery.data;
        if (!data?.thread || !threadId || data.thread.id !== threadId) return;
        const messages = (data.messages ?? []) as ConversationMessage[];
        const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
        if (lastAssistant) {
            setLastResult(messageToResult(lastAssistant, data.thread, data.scope));
        } else {
            setLastResult(null);
        }
    }, [conversationQuery.data, threadId]);

    useEffect(() => {
        const run = synthesisRunQuery.data;
        if (!run || !synthesisRunId || run.id !== synthesisRunId) return;
        if (run.status === "completed" && run.results) {
            const results = run.results as Extract<AssistantSynthesizeResult, { mode: "sync" }> &
                Record<string, unknown>;
            const mapped = synthesizeSyncToResult(
                {
                    mode: "sync",
                    answer: String(results.answer ?? ""),
                    citations: (results.citations as AssistantCitation[] | undefined) ?? [],
                    claims: (results.claims as AssistantClaim[] | undefined) ?? [],
                    scope: (results.scope as AssistantScope) ?? emptyScope(scopeQuery.data),
                    coverage:
                        (results.coverage as AssistantMessageResult["coverage"]) ?? emptyCoverage(),
                    retrieval_trace_id: (results.retrieval_trace_id as string | null) ?? null,
                    citation_validation_status:
                        (results.citation_validation_status as string | undefined) ?? "valid",
                    evidence_revision_hash:
                        (results.evidence_revision_hash as string | null | undefined) ??
                        null,
                    synthesis_provenance: results.synthesis_provenance as
                        | AssistantSynthesisProvenance
                        | undefined,
                },
                synthesisQuery
            );
            setLastResult(mapped);
            setSynthesisRunId(null);
            setSynthesisError(null);
            setMode("ask");
        }
        if (run.status === "failed" || run.status === "cancelled") {
            setSynthesisError(
                run.error_message ||
                    (run.status === "cancelled"
                        ? "Corpus synthesis was cancelled."
                        : "Corpus synthesis run failed.")
            );
            setSynthesisRunId(null);
        }
    }, [synthesisRunQuery.data, synthesisRunId, synthesisQuery, scopeQuery.data]);

    const askMutation = useMutation({
        mutationFn: async (payload: { query: string; intent?: string }) =>
            postAssistantMessageStream(
                corpusId,
                {
                query: payload.query,
                thread_id: threadId,
                intent: payload.intent ?? RESEARCHER_MODE_INTENT[researcherMode],
                },
                (event, eventPayload) => {
                    setAssistantStage(event);
                    if (event === "turn_created" && typeof eventPayload.thread_id === "string") {
                        setThreadId(eventPayload.thread_id);
                    }
                }
            ),
        onSuccess: (result: AssistantMessageResult) => {
            setAssistantStage(null);
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
        onError: () => {
            setAssistantStage(null);
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.assistantThreads(corpusId),
            });
        },
    });

    const synthesizeMutation = useMutation({
        mutationFn: (payload: { query: string }) =>
            postAssistantSynthesize(corpusId, { query: payload.query }),
        onSuccess: (result: AssistantSynthesizeResult, variables) => {
            if (result.mode === "async") {
                setSynthesisQuery(variables.query);
                setSynthesisRunId(result.run_id);
                setLastSynthesisRunId(result.run_id);
                setSynthesisError(null);
                setMode("ask");
                return;
            }
            setLastResult(synthesizeSyncToResult(result, variables.query));
            setSynthesisError(null);
            setMode("ask");
        },
    });

    const saveMemoMutation = useMutation({
        mutationFn: async () => {
            if (!lastResult) throw new Error("There is no answer to save yet.");
            const isSynthesis = Boolean(lastResult.synthesis_provenance) || Boolean(lastSynthesisRunId);
            const title = isSynthesis ? "Corpus synthesis" : "Ask Corpus answer";
            if (!isSynthesis && lastResult.message_id) {
                return saveAssistantAsResearchMemo(corpusId, {
                    message_id: lastResult.message_id,
                    title,
                    body: lastResult.answer,
                });
            }
            return createResearchMemo(projectId, {
                corpus_id: corpusId,
                title,
                body: lastResult.answer,
                source_type: isSynthesis ? "analysis_result" : "manual",
                citations: lastResult.citations as Array<Record<string, unknown>>,
                claims: lastResult.claims as Array<Record<string, unknown>>,
                provenance: {
                    query: lastResult.query,
                    scope: lastResult.scope,
                    synthesis_provenance: lastResult.synthesis_provenance,
                },
                evidence_revision_hash: lastResult.evidence_revision_hash,
                originating_synthesis_run_id: lastSynthesisRunId,
            });
        },
        onSuccess: () => {
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.memos(projectId, corpusId),
            });
        },
    });

    const savePersistedMessageMemoMutation = useMutation({
        mutationFn: (message: { id: string; body: string }) =>
            saveAssistantAsResearchMemo(corpusId, {
                message_id: message.id,
                title: "Ask Corpus answer",
                body: message.body,
            }),
        onSuccess: (_memo, message) => {
            setMemoSavedMessageId(message.id);
            void queryClient.invalidateQueries({
                queryKey: queryKeys.textResearch.memos(projectId, corpusId),
            });
        },
    });

    async function copyPersistedAnswer(messageId: string, content: string) {
        await navigator.clipboard.writeText(content);
        setCopiedMessageId(messageId);
    }

    function submitQuestion(query: string, intentOverride?: string) {
        const trimmed = query.trim();
        if (!trimmed) return;
        const intent = intentOverride ?? RESEARCHER_MODE_INTENT[researcherMode];
        if (intent === "synthesis") {
            synthesizeMutation.mutate({ query: trimmed });
            return;
        }
        askMutation.mutate({ query: trimmed, intent });
    }

    useEffect(() => {
        if (!ctx.pendingAsk) return;
        const pending = ctx.pendingAsk;
        setMode("ask");
        setQuestion(pending.question);
        if (pending.intent === "synthesis") {
            setResearcherMode("synthesis");
        } else if (pending.intent === "semantic_search") {
            setResearcherMode("evidence_search");
        } else {
            setResearcherMode("quick");
        }
        ctx.clearPendingAsk();
        if (pending.autoSubmit !== false && corpusId && pending.question.trim()) {
            submitQuestion(pending.question.trim(), pending.intent ?? "evidence");
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

    const isBusy =
        askMutation.isPending ||
        synthesizeMutation.isPending ||
        Boolean(synthesisRunId);

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

    const threadScope = conversationQuery.data?.scope ?? (threadId ? lastResult?.scope : null);
    const scope = threadScope ?? scopeQuery.data;
    const answerScope = lastResult?.scope ?? threadScope;
    const currentScope = scopeQuery.data;
    const membershipChanged = Boolean(
        answerScope &&
            currentScope &&
            answerScope.scope_hash &&
            currentScope.scope_hash &&
            answerScope.scope_hash !== currentScope.scope_hash
    );
    const evidenceRevisionChanged = Boolean(
        answerScope &&
            currentScope &&
            answerScope.evidence_revision_hash &&
            currentScope.evidence_revision_hash &&
            answerScope.evidence_revision_hash !== currentScope.evidence_revision_hash
    );
    const citations = lastResult?.citations ?? [];
    const filteredCitations = citations.filter((c) => {
        if (evidenceFilter === "used") return c.used_in_answer;
        if (evidenceFilter === "unused") return !c.used_in_answer;
        return true;
    });
    const conversationMessages = (conversationQuery.data?.messages ??
        []) as ConversationMessage[];
    const validationWarning = citationValidationWarning(lastResult?.citation_validation_status);
    const synthesisProvenance = lastResult?.synthesis_provenance;

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

            {mode === "context" ? (
                <Stack spacing={1.5}>
                    <ResearchContextBar />
                    <ScopeEventAudit
                        events={(conversationQuery.data?.scope_events ?? []) as AssistantScopeEvent[]}
                    />
                </Stack>
            ) : null}

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
                    <AssistantScopeControls
                        key={scope?.scope_hash ?? "no-scope"}
                        threadId={threadId}
                        scope={scope ?? null}
                        liveScope={scopeQuery.data ?? null}
                        onUpdated={() => {
                            setLastResult(null);
                            void queryClient.invalidateQueries({
                                queryKey: queryKeys.textResearch.assistantConversation(
                                    threadId ?? ""
                                ),
                            });
                            void queryClient.invalidateQueries({
                                queryKey: queryKeys.textResearch.assistantThreads(corpusId),
                            });
                            void queryClient.invalidateQueries({
                                queryKey: queryKeys.textResearch.assistantScope(corpusId),
                            });
                        }}
                    />
                    {membershipChanged ? (
                        <Alert severity="warning">
                            Current indexed document membership differs from this answer&apos;s
                            evidence snapshot. The historical answer remains scoped to its
                            original document membership.
                        </Alert>
                    ) : null}
                    {evidenceRevisionChanged ? (
                        <Alert severity="warning">
                            Indexed evidence changed since this answer. Its historical evidence
                            revision is preserved and it may not represent the current corpus.
                        </Alert>
                    ) : null}
                    {synthesisProvenance ? (
                        <Alert severity="info">
                            Auditable synthesis · evidence revision{" "}
                            {synthesisProvenance.evidence_revision_hash?.slice(0, 12) ?? "—"}
                            {" · "}
                            {synthesisProvenance.documents_considered?.length ?? 0} documents
                            considered
                            {(synthesisProvenance.documents_omitted?.length ?? 0) > 0
                                ? ` · ${synthesisProvenance.documents_omitted?.length} omitted`
                                : ""}
                            . Per-document retrieval traces and structured map findings are
                            preserved in run provenance.
                        </Alert>
                    ) : null}

                    <ToggleButtonGroup
                        exclusive
                        size="small"
                        fullWidth
                        value={researcherMode}
                        onChange={(_, value: ResearcherMode | null) => {
                            if (value) setResearcherMode(value);
                        }}
                        aria-label="Researcher mode"
                    >
                        {(Object.keys(RESEARCHER_MODE_LABEL) as ResearcherMode[]).map((key) => (
                            <ToggleButton key={key} value={key}>
                                {RESEARCHER_MODE_LABEL[key]}
                            </ToggleButton>
                        ))}
                    </ToggleButtonGroup>

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
                                    if (intent === "synthesis") {
                                        setResearcherMode("synthesis");
                                    }
                                    submitQuestion(label, intent);
                                }}
                                disabled={isBusy || scope?.indexed_count === 0}
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
                                !question.trim() || isBusy || (scope?.indexed_count ?? 0) === 0
                            }
                            onClick={() => submitQuestion(question)}
                        >
                            {researcherMode === "synthesis" ? "Synthesize" : "Ask"}
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

                    {askMutation.isPending || synthesizeMutation.isPending ? (
                        <Stack direction="row" spacing={1} alignItems="center">
                            <CircularProgress size={18} />
                            <Typography variant="body2">
                                {researcherMode === "synthesis" || synthesizeMutation.isPending
                                    ? "Running corpus synthesis…"
                                    : assistantStage === "generation_started"
                                      ? "Analyzing retrieved evidence…"
                                      : assistantStage === "citation_validation_complete"
                                        ? "Validating citations…"
                                        : "Retrieving evidence…"}
                            </Typography>
                        </Stack>
                    ) : null}
                    {synthesisRunId ? (
                        <Stack direction="row" spacing={1} alignItems="center">
                            <CircularProgress size={18} />
                            <Typography variant="body2">
                                Synthesis run {synthesisRunId.slice(0, 8)}…
                                {synthesisRunQuery.data?.progress_stage
                                    ? ` ${synthesisRunQuery.data.progress_stage}`
                                    : ""}
                            </Typography>
                        </Stack>
                    ) : null}
                    {askMutation.isError ? (
                        <Alert severity="error">
                            {getQueryErrorMessage(askMutation.error, "Ask Corpus failed.")}
                        </Alert>
                    ) : null}
                    {synthesizeMutation.isError ? (
                        <Alert severity="error">
                            {getQueryErrorMessage(
                                synthesizeMutation.error,
                                "Corpus synthesis failed."
                            )}
                        </Alert>
                    ) : null}
                    {synthesisError ? (
                        <Alert severity="error" onClose={() => setSynthesisError(null)}>
                            {synthesisError}
                        </Alert>
                    ) : null}

                    {(threadsQuery.data?.length ?? 0) > 0 ? (
                        <Box>
                            <Typography variant="caption" color="text.secondary">
                                Recent threads
                            </Typography>
                            <Stack spacing={0.5} mt={0.5}>
                                {threadsQuery.data?.slice(0, threadDisplayLimit).map((thread) => (
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
                                {(threadsQuery.data?.length ?? 0) > threadDisplayLimit ? (
                                    <Button
                                        size="small"
                                        onClick={() => setThreadDisplayLimit((limit) => limit + 10)}
                                    >
                                        Load more conversations
                                    </Button>
                                ) : null}
                            </Stack>
                        </Box>
                    ) : null}

                    {conversationMessages.length > 0 ? (
                        <Stack spacing={1.25}>
                            <Typography variant="subtitle2">Thread</Typography>
                            {conversationMessages.map((msg, index) => {
                                const role = msg.role ?? "unknown";
                                const isAssistant = role === "assistant";
                                const msgCitations = Array.isArray(msg.citations)
                                    ? msg.citations
                                    : [];
                                const msgClaims = Array.isArray(msg.claims) ? msg.claims : [];
                                const meta = msg.metadata ?? {};
                                const turnFailed = meta.turn_status === "failed";
                                const turnCompleted = meta.turn_status === "completed";
                                const retryable = meta.retryable === true;
                                const retryQuery = String(
                                    meta.original_query ?? meta.query ?? ""
                                ).trim();
                                const status = meta.citation_validation_status as
                                    | string
                                    | undefined;
                                const noContextFound =
                                    Boolean(meta.no_context_found) || status === "no_evidence";
                                const citationValidationFailed =
                                    Boolean(meta.citation_validation_failed) ||
                                    (status != null &&
                                        status !== "valid" &&
                                        status !== "no_evidence") ||
                                    (!msgClaims.length && !noContextFound);
                                const statusWarning = citationValidationWarning(status);
                                return (
                                    <Box
                                        key={msg.id ?? `${role}-${index}`}
                                        sx={{
                                            borderLeft: 2,
                                            borderColor: isAssistant
                                                ? "primary.main"
                                                : "divider",
                                            pl: 1.25,
                                        }}
                                    >
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            display="block"
                                        >
                                            {isAssistant ? "Assistant" : "You"}
                                        </Typography>
                                        {isAssistant ? (
                                            turnFailed ? (
                                                <Alert
                                                    severity="error"
                                                    action={
                                                        retryable && retryQuery ? (
                                                            <Button
                                                                color="inherit"
                                                                size="small"
                                                                disabled={isBusy}
                                                                onClick={() => submitQuestion(retryQuery)}
                                                            >
                                                                Retry
                                                            </Button>
                                                        ) : undefined
                                                    }
                                                >
                                                    {String(
                                                        meta.safe_error_summary ??
                                                            "The assistant could not complete this request."
                                                    )}
                                                    {meta.retrieval_completed === true
                                                        ? " Evidence retrieval completed before generation failed."
                                                        : " Evidence retrieval did not complete."}
                                                </Alert>
                                            ) : (
                                            <>
                                                {statusWarning ? (
                                                    <Alert severity="warning" sx={{ mb: 1 }}>
                                                        {statusWarning}
                                                    </Alert>
                                                ) : null}
                                                <ClaimsWithCitations
                                                    claims={msgClaims}
                                                    citations={msgCitations}
                                                    fallbackAnswer={msg.content ?? ""}
                                                    citationValidationFailed={
                                                        citationValidationFailed
                                                    }
                                                    noContextFound={noContextFound}
                                                    onOpenCitation={onOpenCitation}
                                                />
                                                {msgCitations.filter((c) => c.used_in_answer)
                                                    .length > 0 ? (
                                                    <Stack spacing={0.5} mt={1}>
                                                        <Typography variant="caption">
                                                            Sources
                                                        </Typography>
                                                        {msgCitations
                                                            .filter((c) => c.used_in_answer)
                                                            .map((citation) => (
                                                                <Link
                                                                    key={citation.chunk_id}
                                                                    component="button"
                                                                    type="button"
                                                                    variant="caption"
                                                                    onClick={() =>
                                                                        onOpenCitation?.(citation)
                                                                    }
                                                                >
                                                                    [
                                                                    {citation.citation_number}]{" "}
                                                                    {citation.filename}
                                                                    {citation.page_number != null
                                                                        ? ` · p.${citation.page_number}`
                                                                        : ""}
                                                                </Link>
                                                            ))}
                                                    </Stack>
                                                ) : null}
                                                {turnCompleted && msg.id ? (
                                                    <Stack spacing={0.5} mt={1}>
                                                        <Stack direction="row" spacing={0.75}>
                                                            <Button
                                                            size="small"
                                                            variant="outlined"
                                                            onClick={() =>
                                                                savePersistedMessageMemoMutation.mutate({
                                                                    id: msg.id!,
                                                                    body: msg.content ?? "",
                                                                })
                                                            }
                                                            disabled={
                                                                savePersistedMessageMemoMutation.isPending
                                                            }
                                                        >
                                                            Save as memo
                                                            </Button>
                                                            <Button
                                                            size="small"
                                                            onClick={() =>
                                                                void copyPersistedAnswer(
                                                                    msg.id!,
                                                                    msg.content ?? ""
                                                                )
                                                            }
                                                        >
                                                            {copiedMessageId === msg.id ? "Copied" : "Copy"}
                                                            </Button>
                                                            <Button
                                                            size="small"
                                                            onClick={() => {
                                                                if (conversationQuery.data?.thread) {
                                                                    setLastResult(
                                                                        messageToResult(
                                                                            msg,
                                                                            conversationQuery.data.thread,
                                                                            conversationQuery.data.scope
                                                                        )
                                                                    );
                                                                }
                                                                setMode("evidence");
                                                            }}
                                                        >
                                                            Show evidence
                                                            </Button>
                                                        </Stack>
                                                        {memoSavedMessageId === msg.id ? (
                                                            <Typography variant="caption" color="success.main">
                                                                Research memo saved.
                                                            </Typography>
                                                        ) : null}
                                                        {savePersistedMessageMemoMutation.isError ? (
                                                            <Typography variant="caption" color="error.main">
                                                                Could not save memo: {getQueryErrorMessage(savePersistedMessageMemoMutation.error)}
                                                            </Typography>
                                                        ) : null}
                                                    </Stack>
                                                ) : null}
                                            </>
                                            )
                                        ) : (
                                            <Typography
                                                variant="body2"
                                                sx={{ whiteSpace: "pre-wrap" }}
                                            >
                                                {msg.content}
                                            </Typography>
                                        )}
                                    </Box>
                                );
                            })}
                        </Stack>
                    ) : lastResult ? (
                        <Stack spacing={1}>
                            <EvidenceProvenance
                                scope={answerScope ?? emptyScope()}
                                coverage={lastResult.coverage}
                                evidenceRevisionHash={lastResult.evidence_revision_hash}
                                degraded={lastResult.retrieval_degraded}
                                degradationReason={lastResult.degradation_reason}
                                noResults={lastResult.no_context_found}
                                citations={lastResult.citations}
                                synthesisOmissions={lastResult.synthesis_provenance?.documents_omitted}
                            />
                            {lastResult.no_context_found ? (
                                <Alert severity="info">
                                    No relevant evidence found in this corpus.
                                </Alert>
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
                                    A fully cited answer could not be generated. Retrieved
                                    evidence is shown separately below.
                                </Alert>
                            ) : null}
                            {validationWarning ? (
                                <Alert severity="warning">{validationWarning}</Alert>
                            ) : null}
                            {lastResult.coverage.documents_in_scope > 0 ? (
                                <Typography variant="caption" color="text.secondary">
                                    Evidence retrieved from{" "}
                                    {lastResult.coverage.documents_with_retrieved_evidence} of{" "}
                                    {lastResult.coverage.documents_in_scope} documents
                                </Typography>
                            ) : null}
                            <ClaimsWithCitations
                                claims={lastResult.claims}
                                citations={lastResult.citations}
                                fallbackAnswer={lastResult.answer}
                                citationValidationFailed={
                                    lastResult.citation_validation_failed ||
                                    (!lastResult.claims.length && !lastResult.no_context_found)
                                }
                                noContextFound={lastResult.no_context_found}
                                onOpenCitation={onOpenCitation}
                            />
                            <Button
                                size="small"
                                variant="outlined"
                                onClick={() => saveMemoMutation.mutate()}
                                disabled={saveMemoMutation.isPending}
                            >
                                {saveMemoMutation.isPending ? "Saving…" : "Save as research memo"}
                            </Button>
                            {saveMemoMutation.isError ? (
                                <Alert severity="error">
                                    Could not save memo: {getQueryErrorMessage(saveMemoMutation.error)}
                                </Alert>
                            ) : null}
                            {saveMemoMutation.isSuccess ? (
                                <Alert severity="success">Research memo saved.</Alert>
                            ) : null}
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
                                        <Typography
                                            variant="caption"
                                            display="block"
                                            color="text.secondary"
                                        >
                                            {citation.snippet}
                                        </Typography>
                                    </Box>
                                ))}
                        </Stack>
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
                            {validationWarning ? (
                                <Alert severity="warning">{validationWarning}</Alert>
                            ) : null}
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
                                        {citation.used_in_answer
                                            ? "Used in answer"
                                            : "Retrieved only"}
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
