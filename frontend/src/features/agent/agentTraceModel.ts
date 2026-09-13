/**
 * Structured agent trace steps derived from persisted AgentRun fields (Phase 15).
 * Avoid dumping raw logs; synthesize readable thought/action/result steps.
 */

import type { AgentRun } from "../../api/agent";

export type AgentTraceCategory = "plan" | "tool" | "result" | "error" | "meta";

export type AgentTraceStep = {
    id: string;
    category: AgentTraceCategory;
    title: string;
    detail?: string;
    durationMs?: number | null;
    status?: "ok" | "warn" | "error";
};

export function buildAgentTraceSteps(run: AgentRun): AgentTraceStep[] {
    const steps: AgentTraceStep[] = [];

    const userMessage = run.input_messages.find((m) => m.role === "user")?.content;
    if (userMessage) {
        steps.push({
            id: "input",
            category: "plan",
            title: "Task received",
            detail: userMessage.length > 240 ? `${userMessage.slice(0, 240)}…` : userMessage,
            status: "ok",
        });
    }

    if (run.retrieval_query) {
        steps.push({
            id: "retrieve",
            category: "tool",
            title: "Tool: RAG retrieve",
            detail: `Query: ${run.retrieval_query}`,
            status: run.retrieval_degraded ? "warn" : "ok",
        });
        steps.push({
            id: "retrieve-result",
            category: "result",
            title: "Retrieval result",
            detail:
                run.retrieved_chunk_ids.length > 0
                    ? `${run.retrieved_chunk_ids.length} chunk(s)` +
                      (run.injection_chunks_filtered
                          ? ` · filtered ${run.injection_chunks_filtered} injection chunk(s)`
                          : "")
                    : "No chunks retrieved",
            status: run.retrieved_chunk_ids.length ? "ok" : "warn",
        });
    } else if (run.retrieved_chunk_ids.length > 0) {
        steps.push({
            id: "retrieve-implicit",
            category: "tool",
            title: "Tool: RAG retrieve",
            detail: `${run.retrieved_chunk_ids.length} chunk(s) attached`,
            status: run.retrieval_degraded ? "warn" : "ok",
        });
    }

    steps.push({
        id: "generate",
        category: "tool",
        title: "Tool: Generate",
        detail: `${run.provider_key} / ${run.model_name}`,
        status: run.error_message ? "error" : "ok",
    });

    if (run.memory_run_id) {
        steps.push({
            id: "memory",
            category: "meta",
            title: "Memory linkage",
            detail: run.memory_run_id,
            status: run.memory_degraded ? "warn" : "ok",
        });
    }

    if (run.error_message) {
        steps.push({
            id: "error",
            category: "error",
            title: "Error",
            detail: run.error_message,
            status: "error",
        });
    } else {
        steps.push({
            id: "output",
            category: "result",
            title: "Final output produced",
            detail: run.output_text
                ? `${run.output_text.length} characters`
                : run.output_json
                  ? "JSON artifact"
                  : "Empty output",
            durationMs: run.latency_ms,
            status: "ok",
        });
    }

    steps.push({
        id: "usage",
        category: "meta",
        title: "Usage",
        detail: `Tokens ${run.total_tokens} (in ${run.input_tokens} / out ${run.output_tokens}) · latency ${
            run.latency_ms ?? "—"
        } ms`,
        durationMs: run.latency_ms,
        status: "ok",
    });

    return steps;
}

export function agentRunElapsedMs(run: AgentRun): number | null {
    if (run.latency_ms != null) return run.latency_ms;
    if (!run.created_at || !run.completed_at) return null;
    const start = Date.parse(run.created_at);
    const end = Date.parse(run.completed_at);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
    return end - start;
}

export function formatElapsed(ms: number): string {
    if (ms < 1000) return `${ms} ms`;
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    const minutes = Math.floor(seconds / 60);
    const rem = Math.round(seconds % 60);
    return `${minutes}m ${rem}s`;
}
