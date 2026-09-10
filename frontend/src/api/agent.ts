import { apiFetch } from "./client";

const BASE = "/agent";

export type AgentRun = {
    id: string;
    prompt_template_id: string | null;
    prompt_version_id: string | null;
    provider_key: string;
    model_name: string;
    status: string;
    response_format: string;
    variables: Record<string, unknown>;
    retrieval_query: string | null;
    retrieved_chunk_ids: string[];
    retrieval_degraded: boolean;
    memory_degraded: boolean;
    degradation_reason: string | null;
    injection_chunks_filtered: number;
    input_messages: Array<{ role: string; content: string }>;
    output_text: string | null;
    output_json: Record<string, unknown> | null;
    latency_ms: number | null;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    estimated_cost_micros: number;
    error_message: string | null;
    review_status: string;
    created_at: string;
    completed_at: string | null;
    memory_run_id: string;
};

export type AgentRunRequest = {
    agent_id?: string;
    run_id?: string;
    project_id?: string;
    user_message?: string;
    prompt_template_key?: string;
    prompt_version_id?: string;
    variables?: Record<string, unknown>;
    retrieval_query?: string;
    document_ids?: string[];
    top_k?: number;
    review_required?: boolean;
};

/** Sync agent execution (201). Backend times out long runs; UI shows loading/error. */
export async function createAgentRun(payload: AgentRunRequest): Promise<AgentRun> {
    return apiFetch(`${BASE}/runs`, {
        method: "POST",
        body: JSON.stringify({
            agent_id: "default",
            top_k: 4,
            review_required: false,
            ...payload,
        }),
    });
}

export function formatAgentCostMicros(micros: number): string {
    if (!Number.isFinite(micros) || micros <= 0) return "$0.00";
    return `$${(micros / 1_000_000).toFixed(4)}`;
}
