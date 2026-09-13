import { describe, expect, it } from "vitest";
import type { AgentRun } from "../../api/agent";
import { buildAgentTraceSteps, formatElapsed } from "./agentTraceModel";

const baseRun: AgentRun = {
    id: "run-1",
    prompt_template_id: null,
    prompt_version_id: null,
    agent_id: "default",
    agent_run_id: "mem-1",
    provider_key: "local",
    model_name: "local-heuristic",
    status: "completed",
    response_format: "text",
    variables: {},
    retrieval_query: "find protocols",
    retrieved_chunk_ids: ["c1", "c2"],
    retrieval_degraded: false,
    memory_degraded: false,
    degradation_reason: null,
    injection_chunks_filtered: 1,
    input_messages: [{ role: "user", content: "Summarize the protocols" }],
    output_text: "Here is a summary.",
    output_json: null,
    latency_ms: 1200,
    input_tokens: 10,
    output_tokens: 20,
    total_tokens: 30,
    estimated_cost_micros: 0,
    error_message: null,
    review_status: "not_requested",
    created_at: "2026-09-10T00:00:00Z",
    completed_at: "2026-09-10T00:00:01Z",
    memory_run_id: "mem-1",
};

describe("buildAgentTraceSteps", () => {
    it("builds structured tool/result steps without dumping raw logs", () => {
        const steps = buildAgentTraceSteps(baseRun);
        expect(steps.map((s) => s.category)).toContain("tool");
        expect(steps.map((s) => s.category)).toContain("result");
        expect(steps.some((s) => s.title.includes("RAG retrieve"))).toBe(true);
        expect(steps.some((s) => s.title.includes("Generate"))).toBe(true);
        expect(steps.find((s) => s.id === "retrieve-result")?.detail).toMatch(/filtered 1/);
    });

    it("surfaces errors as dedicated steps", () => {
        const steps = buildAgentTraceSteps({
            ...baseRun,
            error_message: "Provider timeout",
            output_text: null,
        });
        expect(steps.some((s) => s.category === "error")).toBe(true);
    });
});

describe("formatElapsed", () => {
    it("formats ms and seconds", () => {
        expect(formatElapsed(250)).toBe("250 ms");
        expect(formatElapsed(1200)).toBe("1.2 s");
    });
});
