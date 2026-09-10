import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    getAgentRun,
    listAgentRuns,
    type AgentRun,
} from "../../api/agent";
import { getAiOverview } from "../../api/ai";
import { listProjects } from "../../api/projects";
import { listRagDocuments } from "../../api/rag";
import AgentView from "./AgentView";

vi.mock("../../app/snackbarContext", () => ({
    useSnackbar: () => ({ showToast: vi.fn() }),
}));
vi.mock("../../api/agent", () => ({
    createAgentRun: vi.fn(),
    getAgentRun: vi.fn(),
    listAgentRuns: vi.fn(),
    formatAgentCostMicros: () => "$0.00",
}));
vi.mock("../../api/ai", () => ({ getAiOverview: vi.fn() }));
vi.mock("../../api/projects", () => ({ listProjects: vi.fn() }));
vi.mock("../../api/rag", () => ({ listRagDocuments: vi.fn() }));
vi.mock("../../components/layout/SettingsTabs", () => ({ SettingsTabs: () => null }));

const run: AgentRun = {
    id: "run-12345678",
    prompt_template_id: null,
    prompt_version_id: null,
    agent_id: "default",
    agent_run_id: "memory-123",
    provider_key: "local",
    model_name: "local-heuristic",
    status: "completed",
    response_format: "text",
    variables: {},
    retrieval_query: null,
    retrieved_chunk_ids: [],
    retrieval_degraded: false,
    memory_degraded: false,
    degradation_reason: null,
    injection_chunks_filtered: 0,
    input_messages: [],
    output_text: "Persisted response",
    output_json: null,
    latency_ms: 12,
    input_tokens: 3,
    output_tokens: 5,
    total_tokens: 8,
    estimated_cost_micros: 0,
    error_message: null,
    review_status: "not_requested",
    created_at: "2026-09-10T00:00:00Z",
    completed_at: "2026-09-10T00:00:01Z",
    memory_run_id: "memory-123",
};

function renderView() {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
        <QueryClientProvider client={queryClient}>
            <AgentView />
        </QueryClientProvider>
    );
}

describe("AgentView persisted history", () => {
    beforeEach(() => {
        vi.mocked(getAiOverview).mockResolvedValue({
            providers: [],
            prompt_templates: [],
            recent_runs: [],
            documents: [],
            datasets: [],
        });
        vi.mocked(listProjects).mockResolvedValue([]);
        vi.mocked(listRagDocuments).mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
        vi.mocked(listAgentRuns).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
        vi.mocked(getAgentRun).mockResolvedValue(run);
    });

    afterEach(() => vi.restoreAllMocks());

    it("renders a loading placeholder before persisted history resolves", () => {
        vi.mocked(listAgentRuns).mockReturnValue(new Promise(() => undefined));

        const { container } = renderView();

        expect(container.querySelector(".MuiSkeleton-root")).toBeInTheDocument();
    });

    it("renders an empty persisted-history state", async () => {
        renderView();

        expect(await screen.findByText("No agent runs yet")).toBeInTheDocument();
    });

    it("renders a recoverable history error", async () => {
        vi.mocked(listAgentRuns).mockRejectedValue(new Error("History unavailable"));
        renderView();

        expect(await screen.findByText("History unavailable")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    });

    it("loads a selected persisted run and refreshes history", async () => {
        vi.mocked(listAgentRuns).mockResolvedValue({ items: [run], total: 1, limit: 20, offset: 0 });
        renderView();

        fireEvent.click(await screen.findByText(/run-1234.*completed/i));

        expect(await screen.findByText("Persisted response")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Refresh history" }));
        await waitFor(() => expect(listAgentRuns).toHaveBeenCalledTimes(2));
    });
});
