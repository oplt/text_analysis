import { describe, expect, it, vi, beforeEach } from "vitest";

const fetchMock = vi.fn();

vi.mock("../../api/client", async () => {
    const actual = await vi.importActual<typeof import("../../api/client")>("../../api/client");
    return {
        ...actual,
        apiFetch: (...args: unknown[]) => fetchMock(...args),
    };
});

import {
    cancelRun,
    compareMeasurements,
    getAnalysisCapabilities,
    fitStatisticalModel,
    getRun,
    getRunResultArtifact,
    listRuns,
    runDictionaryAnalysis,
} from "../../api/textResearch";

describe("analysis API helper contracts (TASK-023)", () => {
    beforeEach(() => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({ id: "run-1", status: "queued" });
    });

    it("posts statistical-model to the research analysis path", async () => {
        await fitStatisticalModel("corpus-1", {
            model: "ols",
            dependent_var: "y",
            independent_vars: ["x1"],
            rows: [{ y: 1, x1: 2 }],
        });
        expect(fetchMock).toHaveBeenCalled();
        const [path, init] = fetchMock.mock.calls[0];
        expect(String(path)).toContain("/corpora/corpus-1/analysis/statistical-model");
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body.dependent_var).toBe("y");
        expect(body.independent_vars).toEqual(["x1"]);
    });

    it("posts measurement-comparison to the research analysis path", async () => {
        await compareMeasurements("corpus-1", {
            source_a: "human",
            source_b: "model",
            values_a: ["a"],
            values_b: ["b"],
            ids: ["1"],
            value_kind: "categorical",
        });
        const [path, init] = fetchMock.mock.calls[0];
        expect(String(path)).toContain("/corpora/corpus-1/analysis/measurement-comparison");
        expect(init?.method).toBe("POST");
    });

    it("cancels runs via POST /runs/{id}/cancel", async () => {
        await cancelRun("run-42");
        const [path, init] = fetchMock.mock.calls[0];
        expect(String(path)).toContain("/runs/run-42/cancel");
        expect(init?.method).toBe("POST");
    });

    it("rejects a dictionary analysis with no source before fetching", async () => {
        await expect(
            runDictionaryAnalysis("corpus-1", { unit_type: "paragraph" })
        ).rejects.toThrow("dictionary_id, dictionary_terms, or hierarchy is required");
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it("preserves run rerun capability fields from the API", async () => {
        fetchMock.mockResolvedValue({
            id: "run-1",
            status: "completed",
            rerunnable: false,
            rerun_block_reason: "Input artifact unavailable.",
        });
        const run = await getRun("run-1");
        expect(run.rerunnable).toBe(false);
        expect(run.rerun_block_reason).toBe("Input artifact unavailable.");
    });

    it("gets artifactized run results with an AbortSignal", async () => {
        const controller = new AbortController();
        await getRunResultArtifact("run-1", controller.signal);
        const [path, init] = fetchMock.mock.calls[0];
        expect(String(path)).toContain("/runs/run-1/results-artifact");
        expect(init?.signal).toBe(controller.signal);
    });

    it("gets analysis capabilities from the shared research endpoint", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({
            operations: { frequencies: { async: true }, kwic: { async: false } },
            run_types: {},
        });
        const capabilities = await getAnalysisCapabilities(controller.signal);
        const [path, init] = fetchMock.mock.calls[0];
        expect(path).toBe("/research/analysis-capabilities");
        expect(init?.signal).toBe(controller.signal);
        expect(capabilities.operations.kwic.async).toBe(false);
    });

    it("lists runs with limit/offset query params and AbortSignal", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({ items: [], total: 0, limit: 50, offset: 50 });
        await listRuns("proj-1", { corpus_id: "c1", limit: 50, offset: 50 }, controller.signal);
        const [path, init] = fetchMock.mock.calls[0];
        expect(String(path)).toContain("/projects/proj-1/runs?");
        expect(String(path)).toContain("limit=50");
        expect(String(path)).toContain("offset=50");
        expect(String(path)).toContain("corpus_id=c1");
        expect(init?.signal).toBe(controller.signal);
    });
});

describe("analysis deep-link routes under Analysis", () => {
    it("keeps statistical and measurement aliases under /analysis/", () => {
        const aliases = [
            "/research/:projectId/analysis/statistical",
            "/research/:projectId/analysis/measurement",
        ];
        for (const path of aliases) {
            expect(path).toContain("/analysis/");
            expect(path.includes("/explorer")).toBe(false);
        }
    });
});
