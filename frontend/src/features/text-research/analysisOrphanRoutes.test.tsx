import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { AnalysisSubRedirect } from "../../app/router";

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
    getAnnotationProgress,
    getCorpusMetadataFacets,
    getRun,
    getRunResultArtifact,
    getRunResults,
    getSourceText,
    getTextUnitContext,
    listDocuments,
    listRuns,
    listUncertainPredictions,
    runDictionaryAnalysis,
    runFrequencies,
    runKwic,
    runNgrams,
} from "../../api/textResearch";
import { resolveAsyncRunControls } from "./analysisAsync";

function lastCall(): [string, RequestInit | undefined] {
    expect(fetchMock).toHaveBeenCalled();
    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit | undefined];
    return [String(path), init];
}

function LocationProbe() {
    const location = useLocation();
    return (
        <div
            data-testid="location"
            data-pathname={location.pathname}
            data-search={location.search}
        />
    );
}

describe("analysis API helper contracts", () => {
    beforeEach(() => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({ id: "run-1", status: "queued" });
    });

    it("posts statistical-model with typed body defaults", async () => {
        const run = await fitStatisticalModel("corpus-1", {
            model: "ols",
            dependent_var: "y",
            independent_vars: ["x1"],
            rows: [{ y: 1, x1: 2 }],
        });
        expect(run).toEqual({ id: "run-1", status: "queued" });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/statistical-model");
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body).toMatchObject({
            model: "ols",
            add_intercept: true,
            dependent_var: "y",
            independent_vars: ["x1"],
        });
        expect(body.rows).toEqual([{ y: 1, x1: 2 }]);
    });

    it("posts measurement-comparison with categorical default", async () => {
        await compareMeasurements("corpus-1", {
            source_a: "human",
            source_b: "model",
            values_a: ["a"],
            values_b: ["b"],
            ids: ["1"],
            value_kind: "categorical",
        });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/measurement-comparison");
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body.value_kind).toBe("categorical");
        expect(body.source_a).toBe("human");
        expect(body.values_b).toEqual(["b"]);
    });

    it("posts frequencies with corpus language filter and run_async", async () => {
        const asyncPayload = resolveAsyncRunControls(
            { operations: { frequencies: { async: true, default_execution_mode: "auto" } } },
            "frequencies",
            true
        ).payload;
        await runFrequencies("corpus-1", {
            unit_type: "paragraph",
            language: "en",
            top_n: 25,
            ...asyncPayload,
        });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/frequencies");
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body).toMatchObject({
            unit_type: "paragraph",
            language: "en",
            top_n: 25,
            run_async: true,
        });
    });

    it("posts ngrams with defaults and optional async flag", async () => {
        await runNgrams("corpus-1", {
            unit_type: "sentence",
            n: 3,
            top_n: 10,
            ...{ run_async: false },
        });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/ngrams");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body).toMatchObject({ unit_type: "sentence", n: 3, top_n: 10, run_async: false });
    });

    it("posts kwic with query_language distinct from corpus language filter", async () => {
        await runKwic("corpus-1", {
            unit_type: "paragraph",
            keyword: "democracy",
            language: "de",
            query_mode: "lemma",
            query_language: "en",
            window_size: 7,
        });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/kwic");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body.language).toBe("de");
        expect(body.query_language).toBe("en");
        expect(body.query_mode).toBe("lemma");
        expect(body.keyword).toBe("democracy");
        expect(body.window_size).toBe(7);
        expect(body.case_sensitive).toBe(false);
    });

    it("cancels runs via POST /runs/{id}/cancel", async () => {
        await cancelRun("run-42");
        const [path, init] = lastCall();
        expect(path).toContain("/runs/run-42/cancel");
        expect(init?.method).toBe("POST");
    });

    it("rejects a dictionary analysis with no source before fetching", async () => {
        await expect(
            runDictionaryAnalysis("corpus-1", { unit_type: "paragraph" })
        ).rejects.toThrow("dictionary_id, dictionary_terms, or hierarchy is required");
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it("posts dictionary analysis when terms are provided", async () => {
        await runDictionaryAnalysis("corpus-1", {
            unit_type: "paragraph",
            dictionary_terms: ["freedom", "rights"],
        });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/analysis/dictionary");
        const body = JSON.parse(String(init?.body ?? "{}"));
        expect(body.dictionary_terms).toEqual(["freedom", "rights"]);
    });

    it("lists corpus documents with language query param", async () => {
        fetchMock.mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
        await listDocuments("corpus-1", { language: "fr", limit: 20 });
        const [path, init] = lastCall();
        expect(path).toContain("/corpora/corpus-1/documents?");
        expect(path).toContain("language=fr");
        expect(path).toContain("limit=20");
        expect(init?.method).toBeUndefined();
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
        const [path] = lastCall();
        expect(path).toContain("/runs/run-1");
    });

    it("gets artifactized run results with an AbortSignal", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({
            artifact_id: "art-1",
            checksum: "abc",
            payload: { summary: true },
        });
        const artifact = await getRunResultArtifact("run-1", controller.signal);
        expect(artifact).toMatchObject({ artifact_id: "art-1", checksum: "abc" });
        const [path, init] = lastCall();
        expect(path).toContain("/runs/run-1/results-artifact");
        expect(init?.signal).toBe(controller.signal);
    });

    it("pages authorized run results with key/offset and AbortSignal", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({
            artifact_id: "export:abc",
            checksum: "abc",
            items: [{ id: 1 }],
            total: 250,
            row_count: 250,
            limit: 100,
            offset: 100,
            artifactized: true,
        });
        const page = await getRunResults(
            "run-1",
            { key: "pairs", limit: 100, offset: 100 },
            controller.signal
        );
        expect(page.artifactized).toBe(true);
        expect(page.items).toEqual([{ id: 1 }]);
        const [path, init] = lastCall();
        expect(path).toContain("/runs/run-1/results?");
        expect(path).toContain("key=pairs");
        expect(path).toContain("limit=100");
        expect(path).toContain("offset=100");
        expect(init?.signal).toBe(controller.signal);
    });

    it("gets analysis capabilities and maps async support for UI payload", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({
            operations: {
                frequencies: { async: true, default_execution_mode: "auto" },
                kwic: { async: false, default_execution_mode: "inline" },
            },
            run_types: {},
        });
        const capabilities = await getAnalysisCapabilities(controller.signal);
        const [path, init] = lastCall();
        expect(path).toBe("/research/analysis-capabilities");
        expect(init?.signal).toBe(controller.signal);
        expect(capabilities.operations.kwic.async).toBe(false);

        const freq = resolveAsyncRunControls(capabilities, "frequencies", true);
        expect(freq.supportsAsync).toBe(true);
        expect(freq.payload).toEqual({ run_async: true });
        const kwic = resolveAsyncRunControls(capabilities, "kwic", true);
        expect(kwic.showInlineOnly).toBe(true);
        expect(kwic.payload).toEqual({});
    });

    it("lists runs with limit/offset query params and AbortSignal", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({ items: [], total: 0, limit: 50, offset: 50 });
        const page = await listRuns(
            "proj-1",
            { corpus_id: "c1", limit: 50, offset: 50 },
            controller.signal
        );
        expect(page).toEqual({ items: [], total: 0, limit: 50, offset: 50 });
        const [path, init] = lastCall();
        expect(path).toContain("/projects/proj-1/runs?");
        expect(path).toContain("limit=50");
        expect(path).toContain("offset=50");
        expect(path).toContain("corpus_id=c1");
        expect(init?.signal).toBe(controller.signal);
    });
});

describe("high-volume GET helpers forward AbortSignal", () => {
    beforeEach(() => {
        fetchMock.mockReset();
        fetchMock.mockResolvedValue({});
    });

    it("forwards signal for uncertain predictions, source text, unit context, progress, facets", async () => {
        const controller = new AbortController();
        fetchMock.mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });

        await listUncertainPredictions("model-1", { limit: 20, offset: 0 }, controller.signal);
        expect(fetchMock.mock.calls.at(-1)?.[1]?.signal).toBe(controller.signal);
        expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain("/active-learning/queue?");

        await getSourceText("doc-1", controller.signal);
        expect(fetchMock.mock.calls.at(-1)?.[1]?.signal).toBe(controller.signal);
        expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain("/documents/doc-1/source-text");

        await getTextUnitContext("unit-1", 2, controller.signal);
        expect(fetchMock.mock.calls.at(-1)?.[1]?.signal).toBe(controller.signal);
        expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain("/text-units/unit-1/context?");

        await getAnnotationProgress("corpus-1", controller.signal);
        expect(fetchMock.mock.calls.at(-1)?.[1]?.signal).toBe(controller.signal);
        expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain(
            "/corpora/corpus-1/annotations/progress"
        );

        await getCorpusMetadataFacets("corpus-1", controller.signal);
        expect(fetchMock.mock.calls.at(-1)?.[1]?.signal).toBe(controller.signal);
        expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain("/corpora/corpus-1/facets");
    });
});

describe("analysis deep-link redirects under Analysis", () => {
    it.each([
        ["statistical", "/research/p1/analysis/statistical"],
        ["measurement", "/research/p1/analysis/measurement"],
    ] as const)("redirects %s alias to analysis?tab=", async (tab, entry) => {
        render(
            <MemoryRouter initialEntries={[entry]}>
                <Routes>
                    <Route
                        path={`/research/:projectId/analysis/${tab}`}
                        element={<AnalysisSubRedirect tab={tab} />}
                    />
                    <Route path="/research/:projectId/analysis" element={<LocationProbe />} />
                </Routes>
            </MemoryRouter>
        );
        const loc = await screen.findByTestId("location");
        expect(loc).toHaveAttribute("data-pathname", "/research/p1/analysis");
        expect(loc).toHaveAttribute("data-search", `?tab=${tab}`);
        expect(entry.includes("/explorer")).toBe(false);
    });
});
