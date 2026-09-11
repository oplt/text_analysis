import { describe, expect, it } from "vitest";
import {
    canonicalAnalysisResults,
    rEngineOptionLabel,
    rEngineSelectionState,
} from "./analysisEngine";

describe("R analysis engine selector", () => {
    const rEngine = {
        name: "r" as const,
        implementation: "quanteda",
        available: true,
        ready: true,
        analyses: ["frequencies", "dfm", "kwic", "dictionary", "keyness", "cooccurrence"],
    };

    it("keeps R disabled while capabilities are loading or unavailable", () => {
        expect(rEngineSelectionState(undefined, "frequencies")).toBe("checking");
        expect(rEngineOptionLabel("checking")).toContain("checking availability");
        expect(rEngineSelectionState([], "frequencies")).toBe("unavailable");
        expect(rEngineOptionLabel("unavailable")).toContain("unavailable");
        expect(rEngineSelectionState([{ ...rEngine, available: false }], "frequencies")).toBe("unavailable");
    });

    it("keeps configured R disabled until a worker is ready", () => {
        expect(rEngineSelectionState([{ ...rEngine, ready: false }], "frequencies")).toBe("not_ready");
        expect(rEngineOptionLabel("not_ready")).toContain("worker offline");
    });

    it("enables supported analyses and disables unsupported ones", () => {
        expect(rEngineSelectionState([rEngine], "dictionary")).toBe("available");
        expect(rEngineSelectionState([rEngine], "dictionaries")).toBe("available");
        expect(rEngineSelectionState([rEngine], "ngrams")).toBe("unsupported");
        expect(rEngineOptionLabel("unsupported")).toContain("not supported");
    });
});

describe("canonicalAnalysisResults", () => {
    it("uses the canonical R payload while retaining ordinary persisted results", () => {
        expect(canonicalAnalysisResults({
            cooccurrence: [],
            analysis_result: { results: { pairs: [{ term_a: "a", term_b: "b" }] } },
        })).toEqual({ pairs: [{ term_a: "a", term_b: "b" }] });
        expect(canonicalAnalysisResults({ frequencies: [{ term: "analysis", count: 1 }] })).toEqual({
            frequencies: [{ term: "analysis", count: 1 }],
        });
    });
});
