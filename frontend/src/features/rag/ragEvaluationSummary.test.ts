import { describe, expect, it } from "vitest";
import {
    documentPipelineLabel,
    formatCoverage,
    indexingParamsFromChunkMeta,
    summarizeRagAsk,
} from "./ragEvaluationSummary";

describe("summarizeRagAsk", () => {
    it("computes citation coverage from retrieved vs cited chunks", () => {
        const summary = summarizeRagAsk({
            citations: [
                { document_id: "d1", chunk_id: "c1", score: 0.9 },
                { document_id: "d1", chunk_id: "c2", score: 0.7 },
            ],
            retrieved_chunk_ids: ["c1", "c2", "c3", "c4"],
            no_context_found: false,
            retrieval_degraded: false,
            memory_degraded: false,
            injection_chunks_filtered: 0,
        });
        expect(summary.retrievedCount).toBe(4);
        expect(summary.citationCount).toBe(2);
        expect(summary.citationCoverage).toBeCloseTo(0.5);
        expect(summary.distinctDocuments).toBe(1);
        expect(formatCoverage(summary.citationCoverage)).toBe("50%");
    });

    it("flags missing grounding", () => {
        const summary = summarizeRagAsk({
            citations: [],
            retrieved_chunk_ids: ["c1"],
            no_context_found: false,
            retrieval_degraded: true,
            memory_degraded: false,
            injection_chunks_filtered: 1,
        });
        expect(summary.groundingNotes.join(" ")).toMatch(/degraded/i);
        expect(summary.groundingNotes.join(" ")).toMatch(/without citations/i);
    });
});

describe("indexingParamsFromChunkMeta", () => {
    it("extracts embedding and chunk settings when present", () => {
        const rows = indexingParamsFromChunkMeta({
            embedding_model: "text-embedding-3-small",
            chunk_size: 1000,
            ignored: { nested: true },
        });
        expect(rows.map((r) => r.key)).toEqual(["embedding_model", "chunk_size"]);
    });
});

describe("documentPipelineLabel", () => {
    it("maps indexed status to complete parsing and indexed state", () => {
        expect(documentPipelineLabel("indexed")).toEqual({
            parsing: "Complete",
            indexing: "Indexed",
        });
    });
});
