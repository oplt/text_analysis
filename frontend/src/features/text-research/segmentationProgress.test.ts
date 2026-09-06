import { describe, expect, it } from "vitest";
import {
    isSegmentationRunActive,
    segmentationProgressLines,
    segmentationStageLabel,
} from "./segmentationProgress";
import type { AnalysisRun } from "./types";

function run(partial: Partial<AnalysisRun>): AnalysisRun {
    return {
        id: "r1",
        project_id: "p1",
        corpus_id: "c1",
        run_type: "segmentation",
        status: "running",
        progress_stage: "segmenting_documents",
        parameters: { unit_type: "paragraph", document_count: 187 },
        metrics: {
            unit_type: "paragraph",
            documents_total: 187,
            documents_segmented: 42,
            text_units_created: 4812,
        },
        results: null,
        artifact_path: null,
        random_seed: null,
        created_by: "u1",
        started_at: null,
        completed_at: null,
        error_message: null,
        created_at: new Date().toISOString(),
        ...partial,
    };
}

describe("segmentationProgress", () => {
    it("labels preparation stages for the workspace", () => {
        expect(segmentationStageLabel("preparing")).toBe("Preparing corpus");
        expect(segmentationStageLabel("segmenting_documents")).toBe("Segmenting documents");
    });

    it("formats document and unit progress lines", () => {
        expect(segmentationProgressLines(run({}))).toEqual([
            "Segmenting documents",
            "42 / 187 documents",
            "4,812 paragraph units created",
        ]);
    });

    it("treats queued and running as active", () => {
        expect(isSegmentationRunActive(run({ status: "queued" }))).toBe(true);
        expect(isSegmentationRunActive(run({ status: "completed" }))).toBe(false);
    });
});
