import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getRunResultProvenance } from "./resultProvenance";
import { ResearchResultsTable } from "./ResearchResults";

describe("ResearchResultsTable", () => {
    it("sorts and paginates analysis rows", () => {
        render(
            <ResearchResultsTable
                pageSize={1}
                rows={[
                    { id: "a", term: "zebra", count: 1 },
                    { id: "b", term: "apple", count: 2 },
                ]}
                columns={[
                    { id: "term", label: "Term", value: (row) => row.term },
                    { id: "count", label: "Count", value: (row) => row.count },
                ]}
            />
        );

        expect(screen.getByText("zebra")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Term" }));
        expect(screen.getByText("apple")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Next" }));
        expect(screen.getByText("zebra")).toBeInTheDocument();
    });
});

describe("getRunResultProvenance", () => {
    it("reads runtime, identity, and artifacts from the canonical R result envelope", () => {
        const provenance = getRunResultProvenance({
            id: "run-1",
            project_id: "project-1",
            corpus_id: "corpus-1",
            run_type: "frequency_analysis",
            status: "completed",
            progress_stage: null,
            parameters: null,
            metrics: null,
            results: {
                analysis_result: {
                    runtime: { engine: "r", implementation: "quanteda" },
                    identity: { spec_hash: "spec-123" },
                    artifacts: [{ name: "result.json" }],
                },
            },
            artifact_path: null,
            random_seed: null,
            created_by: "user-1",
            started_at: null,
            completed_at: null,
            error_message: null,
            created_at: "2026-01-01T00:00:00Z",
        });

        expect(provenance.runtime).toMatchObject({ engine: "r", implementation: "quanteda" });
        expect(provenance.identity).toMatchObject({ spec_hash: "spec-123" });
        expect(provenance.artifacts).toEqual([{ name: "result.json" }]);
    });
});
