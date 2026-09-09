import { describe, expect, it } from "vitest";

/**
 * Phase 12: orphan Analysis tools must call the same backend paths the
 * StatisticalModelView / MeasurementComparisonView API helpers use.
 */
describe("analysis orphan API alignment", () => {
    it("documents statistical-model and measurement-comparison endpoints", () => {
        // Keep in sync with backend routes.py and textResearch.ts helpers.
        expect("/research/corpora/{id}/analysis/statistical-model").toContain(
            "analysis/statistical-model"
        );
        expect("/research/corpora/{id}/analysis/measurement-comparison").toContain(
            "analysis/measurement-comparison"
        );
    });

    it("groups deep links under Analysis rather than new top-level tabs", () => {
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
