import { describe, expect, it } from "vitest";
import {
    formatPercent,
    formatRunType,
    languageSummary,
    latestRunStatusLabel,
    pickReliabilityHighlights,
    totalTextUnits,
    workspaceHrefForRunType,
} from "./dashboardSummary";
import type { DashboardSummary } from "./types";

function baseSummary(overrides: Partial<DashboardSummary> = {}): DashboardSummary {
    return {
        corpus: { id: "c1", name: "Demo" },
        document_count: 12,
        text_unit_counts: { document: 0, paragraph: 40, sentence: 120 },
        codebook_count: 2,
        training_dataset_snapshot_count: 1,
        trained_model_count: 1,
        latest_model: null,
        analysis_run_counts_by_type: {},
        analysis_run_counts_by_status: { completed: 3 },
        latest_reliability: null,
        annotation_task_count: 10,
        annotation_completed_count: 4,
        annotation_completion_rate: 0.4,
        ...overrides,
    };
}

describe("dashboardSummary helpers", () => {
    it("sums text units across types", () => {
        expect(totalTextUnits(baseSummary())).toBe(160);
    });

    it("prefers the most recent run for latest status", () => {
        const result = latestRunStatusLabel(
            baseSummary({
                recent_runs: [
                    {
                        id: "run-12345678",
                        run_type: "frequency_analysis",
                        status: "failed",
                        created_at: "2026-01-01T00:00:00Z",
                    },
                ],
            })
        );
        expect(result.value).toBe("failed");
        expect(result.color).toBe("error");
        expect(result.description).toContain("frequency analysis");
    });

    it("formats languages and percent coverage", () => {
        expect(languageSummary({ en: 8, tr: 3, de: 1 })).toBe("en (8), tr (3), de (1)");
        expect(formatPercent(0.835)).toBe("84%");
        expect(formatRunType("topic_model")).toBe("topic model");
    });

    it("extracts readable reliability highlights", () => {
        expect(
            pickReliabilityHighlights({
                mean_krippendorff_alpha: 0.81,
                mean_cohens_kappa: 0.77,
                nested: { ignored: true },
            })
        ).toEqual([
            { label: "Mean Krippendorff α", value: "0.81" },
            { label: "Mean Cohen's κ", value: "0.77" },
        ]);
    });

    it("routes recent activity into the right workspace", () => {
        expect(workspaceHrefForRunType("p1", "segmentation")).toBe("/research/p1/prepare");
        expect(workspaceHrefForRunType("p1", "train_classifier")).toBe(
            "/research/p1/classification"
        );
        expect(workspaceHrefForRunType("p1", "export_dataset")).toBe("/research/p1/exports");
        expect(workspaceHrefForRunType("p1", "kwic")).toBe("/research/p1/runs");
    });
});
