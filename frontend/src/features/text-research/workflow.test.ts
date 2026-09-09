import { describe, expect, it } from "vitest";
import { deriveWorkflowStages } from "./workflow";
import type { DashboardSummary } from "./types";

function summary(partial: Partial<DashboardSummary>): DashboardSummary {
    return {
        corpus: { id: "c1", name: "Demo" },
        document_count: 0,
        text_unit_counts: {},
        codebook_count: 0,
        training_dataset_snapshot_count: 0,
        trained_model_count: 0,
        latest_model: null,
        analysis_run_counts_by_type: {},
        analysis_run_counts_by_status: {},
        latest_reliability: null,
        annotation_task_count: 0,
        annotation_completed_count: 0,
        annotation_completion_rate: 0,
        ...partial,
    };
}

describe("deriveWorkflowStages", () => {
    it("marks early stages blocked without corpus documents", () => {
        const stages = deriveWorkflowStages({
            activeRoute: "dashboard",
            hasCorpus: false,
            hasCodebook: false,
            labelCount: 0,
            unitType: "paragraph",
            summary: null,
        });

        expect(stages.find((s) => s.id === "corpus")?.status).toBe("blocked");
        expect(stages.find((s) => s.id === "prepare")?.status).toBe("blocked");
        expect(stages.find((s) => s.id === "annotate")?.status).toBe("blocked");
    });

    it("shows document and unit counts when prepare is complete", () => {
        const stages = deriveWorkflowStages({
            activeRoute: "prepare",
            hasCorpus: true,
            hasCodebook: true,
            labelCount: 2,
            unitType: "paragraph",
            summary: summary({
                document_count: 42,
                text_unit_counts: { paragraph: 4812 },
            }),
        });

        expect(stages.find((s) => s.id === "corpus")).toMatchObject({
            status: "complete",
            detail: "42 documents",
        });
        expect(stages.find((s) => s.id === "prepare")).toMatchObject({
            status: "current",
            detail: "4,812 paragraph units",
        });
    });

    it("surfaces annotation progress and kappa when available", () => {
        const stages = deriveWorkflowStages({
            activeRoute: "reliability",
            hasCorpus: true,
            hasCodebook: true,
            labelCount: 3,
            unitType: "paragraph",
            summary: summary({
                document_count: 10,
                text_unit_counts: { paragraph: 100 },
                annotation_task_count: 300,
                annotation_completed_count: 128,
                annotation_completion_rate: 128 / 300,
                latest_reliability: {
                    run_id: "r1",
                    metrics: { mean_cohens_kappa: 0.74 },
                },
            }),
        });

        expect(stages.find((s) => s.id === "annotate")).toMatchObject({
            status: "warning",
            detail: "128/300 completed",
        });
        expect(stages.find((s) => s.id === "reliability")).toMatchObject({
            status: "current",
            detail: "κ 0.74",
        });
    });

    it("reports trained model counts for classify", () => {
        const stages = deriveWorkflowStages({
            activeRoute: "classification",
            hasCorpus: true,
            hasCodebook: true,
            labelCount: 2,
            unitType: "paragraph",
            summary: summary({
                document_count: 4,
                text_unit_counts: { paragraph: 20 },
                annotation_completed_count: 10,
                annotation_task_count: 10,
                annotation_completion_rate: 1,
                trained_model_count: 2,
            }),
        });

        expect(stages.find((s) => s.id === "classify")).toMatchObject({
            status: "current",
            detail: "2 trained models",
        });
    });

    it("places codebook, contextual analysis, and export in the workflow", () => {
        const stages = deriveWorkflowStages({
            activeRoute: "codebook",
            hasCorpus: true,
            hasCodebook: true,
            codebook: { name: "Values", version: "2", is_frozen: true },
            labelCount: 3,
            unitType: "paragraph",
            summary: summary({ document_count: 2, text_unit_counts: { paragraph: 8 } }),
        });

        expect(stages.map((stage) => stage.id)).toEqual([
            "corpus", "prepare", "codebook", "annotate", "reliability", "analyze",
            "topics", "classify", "models", "predictions", "drift", "validate", "explore", "contextual", "export",
        ]);
        expect(stages.find((stage) => stage.id === "codebook")).toMatchObject({
            status: "current",
            detail: "Values · v2 · frozen · 3 labels",
        });
        expect(stages.find((stage) => stage.id === "export")?.route).toBe("exports");
    });
});
