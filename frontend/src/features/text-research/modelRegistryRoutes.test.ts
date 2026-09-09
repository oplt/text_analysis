import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Phase 16: model registry / prediction / drift routes must remain mounted.
 */
describe("model registry frontend routes", () => {
    it("registers models, predictions, and drift under research project routes", () => {
        const routerSource = readFileSync(
            resolve(__dirname, "../../app/router.tsx"),
            "utf8"
        );
        expect(routerSource).toContain('path="models"');
        expect(routerSource).toContain("ModelRegistryView");
        expect(routerSource).toContain('path="predictions"');
        expect(routerSource).toContain("PredictionSetsView");
        expect(routerSource).toContain('path="drift"');
        expect(routerSource).toContain("DriftMonitoringView");
    });

    it("workflow stages expose models, predictions, and drift", async () => {
        const { deriveWorkflowStages } = await import("./workflow");
        const stages = deriveWorkflowStages({
            activeRoute: "models",
            hasCorpus: true,
            hasCodebook: true,
            labelCount: 1,
            unitType: "paragraph",
            summary: {
                corpus: { id: "c1", name: "Demo" },
                document_count: 1,
                text_unit_counts: { paragraph: 1 },
                codebook_count: 1,
                training_dataset_snapshot_count: 0,
                trained_model_count: 1,
                latest_model: null,
                analysis_run_counts_by_type: {},
                analysis_run_counts_by_status: {},
                latest_reliability: null,
                annotation_task_count: 0,
                annotation_completed_count: 0,
                annotation_completion_rate: 0,
            },
        });
        const ids = stages.map((stage) => stage.id);
        expect(ids).toContain("models");
        expect(ids).toContain("predictions");
        expect(ids).toContain("drift");
        expect(stages.find((stage) => stage.id === "models")?.route).toBe("models");
        expect(stages.find((stage) => stage.id === "predictions")?.route).toBe("predictions");
        expect(stages.find((stage) => stage.id === "drift")?.route).toBe("drift");
    });
});
