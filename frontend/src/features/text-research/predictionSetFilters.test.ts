import { describe, expect, it } from "vitest";

import {
    PREDICTION_LAYER_LABELS,
    buildPredictionRows,
    filterPredictionRows,
    layersRemainSeparate,
} from "./predictionSetFilters";

describe("predictionSetFilters", () => {
    const baseRows = buildPredictionRows({
        predictions: [
            {
                id: "p1",
                trained_model_id: "m1",
                text_unit_id: "u1",
                predicted_labels: ["lib"],
                scores: { lib: 0.9, cons: 0.1 },
                uncertainty: 0.2,
                created_at: "2026-01-01T00:00:00Z",
            },
            {
                id: "p2",
                trained_model_id: "m1",
                text_unit_id: "u2",
                predicted_labels: ["cons"],
                scores: { cons: 0.55 },
                uncertainty: 0.7,
                created_at: "2026-01-01T00:00:00Z",
            },
        ],
        annotations: [
            {
                id: "a1",
                text_unit_id: "u1",
                label_id: "l1",
                annotator_id: "ann1",
                value: "lib",
                confidence: null,
                comment: null,
                codebook_version: "1",
                created_at: "2026-01-01T00:00:00Z",
                updated_at: "2026-01-01T00:00:00Z",
            },
            {
                id: "a2",
                text_unit_id: "u1",
                label_id: "l1",
                annotator_id: "ann2",
                value: "cons",
                confidence: null,
                comment: null,
                codebook_version: "1",
                created_at: "2026-01-01T00:00:00Z",
                updated_at: "2026-01-01T00:00:00Z",
            },
        ],
        adjudications: [
            {
                id: "g1",
                text_unit_id: "u2",
                label_id: "l1",
                codebook_version: "1",
                final_value: "lib",
                adjudicator_id: "adj1",
                comment: null,
                created_at: "2026-01-01T00:00:00Z",
            },
        ],
    });

    it("keeps model / human / gold layers distinct on each row", () => {
        expect(PREDICTION_LAYER_LABELS.model).toBe("MODEL PREDICTION");
        const u1 = baseRows.find((row) => row.text_unit_id === "u1")!;
        const u2 = baseRows.find((row) => row.text_unit_id === "u2")!;
        expect(u1.predicted_labels).toEqual(["lib"]);
        expect(u1.human_values).toEqual(["cons", "lib"]);
        expect(u1.gold_values).toEqual([]);
        expect(u1.human_disagreement).toBe(true);
        expect(u1.review_status).toBe("annotated");
        expect(u2.review_status).toBe("adjudicated");
        expect(u2.model_vs_gold_disagreement).toBe(true);
        expect(u2.gold_values).toEqual(["lib"]);
        expect(u2.predicted_labels).toEqual(["cons"]);
    });

    it("filters by label, confidence, uncertainty, review, disagreement", () => {
        expect(
            filterPredictionRows(baseRows, {
                predictedLabel: "lib",
                minConfidence: "",
                maxUncertainty: "",
                reviewStatus: "",
                humanDisagreementOnly: false,
            }).map((r) => r.id)
        ).toEqual(["p1"]);

        expect(
            filterPredictionRows(baseRows, {
                predictedLabel: "",
                minConfidence: "0.8",
                maxUncertainty: "",
                reviewStatus: "",
                humanDisagreementOnly: false,
            }).map((r) => r.id)
        ).toEqual(["p1"]);

        expect(
            filterPredictionRows(baseRows, {
                predictedLabel: "",
                minConfidence: "",
                maxUncertainty: "0.5",
                reviewStatus: "",
                humanDisagreementOnly: false,
            }).map((r) => r.id)
        ).toEqual(["p1"]);

        expect(
            filterPredictionRows(baseRows, {
                predictedLabel: "",
                minConfidence: "",
                maxUncertainty: "",
                reviewStatus: "adjudicated",
                humanDisagreementOnly: false,
            }).map((r) => r.id)
        ).toEqual(["p2"]);

        expect(
            filterPredictionRows(baseRows, {
                predictedLabel: "",
                minConfidence: "",
                maxUncertainty: "",
                reviewStatus: "",
                humanDisagreementOnly: true,
            }).map((r) => r.id)
        ).toEqual(["p1"]);
    });

    it("does not silently convert predictions into gold", () => {
        const layers = layersRemainSeparate(["lib"], ["cons"], []);
        expect(layers.gold).toEqual([]);
        expect(layers.model).toEqual(["lib"]);
        expect(layers.human).toEqual(["cons"]);
    });
});
