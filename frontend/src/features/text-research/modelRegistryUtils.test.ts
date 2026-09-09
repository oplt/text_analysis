import { describe, expect, it } from "vitest";

import {
    aggregatePredictionsForDrift,
    extractMacroF1,
    lifecycleDisplayLabel,
    modelDisplayName,
    normalizePredictionRow,
} from "./modelRegistryUtils";

describe("modelRegistryUtils", () => {
    it("maps lifecycle statuses to registry labels", () => {
        expect(lifecycleDisplayLabel("approved")).toBe("Production");
        expect(lifecycleDisplayLabel("deprecated")).toBe("Deprecated");
        expect(lifecycleDisplayLabel("candidate")).toBe("Candidate");
    });

    it("extracts macro F1 from common metric shapes", () => {
        expect(extractMacroF1({ f1_macro: 0.84 })).toBe(0.84);
        expect(extractMacroF1({ holdout: { f1_macro: 0.71 } })).toBe(0.71);
        expect(extractMacroF1({})).toBeNull();
    });

    it("builds display names from name or id+version", () => {
        expect(modelDisplayName({ id: "abcdefghij", name: "liberalism-v7", version: 7 })).toBe(
            "liberalism-v7"
        );
        expect(modelDisplayName({ id: "abcdefghij", name: null, version: 3 })).toBe("abcdefgh-v3");
    });

    it("normalizes ORM-style prediction rows and aggregates drift payloads", () => {
        const row = normalizePredictionRow({
            predicted_labels_json: '["lib"]',
            scores_json: '{"lib": 0.9}',
            uncertainty: 0.1,
        });
        expect(row).toEqual({
            predicted_labels: ["lib"],
            scores: { lib: 0.9 },
            uncertainty: 0.1,
        });

        const agg = aggregatePredictionsForDrift([
            { predicted_labels: ["lib"], scores: { lib: 0.9 }, uncertainty: 0.1 },
            { predicted_labels: [], scores: {}, uncertainty: null },
        ]);
        expect(agg.label_counts).toEqual({ lib: 1, __unlabeled__: 1 });
        expect(agg.scores).toEqual([0.9, 0.1]);
    });
});
