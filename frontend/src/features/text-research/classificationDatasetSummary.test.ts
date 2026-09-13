import { describe, expect, it } from "vitest";
import {
    assessClassImbalance,
    plannedSplitFractions,
} from "./classificationDatasetSummary";

describe("assessClassImbalance", () => {
    it("returns no warnings for balanced classes", () => {
        const result = assessClassImbalance([
            { label: "a", value: 40 },
            { label: "b", value: 35 },
            { label: "c", value: 30 },
        ]);
        expect(result.warnings).toEqual([]);
        expect(result.imbalanceRatio).toBeCloseTo(40 / 30, 5);
    });

    it("warns when majority share is very high", () => {
        const result = assessClassImbalance([
            { label: "yes", value: 90 },
            { label: "no", value: 10 },
        ]);
        expect(result.majorityShare).toBeCloseTo(0.9);
        expect(result.warnings[0]).toMatch(/Class imbalance/i);
        expect(result.warnings[0]).toMatch(/macro F1/i);
    });

    it("warns on high majority/minority ratio", () => {
        const result = assessClassImbalance([
            { label: "a", value: 50 },
            { label: "b", value: 40 },
            { label: "c", value: 8 },
        ]);
        expect(result.imbalanceRatio).toBeGreaterThanOrEqual(5);
        expect(result.warnings.length).toBeGreaterThan(0);
    });
});

describe("plannedSplitFractions", () => {
    it("computes train remainder after test and validation", () => {
        expect(plannedSplitFractions(0.25, 0.2)).toEqual({
            testFraction: 0.25,
            validationFraction: 0.2,
            trainFraction: 0.55,
        });
    });
});
