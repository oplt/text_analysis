import { describe, expect, it } from "vitest";
import {
    formatAgreementWithInterpretation,
    interpretChanceCorrectedAgreement,
    summarizeReliabilityCoverage,
} from "./reliabilityInterpretation";

describe("interpretChanceCorrectedAgreement", () => {
    it("returns null for missing values", () => {
        expect(interpretChanceCorrectedAgreement(null)).toBeNull();
        expect(interpretChanceCorrectedAgreement(Number.NaN)).toBeNull();
    });

    it("maps illustrative bands without claiming universal truth", () => {
        expect(interpretChanceCorrectedAgreement(0.82)?.label).toBe("Almost perfect agreement");
        expect(interpretChanceCorrectedAgreement(0.65)?.label).toBe("Substantial agreement");
        expect(interpretChanceCorrectedAgreement(0.45)?.label).toBe("Moderate agreement");
        expect(interpretChanceCorrectedAgreement(-0.1)?.label).toBe("Below chance");
        expect(interpretChanceCorrectedAgreement(0.82)?.caveat).toMatch(/Illustrative heuristic/);
        expect(interpretChanceCorrectedAgreement(0.82)?.caveat).toMatch(/discipline/);
    });

    it("formats value text beside interpretation", () => {
        expect(formatAgreementWithInterpretation(0.821).valueText).toBe("0.82");
        expect(formatAgreementWithInterpretation(null).valueText).toBe("—");
    });
});

describe("summarizeReliabilityCoverage", () => {
    it("aggregates coder and pairable unit coverage", () => {
        expect(
            summarizeReliabilityCoverage({
                a: { n_coders: 2, n_pairable_units: 40 },
                b: { n_coders: 3, n_pairable_units: 25 },
            })
        ).toEqual({
            maxCoders: 3,
            totalPairableUnits: 65,
            labelsWithCoders: 2,
        });
    });
});
