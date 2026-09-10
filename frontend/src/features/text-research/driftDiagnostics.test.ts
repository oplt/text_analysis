import { describe, expect, it } from "vitest";

import {
    DRIFT_DISCLAIMER,
    parseDriftReport,
    topTermsFromCoefficients,
    withPerformanceSection,
} from "./driftDiagnostics";

describe("driftDiagnostics", () => {
    it("parses prediction, prevalence, score, and feature sections into table rows", () => {
        const { rows, sectionCount, warningLevel, provenance } = parseDriftReport({
            analysis_run_id: "run-1",
            mode: "MODEL_COMPARISON",
            summary: { section_count: 3, warning_level: "investigate", n_observations: 40 },
            provenance: {
                baseline_prediction_set_id: "ps-a",
                current_prediction_set_id: "ps-b",
                n_baseline: 20,
                n_current: 20,
                aggregation: "full_prediction_set",
            },
            baseline: { label_counts: { a: 10, b: 10 }, n: 20 },
            current: { label_counts: { a: 18, b: 2 }, n: 20 },
            sections: {
                prediction_distribution: {
                    total_variation_distance: 0.3,
                    psi_like: 0.4,
                },
                score_distribution: {
                    method: "ks_2samp",
                    statistic: 0.15,
                    baseline_n: 20,
                    current_n: 20,
                },
                uncertainty_distribution: {
                    method: "ks_2samp",
                    statistic: 0.2,
                    baseline_n: 20,
                    current_n: 20,
                },
                feature_presence: {
                    jaccard_similarity: 0.4,
                    baseline_terms: ["x", "y"],
                    current_terms: ["y", "z"],
                },
            },
        });

        expect(sectionCount).toBe(3);
        expect(warningLevel).toBe("investigate");
        expect(provenance.baselinePredictionSetId).toBe("ps-a");
        expect(provenance.nObservations).toBe(40);
        expect(rows.some((r) => r.kind === "prediction_distribution")).toBe(true);
        expect(rows.some((r) => r.kind === "class_prevalence")).toBe(true);
        expect(rows.some((r) => r.kind === "score_distribution")).toBe(true);
        expect(rows.some((r) => r.kind === "uncertainty_distribution")).toBe(true);
        expect(rows.some((r) => r.kind === "feature_input")).toBe(true);
        expect(rows.find((r) => r.id === "pred-tvd")?.status).toBe("investigate");
        expect(rows.find((r) => r.id === "feature-jaccard")?.status).toBe("investigate");
        expect(DRIFT_DISCLAIMER.toLowerCase()).toContain("not proof of model degradation");
    });

    it("attaches labeled performance without claiming distribution equals degradation", () => {
        const report = withPerformanceSection(
            { sections: {}, summary: { section_count: 0 } },
            {
                baseline_macro_f1: 0.84,
                current_macro_f1: 0.7,
                source: "holdout metrics",
            }
        );
        const { rows } = parseDriftReport(report);
        const perf = rows.find((r) => r.kind === "performance");
        expect(perf?.status).toBe("investigate");
        expect(perf?.note.toLowerCase()).toContain("distribution drift");
    });

    it("extracts unique top terms from coefficients by absolute weight", () => {
        expect(
            topTermsFromCoefficients(
                [
                    { feature: "alpha", coefficient: 0.1 },
                    { feature: "beta", coefficient: -2 },
                    { feature: "alpha", coefficient: 0.5 },
                    { feature: "gamma", coefficient: 0.2 },
                ],
                2
            )
        ).toEqual(["beta", "alpha"]);
    });
});
