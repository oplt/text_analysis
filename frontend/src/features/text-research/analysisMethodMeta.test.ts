import { describe, expect, it } from "vitest";
import { analysisMethodMeta } from "./analysisMethodMeta";

describe("analysisMethodMeta", () => {
    it("provides laboratory notes for core workspaces", () => {
        expect(analysisMethodMeta("overview").label).toMatch(/Corpus overview/i);
        expect(analysisMethodMeta("frequencies").chart?.title).toBeTruthy();
        expect(analysisMethodMeta("keyness").methodNote).toMatch(/not causal/i);
        expect(analysisMethodMeta("frequencies").chart?.methodologicalNote).toBeTruthy();
    });

    it("keeps chart chrome fields for plotted methods", () => {
        const chart = analysisMethodMeta("frequencies").chart!;
        expect(chart.xAxisLabel).toBeTruthy();
        expect(chart.yAxisLabel).toBeTruthy();
        expect(chart.seriesLabel).toBeTruthy();
    });
});
