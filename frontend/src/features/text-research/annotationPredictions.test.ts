import { describe, expect, it } from "vitest";

import {
    isBlindReliabilityCoding,
    shouldFetchAdjudications,
    shouldFetchPeerAnnotations,
    shouldFetchPredictions,
} from "./annotationPredictions";

describe("shouldFetchPredictions", () => {
    it("returns false without a model", () => {
        expect(
            shouldFetchPredictions({
                selectedModelId: null,
                blindPolicy: {
                    hide_model_predictions: false,
                    ai_assistance_enabled: true,
                    blind_mode: false,
                },
            })
        ).toBe(false);
    });

    it("returns false in blind reliability mode", () => {
        expect(
            shouldFetchPredictions({
                selectedModelId: "model-1",
                blindPolicy: {
                    blind_mode: true,
                    hide_model_predictions: true,
                    ai_assistance_enabled: false,
                },
            })
        ).toBe(false);
    });

    it("returns true in AI-assisted mode", () => {
        expect(
            shouldFetchPredictions({
                selectedModelId: "model-1",
                campaign: {
                    id: "c1",
                    name: "Campaign",
                    annotation_mode: "ai_assisted",
                    blind_mode: false,
                    ai_assistance_enabled: true,
                },
            })
        ).toBe(true);
    });

    it("prefers blind_policy over campaign summary", () => {
        expect(
            shouldFetchPredictions({
                selectedModelId: "model-1",
                blindPolicy: {
                    blind_mode: true,
                    hide_model_predictions: true,
                    ai_assistance_enabled: false,
                },
                campaign: {
                    id: "c1",
                    name: "Campaign",
                    annotation_mode: "ai_assisted",
                    blind_mode: false,
                    ai_assistance_enabled: true,
                },
            })
        ).toBe(false);
    });
});

describe("shouldFetchPeerAnnotations", () => {
    it("returns false when incomplete blind coding hides peers", () => {
        expect(
            shouldFetchPeerAnnotations({
                blindPolicy: {
                    blind_mode: true,
                    hide_model_predictions: true,
                    hide_peer_annotations: true,
                    hide_adjudications: true,
                    ai_assistance_enabled: false,
                },
            })
        ).toBe(false);
    });

    it("returns true after task complete even in blind campaign", () => {
        expect(
            shouldFetchPeerAnnotations({
                blindPolicy: {
                    blind_mode: true,
                    hide_model_predictions: true,
                    hide_peer_annotations: false,
                    hide_adjudications: false,
                    ai_assistance_enabled: false,
                },
            })
        ).toBe(true);
    });
});

describe("shouldFetchAdjudications", () => {
    it("returns false when blind policy hides gold labels", () => {
        expect(
            shouldFetchAdjudications({
                blindPolicy: {
                    blind_mode: true,
                    hide_model_predictions: true,
                    hide_peer_annotations: true,
                    hide_adjudications: true,
                    ai_assistance_enabled: false,
                },
            })
        ).toBe(false);
    });

    it("defaults to false for blind campaign without explicit policy flags", () => {
        expect(
            shouldFetchAdjudications({
                campaign: {
                    id: "c1",
                    name: "Campaign",
                    annotation_mode: "blind_reliability",
                    blind_mode: true,
                    ai_assistance_enabled: false,
                },
            })
        ).toBe(false);
    });
});

describe("isBlindReliabilityCoding", () => {
    it("detects blind mode from policy or campaign", () => {
        expect(
            isBlindReliabilityCoding(
                { blind_mode: true, hide_model_predictions: true, ai_assistance_enabled: false },
                null
            )
        ).toBe(true);
        expect(
            isBlindReliabilityCoding(null, {
                id: "c1",
                name: "Campaign",
                annotation_mode: "blind_reliability",
                blind_mode: true,
                ai_assistance_enabled: false,
            })
        ).toBe(true);
        expect(isBlindReliabilityCoding(null, null)).toBe(false);
    });
});
