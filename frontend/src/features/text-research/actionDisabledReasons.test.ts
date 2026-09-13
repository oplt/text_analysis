import { describe, expect, it } from "vitest";

import {
    RELIABILITY_ANNOTATOR_HINT,
    agentRunDisabledReason,
    analysisRunDisabledReason,
    assignUncertainDisabledReason,
    comparativeRunDisabledReason,
    computeReliabilityDisabledReason,
    driftCompareDisabledReason,
    modelLifecycleDisabledReason,
    predictModelDisabledReason,
    robustnessSweepDisabledReason,
    segmentCorpusDisabledReason,
    trainClassifierDisabledReason,
    trainTopicModelDisabledReason,
} from "./actionDisabledReasons";

describe("actionDisabledReasons", () => {
    it("explains missing training snapshot", () => {
        expect(trainClassifierDisabledReason({})).toMatch(/training dataset snapshot/i);
        expect(trainClassifierDisabledReason({ snapshotId: "snap-1" })).toBeNull();
    });

    it("explains reliability prerequisites", () => {
        expect(computeReliabilityDisabledReason({})).toMatch(/corpus/i);
        expect(
            computeReliabilityDisabledReason({ corpusId: "c1", codebookId: "cb1", labelCount: 0 })
        ).toMatch(/label/i);
        expect(RELIABILITY_ANNOTATOR_HINT).toMatch(/two completed annotators/i);
    });

    it("explains analysis and KWIC requirements", () => {
        expect(analysisRunDisabledReason({})).toMatch(/corpus/i);
        expect(
            analysisRunDisabledReason({
                corpusId: "c1",
                tab: "kwic",
                kwicKeyword: "",
            })
        ).toMatch(/KWIC/i);
        expect(
            analysisRunDisabledReason({
                corpusId: "c1",
                tab: "keyness",
                keynessA: "a",
                keynessB: "",
            })
        ).toMatch(/keyness/i);
    });

    it("covers primary AI and model actions", () => {
        expect(agentRunDisabledReason({ message: "  " })).toMatch(/task message/i);
        expect(predictModelDisabledReason({})).toMatch(/trained model/i);
        expect(trainTopicModelDisabledReason({})).toMatch(/corpus/i);
        expect(robustnessSweepDisabledReason({})).toMatch(/snapshot/i);
        expect(comparativeRunDisabledReason({ corpusId: "c1" })).toMatch(/codebook/i);
        expect(segmentCorpusDisabledReason({ documentCount: 0 })).toMatch(/documents/i);
        expect(
            assignUncertainDisabledReason({
                modelId: "m1",
                selectedUnitCount: 0,
                annotatorCount: 1,
            })
        ).toMatch(/uncertain units/i);
        expect(
            modelLifecycleDisabledReason({
                action: "production",
                currentStatus: "production",
            })
        ).toMatch(/already in production/i);
        expect(driftCompareDisabledReason({ baselineId: "a" })).toMatch(/prediction sets/i);
    });
});
