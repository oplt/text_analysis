import { describe, expect, it } from "vitest";
import { reproduceActionState } from "./reproduceAction";

describe("reproduceActionState", () => {
    it("enables Reproduce when the API marks the run rerunnable", () => {
        expect(
            reproduceActionState({
                rerunnable: true,
                rerun_block_reason: null,
            }),
        ).toEqual({ enabled: true, reason: null });
    });

    it("disables Reproduce with the API block reason", () => {
        expect(
            reproduceActionState({
                rerunnable: false,
                rerun_block_reason:
                    "Clustering runs are not rerunnable: exact reproduction is not registered.",
            }),
        ).toEqual({
            enabled: false,
            reason:
                "Clustering runs are not rerunnable: exact reproduction is not registered.",
        });
    });

    it("falls back to a generic reason when block reason is missing", () => {
        expect(
            reproduceActionState({
                rerunnable: false,
                rerun_block_reason: null,
            }),
        ).toEqual({
            enabled: false,
            reason: "Reproduce is not available for this run.",
        });
    });
});
