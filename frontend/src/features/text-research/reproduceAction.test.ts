import { describe, expect, it } from "vitest";
import { reproduceActionState } from "./reproduceAction";

describe("reproduceActionState", () => {
    it("enables Replay when the API marks the run rerunnable", () => {
        expect(
            reproduceActionState({
                rerunnable: true,
                rerun_block_reason: null,
            }),
        ).toEqual({
            replay: { enabled: true, reason: null },
            exact: {
                enabled: false,
                reason: "Exact reproduce requires frozen inputs and checksums.",
            },
        });
    });

    it("disables Replay with the API block reason", () => {
        expect(
            reproduceActionState({
                rerunnable: false,
                rerun_block_reason:
                    "Clustering runs are not rerunnable: exact reproduction is not registered.",
            }),
        ).toEqual({
            replay: {
                enabled: false,
                reason:
                    "Clustering runs are not rerunnable: exact reproduction is not registered.",
            },
            exact: {
                enabled: false,
                reason:
                    "Clustering runs are not rerunnable: exact reproduction is not registered.",
            },
        });
    });

    it("falls back to a generic reason when block reason is missing", () => {
        expect(
            reproduceActionState({
                rerunnable: false,
                rerun_block_reason: null,
            }),
        ).toEqual({
            replay: { enabled: false, reason: "Replay is not available for this run." },
            exact: {
                enabled: false,
                reason: "Exact reproduce is not available for this run.",
            },
        });
    });

    it("exposes Exact reproduce only for frozen runs", () => {
        expect(
            reproduceActionState({
                replayable: true,
                exact_reproducible: true,
                rerunnable: true,
            }),
        ).toEqual({
            replay: { enabled: true, reason: null },
            exact: { enabled: true, reason: null },
        });
    });
});
