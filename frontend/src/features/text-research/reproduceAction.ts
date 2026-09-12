import type { AnalysisRun } from "./types";

/** UI state for the Runs replay / exact reproduce actions. */
export function reproduceActionState(
    run: Partial<Pick<
        AnalysisRun,
        "rerunnable" | "rerun_block_reason" | "replayable" | "exact_reproducible" | "exact_reproduce_block_reason"
    >>,
): {
    replay: { enabled: boolean; reason: string | null };
    exact: { enabled: boolean; reason: string | null };
} {
    const replayable = run.replayable ?? run.rerunnable;
    if (replayable === false) {
        return {
            replay: {
                enabled: false,
                reason: run.rerun_block_reason?.trim() || "Replay is not available for this run.",
            },
            exact: {
                enabled: false,
                reason: run.exact_reproduce_block_reason?.trim()
                    || run.rerun_block_reason?.trim()
                    || "Exact reproduce is not available for this run.",
            },
        };
    }
    return {
        replay: { enabled: true, reason: null },
        exact: {
            enabled: run.exact_reproducible === true,
            reason:
                run.exact_reproducible === true
                    ? null
                    : run.exact_reproduce_block_reason?.trim()
                        || "Exact reproduce requires frozen inputs and checksums.",
        },
    };
}
