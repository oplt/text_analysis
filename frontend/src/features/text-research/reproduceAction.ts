import type { AnalysisRun } from "../types";

/** UI state for the Runs “Reproduce” action, driven by API capability fields. */
export function reproduceActionState(
    run: Pick<AnalysisRun, "rerunnable" | "rerun_block_reason">,
): { enabled: boolean; reason: string | null } {
    if (run.rerunnable === false) {
        return {
            enabled: false,
            reason:
                run.rerun_block_reason?.trim() ||
                "Reproduce is not available for this run.",
        };
    }
    return { enabled: true, reason: null };
}
