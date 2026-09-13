export type AnalysisCapabilityMap = {
    operations?: Record<
        string,
        { async?: boolean; default_execution_mode?: string } | undefined
    >;
};

/** Derive UI + request payload from per-operation async capability. */
export function resolveAsyncRunControls(
    capabilities: AnalysisCapabilityMap | undefined,
    operation: string,
    runAsync: boolean
): {
    supportsAsync: boolean;
    showInlineOnly: boolean;
    defaultExecutionMode: string;
    payload: { run_async?: boolean };
} {
    const entry = capabilities?.operations?.[operation];
    const supportsAsync = entry?.async === true;
    return {
        supportsAsync,
        showInlineOnly: !supportsAsync,
        defaultExecutionMode: entry?.default_execution_mode ?? (supportsAsync ? "auto" : "inline"),
        payload: supportsAsync ? { run_async: runAsync } : {},
    };
}
