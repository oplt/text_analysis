export type AnalysisCapabilityMap = {
    operations?: Record<string, { async?: boolean } | undefined>;
};

/** Derive UI + request payload from per-operation async capability. */
export function resolveAsyncRunControls(
    capabilities: AnalysisCapabilityMap | undefined,
    operation: string,
    runAsync: boolean
): {
    supportsAsync: boolean;
    showInlineOnly: boolean;
    payload: { run_async?: boolean };
} {
    const supportsAsync = capabilities?.operations?.[operation]?.async === true;
    return {
        supportsAsync,
        showInlineOnly: !supportsAsync,
        payload: supportsAsync ? { run_async: runAsync } : {},
    };
}
