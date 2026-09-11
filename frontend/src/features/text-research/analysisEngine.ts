import type { AnalysisEngineCapability } from "../../api/textResearch";

export type REngineSelectionState =
    | "checking"
    | "available"
    | "not_ready"
    | "unavailable"
    | "unsupported";

function asRecord(value: unknown): Record<string, unknown> | null {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
    }
    return null;
}

export function canonicalAnalysisResults(value: unknown): Record<string, unknown> | null {
    const persistedResults = asRecord(value);
    const canonicalResult = asRecord(persistedResults?.analysis_result);
    return asRecord(canonicalResult?.results) ?? persistedResults;
}

export function analysisTypeForTab(tab: string): string {
    if (tab === "dictionaries") return "dictionary";
    return tab;
}

export function rEngineSelectionState(
    engines: AnalysisEngineCapability[] | undefined,
    analysis: string
): REngineSelectionState {
    if (!engines) return "checking";
    const rEngine = engines.find((engine) => engine.name === "r");
    if (!rEngine || !rEngine.available) return "unavailable";
    if (rEngine.ready !== true) return "not_ready";
    const canonical = analysisTypeForTab(analysis);
    return rEngine.analyses.includes(canonical) ? "available" : "unsupported";
}

export function rEngineOptionLabel(state: REngineSelectionState): string {
    switch (state) {
        case "checking":
            return "R / quanteda (checking availability)";
        case "unavailable":
            return "R / quanteda (unavailable)";
        case "not_ready":
            return "R / quanteda (worker offline)";
        case "unsupported":
            return "R / quanteda (not supported for this analysis)";
        default:
            return "R / quanteda";
    }
}
