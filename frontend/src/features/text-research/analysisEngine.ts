import type { AnalysisEngineCapability } from "../../api/textResearch";

export type REngineSelectionState = "checking" | "available" | "unavailable" | "unsupported";

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

export function rEngineSelectionState(
    engines: AnalysisEngineCapability[] | undefined,
    analysis: string
): REngineSelectionState {
    if (!engines) return "checking";
    const rEngine = engines.find((engine) => engine.name === "r");
    if (!rEngine || !rEngine.available) return "unavailable";
    return rEngine.analyses.includes(analysis) ? "available" : "unsupported";
}

export function rEngineOptionLabel(state: REngineSelectionState): string {
    switch (state) {
        case "checking":
            return "R / quanteda (checking availability)";
        case "unavailable":
            return "R / quanteda (unavailable)";
        case "unsupported":
            return "R / quanteda (not supported for this analysis)";
        default:
            return "R / quanteda";
    }
}
