import type { AnalysisRun } from "../types";

export type ResultProvenance = {
    runtime: Record<string, unknown> | null;
    identity: Record<string, unknown> | null;
    artifacts: unknown[];
};

function asRecord(value: unknown): Record<string, unknown> | null {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
    }
    return null;
}

export function formatProvenanceValue(value: unknown): string {
    return typeof value === "object" ? JSON.stringify(value) : String(value);
}

export function getRunResultProvenance(run: AnalysisRun): ResultProvenance {
    const persistedResults = asRecord(run.results);
    const canonicalResult = asRecord(persistedResults?.analysis_result);
    const canonicalArtifacts = canonicalResult?.artifacts;
    const persistedArtifacts = persistedResults?.artifacts;
    return {
        runtime:
            asRecord(canonicalResult?.runtime) ??
            asRecord(persistedResults?.runtime) ??
            asRecord(run.parameters?.engine),
        identity:
            asRecord(canonicalResult?.identity) ?? asRecord(persistedResults?.identity),
        artifacts: Array.isArray(canonicalArtifacts)
            ? canonicalArtifacts
            : Array.isArray(persistedArtifacts)
              ? persistedArtifacts
              : [],
    };
}
