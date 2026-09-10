/**
 * Mutability-aware TanStack Query stale times.
 * Prefer these over ad-hoc numbers so research data freshness matches lifecycle.
 */
export const QUERY_STALE_TIMES = {
    default: 60_000,
    notifications: 30_000,
    notificationPreferences: 5 * 60_000,
    userProfile: 90_000,
    userSessions: 60_000,
    userDirectory: 60_000,
    platformMetadata: 5 * 60_000,
    /** Projects list/detail — moderate freshness */
    projects: 30_000,
    calendar: 60_000,
    aiOverview: 60_000,
    aiReviews: 30_000,
    aiEvaluationRuns: 60_000,
    aiPromptVersions: 60_000,
    aiDatasetCases: 60_000,

    /**
     * Generic research reference (corpora lists, profiles, dictionaries).
     * Prefer more specific keys below when the resource type is known.
     */
    researchReference: 5 * 60_000,

    /** Active AnalysisRun — aggressive; pair with SSE + refetchInterval */
    researchActiveRun: 0,

    /** Completed / terminal analysis runs — treat as immutable for the session */
    researchCompletedRun: 30 * 60_000,

    /** Annotation queue / progress — short stale time */
    researchAnnotationQueue: 15_000,

    /** Metadata facets — change rarely during a session */
    researchMetadataFacets: 5 * 60_000,

    /** Codebook versions / labels — long (versions are append-only after freeze) */
    researchCodebook: 15 * 60_000,

    /** Frozen training dataset snapshots — session-immutable */
    researchFrozenSnapshot: 30 * 60_000,

    /** Run provenance payloads — effectively immutable once written */
    researchProvenance: 30 * 60_000,

    /** Model holdout metrics / coefficients — long (tied to a trained artifact) */
    researchModelMetrics: 10 * 60_000,

    /** Model lifecycle status / events — shorter (mutable promotions) */
    researchModelLifecycle: 30_000,

    /** Prediction sets listing — moderate */
    researchPredictionSets: 60_000,
} as const;

export const NOTIFICATIONS_REFETCH_INTERVAL_MS = 60_000;
export const QUERY_GC_TIME_MS = 10 * 60_000;

const ACTIVE_RUN_STATUSES = new Set(["queued", "running", "pending"]);

/** Stale time for AnalysisRun queries based on last-known status. */
export function researchRunStaleTime(status?: string | null): number {
    if (status && !ACTIVE_RUN_STATUSES.has(status)) {
        return QUERY_STALE_TIMES.researchCompletedRun;
    }
    return QUERY_STALE_TIMES.researchActiveRun;
}
