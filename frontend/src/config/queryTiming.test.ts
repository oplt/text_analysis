import { describe, expect, it } from "vitest";
import { QUERY_STALE_TIMES, researchRunStaleTime } from "./queryTiming";

describe("queryTiming", () => {
    it("uses aggressive stale time for active runs and long for completed", () => {
        expect(researchRunStaleTime("running")).toBe(QUERY_STALE_TIMES.researchActiveRun);
        expect(researchRunStaleTime("queued")).toBe(QUERY_STALE_TIMES.researchActiveRun);
        expect(researchRunStaleTime("pending")).toBe(QUERY_STALE_TIMES.researchActiveRun);
        expect(researchRunStaleTime("completed")).toBe(QUERY_STALE_TIMES.researchCompletedRun);
        expect(researchRunStaleTime("failed")).toBe(QUERY_STALE_TIMES.researchCompletedRun);
        expect(researchRunStaleTime(undefined)).toBe(QUERY_STALE_TIMES.researchActiveRun);
    });

    it("keeps mutability tiers ordered as expected", () => {
        expect(QUERY_STALE_TIMES.researchActiveRun).toBeLessThan(
            QUERY_STALE_TIMES.researchAnnotationQueue
        );
        expect(QUERY_STALE_TIMES.researchAnnotationQueue).toBeLessThan(
            QUERY_STALE_TIMES.researchModelLifecycle
        );
        expect(QUERY_STALE_TIMES.researchModelLifecycle).toBeLessThanOrEqual(
            QUERY_STALE_TIMES.projects
        );
        expect(QUERY_STALE_TIMES.researchMetadataFacets).toBeGreaterThanOrEqual(
            QUERY_STALE_TIMES.projects
        );
        expect(QUERY_STALE_TIMES.researchFrozenSnapshot).toBeGreaterThan(
            QUERY_STALE_TIMES.researchModelLifecycle
        );
        expect(QUERY_STALE_TIMES.researchProvenance).toBe(
            QUERY_STALE_TIMES.researchCompletedRun
        );
    });
});
