import { describe, expect, it } from "vitest";
import {
    RESEARCH_NAV_GROUPS,
    allResearchNavRoutes,
    pathForNavItem,
    resolveResearchNav,
} from "./researchNavigation";
import { RESEARCH_TABS } from "./types";

describe("researchNavigation", () => {
    it("covers prepare and all RESEARCH_TABS routes", () => {
        const routes = new Set(allResearchNavRoutes());
        expect(routes.has("prepare")).toBe(true);
        for (const tab of RESEARCH_TABS) {
            expect(routes.has(tab.slug)).toBe(true);
        }
    });

    it("preserves existing URL shapes", () => {
        expect(pathForNavItem("p1", { id: "dashboard", label: "Dashboard", route: "dashboard" })).toBe(
            "/research/p1/dashboard"
        );
        expect(
            pathForNavItem("p1", {
                id: "prepare",
                label: "Prepare",
                route: "prepare",
                tab: "segment",
            })
        ).toBe("/research/p1/prepare");
        expect(
            pathForNavItem("p1", {
                id: "cleaning",
                label: "Cleaning",
                route: "prepare",
                tab: "cleaning",
            })
        ).toBe("/research/p1/prepare?tab=cleaning");
        expect(
            pathForNavItem("p1", { id: "models", label: "Model Registry", route: "models" })
        ).toBe("/research/p1/models");
    });

    it("resolves group and item from path + search", () => {
        expect(resolveResearchNav("/research/p1/dashboard", "p1")).toMatchObject({
            group: { id: "overview" },
            item: { id: "dashboard" },
        });
        expect(resolveResearchNav("/research/p1/prepare", "p1")).toMatchObject({
            group: { id: "data" },
            item: { id: "prepare" },
        });
        expect(resolveResearchNav("/research/p1/prepare", "p1", "?tab=ingestion")).toMatchObject({
            group: { id: "data" },
            item: { id: "ingestion" },
        });
        expect(resolveResearchNav("/research/p1/prepare", "p1", "?tab=segment")).toMatchObject({
            group: { id: "data" },
            item: { id: "prepare" },
        });
        expect(resolveResearchNav("/research/p1/prepare", "p1", "?tab=cleaning")).toMatchObject({
            group: { id: "data" },
            item: { id: "cleaning" },
        });
        expect(resolveResearchNav("/research/p1/reliability", "p1")).toMatchObject({
            group: { id: "coding" },
            item: { id: "reliability" },
        });
        expect(resolveResearchNav("/research/p1/classification", "p1")).toMatchObject({
            group: { id: "models" },
            item: { id: "classification" },
        });
        expect(resolveResearchNav("/research/p1/runs", "p1")).toMatchObject({
            group: { id: "outputs" },
            item: { id: "runs" },
        });
    });

    it("exposes six top-level groups from Phase 2 IA", () => {
        expect(RESEARCH_NAV_GROUPS.map((g) => g.id)).toEqual([
            "overview",
            "data",
            "coding",
            "analysis",
            "models",
            "outputs",
        ]);
    });
});
