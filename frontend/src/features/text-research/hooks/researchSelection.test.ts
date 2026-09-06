import { afterEach, describe, expect, it } from "vitest";
import { resolveResearchSelectionId } from "./researchSelection";

const storageKey = (projectId: string) => `text-research:corpus:${projectId}`;

describe("resolveResearchSelectionId", () => {
    afterEach(() => {
        localStorage.clear();
    });

    it("clears selection and storage when nothing is available", () => {
        localStorage.setItem(storageKey("p1"), "stale-id");
        const next = resolveResearchSelectionId({
            projectId: "p1",
            storageKey,
            availableIds: [],
            currentId: "stale-id",
        });
        expect(next).toBe("");
        expect(localStorage.getItem(storageKey("p1"))).toBeNull();
    });

    it("drops invalid stored ids and falls back to the first available id", () => {
        localStorage.setItem(storageKey("p1"), "deleted-corpus");
        const next = resolveResearchSelectionId({
            projectId: "p1",
            storageKey,
            availableIds: ["c1", "c2"],
            currentId: "deleted-corpus",
        });
        expect(next).toBe("c1");
        expect(localStorage.getItem(storageKey("p1"))).toBe("c1");
    });

    it("keeps a valid stored id", () => {
        localStorage.setItem(storageKey("p1"), "c2");
        const next = resolveResearchSelectionId({
            projectId: "p1",
            storageKey,
            availableIds: ["c1", "c2"],
            currentId: "c1",
        });
        expect(next).toBe("c2");
    });
});
