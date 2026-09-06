import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetchItems } from "./client";

describe("apiFetchItems", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("returns items without changing request behavior", async () => {
        const fetchMock = vi.fn().mockResolvedValue(
            new Response(
                JSON.stringify({ items: [{ id: "item-1" }], total: 1, limit: 50, offset: 0 }),
                { status: 200, headers: { "Content-Type": "application/json" } }
            )
        );
        vi.stubGlobal("fetch", fetchMock);

        const items = await apiFetchItems<{ id: string }>("/items");

        expect(items).toEqual([{ id: "item-1" }]);
        expect(fetchMock).toHaveBeenCalledWith(
            "http://localhost:8000/api/v1/items",
            expect.objectContaining({ credentials: "include" })
        );
    });
});
