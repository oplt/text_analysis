import { describe, expect, it } from "vitest";

import { activeRunRefetchInterval } from "./runPolling";

const activeQuery = { state: { data: { status: "running" } } };

describe("activeRunRefetchInterval", () => {
    it("uses polling only when SSE is unavailable", () => {
        expect(activeRunRefetchInterval(activeQuery)).toBe(2000);
        expect(activeRunRefetchInterval(activeQuery, true)).toBe(false);
    });
});
