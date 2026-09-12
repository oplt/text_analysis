import { describe, expect, it } from "vitest";
import { resolveAsyncRunControls } from "./analysisAsync";

describe("resolveAsyncRunControls", () => {
    it("enables Run async payload when operation supports async", () => {
        const controls = resolveAsyncRunControls(
            { operations: { frequencies: { async: true }, kwic: { async: false } } },
            "frequencies",
            true
        );
        expect(controls.supportsAsync).toBe(true);
        expect(controls.showInlineOnly).toBe(false);
        expect(controls.payload).toEqual({ run_async: true });
    });

    it("forces Inline only and omits run_async for unsupported ops", () => {
        const controls = resolveAsyncRunControls(
            { operations: { frequencies: { async: true }, kwic: { async: false } } },
            "kwic",
            true
        );
        expect(controls.supportsAsync).toBe(false);
        expect(controls.showInlineOnly).toBe(true);
        expect(controls.payload).toEqual({});
    });
});
