import { describe, expect, it } from "vitest";
import { resolveAsyncRunControls } from "./analysisAsync";

describe("resolveAsyncRunControls", () => {
    it("enables Run async payload when operation supports async", () => {
        const controls = resolveAsyncRunControls(
            {
                operations: {
                    frequencies: { async: true, default_execution_mode: "auto" },
                    kwic: { async: false, default_execution_mode: "inline" },
                },
            },
            "frequencies",
            true
        );
        expect(controls.supportsAsync).toBe(true);
        expect(controls.showInlineOnly).toBe(false);
        expect(controls.defaultExecutionMode).toBe("auto");
        expect(controls.payload).toEqual({ run_async: true });
    });

    it("forces Inline only and omits run_async for unsupported ops", () => {
        const controls = resolveAsyncRunControls(
            {
                operations: {
                    frequencies: { async: true, default_execution_mode: "auto" },
                    kwic: { async: false, default_execution_mode: "inline" },
                },
            },
            "kwic",
            true
        );
        expect(controls.supportsAsync).toBe(false);
        expect(controls.showInlineOnly).toBe(true);
        expect(controls.defaultExecutionMode).toBe("inline");
        expect(controls.payload).toEqual({});
    });
});
