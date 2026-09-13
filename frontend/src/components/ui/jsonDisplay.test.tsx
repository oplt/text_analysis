import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { JsonBlock } from "./JsonBlock";
import {
    formatDisplayValue,
    recordToKeyValueItems,
    stringifyPretty,
} from "./jsonDisplay";

vi.mock("../../app/snackbarContext", () => ({
    useSnackbar: () => ({ showToast: vi.fn() }),
}));

describe("jsonDisplay helpers", () => {
    it("formats nested values without dumping full JSON", () => {
        expect(formatDisplayValue({ a: 1, b: 2 })).toBe("{a, b}");
        expect(formatDisplayValue(["x", "y"])).toBe("x, y");
        expect(formatDisplayValue([{ id: 1 }, { id: 2 }])).toBe("2 items");
    });

    it("flattens records for key/value lists", () => {
        expect(recordToKeyValueItems({ train_size: 80, seed: 7 })).toEqual([
            { key: "train_size", label: "train size", value: "80" },
            { key: "seed", label: "seed", value: "7" },
        ]);
        expect(stringifyPretty({ a: 1 })).toContain('"a": 1');
    });
});

describe("JsonBlock", () => {
    it("renders pretty JSON and copy control", async () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.assign(navigator, { clipboard: { writeText } });

        render(<JsonBlock data={{ hello: "world" }} />);
        expect(screen.getByText(/"hello": "world"/)).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Copy JSON" }));
        await waitFor(() => expect(writeText).toHaveBeenCalled());
    });
});
