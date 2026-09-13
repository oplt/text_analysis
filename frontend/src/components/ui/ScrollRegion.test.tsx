import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { ScrollRegion } from "./ScrollRegion";

describe("ScrollRegion", () => {
    it("renders children in a horizontal scroll container", () => {
        const { container } = render(
            <ThemeProvider theme={createTheme()}>
                <ScrollRegion>
                    <div>wide content</div>
                </ScrollRegion>
            </ThemeProvider>
        );
        expect(container.textContent).toContain("wide content");
        const el = container.firstElementChild as HTMLElement;
        expect(getComputedStyle(el).overflowX).toBe("auto");
    });
});
