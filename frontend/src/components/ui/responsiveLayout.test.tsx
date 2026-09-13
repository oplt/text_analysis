import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { FormGrid } from "./FormGrid";
import { WorkspaceSplit } from "./WorkspaceSplit";
import { MetricGrid } from "./MetricGrid";

function wrap(ui: React.ReactElement) {
    return render(<ThemeProvider theme={createTheme()}>{ui}</ThemeProvider>);
}

describe("FormGrid", () => {
    it("renders children in a grid without forcing fixed min widths", () => {
        const { container } = wrap(
            <FormGrid columns="4-4-4">
                <div>A</div>
                <div>B</div>
                <div>C</div>
            </FormGrid>
        );
        expect(container.textContent).toContain("ABC");
        const grid = container.firstElementChild as HTMLElement;
        expect(getComputedStyle(grid).display).toBe("grid");
    });
});

describe("WorkspaceSplit", () => {
    it("renders main content and optional side inspector without nesting landmarks", () => {
        wrap(
            <WorkspaceSplit
                main={<div>Main workspace</div>}
                side={<div>Inspector</div>}
                ratio="8-4"
            />
        );
        expect(screen.getByText("Main workspace")).toBeInTheDocument();
        expect(screen.getByText("Inspector")).toBeInTheDocument();
        expect(screen.queryByRole("main")).not.toBeInTheDocument();
        expect(screen.getByRole("complementary")).toBeInTheDocument();
    });
});

describe("MetricGrid", () => {
    it("uses responsive card grid for KPI rows", () => {
        const { container } = wrap(
            <MetricGrid>
                <div>1</div>
                <div>2</div>
                <div>3</div>
                <div>4</div>
            </MetricGrid>
        );
        expect(container.textContent).toBe("1234");
    });
});
