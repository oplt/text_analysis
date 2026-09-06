import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MatrixHeatmap } from "./ResearchCharts";

describe("MatrixHeatmap", () => {
    it("distinguishes missing values from zero agreement", () => {
        render(
            <MatrixHeatmap
                rowLabels={["Coder A"]}
                colLabels={["Coder B", "Coder C"]}
                values={[[null, 0]]}
                cellDetails={(_row, col) => (col === 0 ? "No shared annotations" : "3 shared units")}
            />
        );

        expect(screen.getByText("—")).toBeInTheDocument();
        expect(screen.getByText("0.00")).toBeInTheDocument();
    });
});
