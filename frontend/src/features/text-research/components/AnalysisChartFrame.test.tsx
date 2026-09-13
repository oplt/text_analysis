import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnalysisChartFrame } from "./AnalysisChartFrame";

describe("AnalysisChartFrame", () => {
    it("renders standardized chart chrome", () => {
        render(
            <AnalysisChartFrame
                title="Top terms"
                subtitle="Highest-ranked tokens"
                legend="Frequency"
                xAxisLabel="Count"
                yAxisLabel="Term"
                methodologicalNote="Ranks depend on tokenization."
                exportAction={<button type="button">Export CSV</button>}
            >
                <div>chart-body</div>
            </AnalysisChartFrame>
        );

        expect(screen.getByText("Top terms")).toBeInTheDocument();
        expect(screen.getByText("Highest-ranked tokens")).toBeInTheDocument();
        expect(screen.getByText("Frequency")).toBeInTheDocument();
        expect(screen.getByText(/Vertical: Term/)).toBeInTheDocument();
        expect(screen.getByText(/Horizontal: Count/)).toBeInTheDocument();
        expect(screen.getByText("Ranks depend on tokenization.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Export CSV" })).toBeInTheDocument();
        expect(screen.getByText("chart-body")).toBeInTheDocument();
    });
});
