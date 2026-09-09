import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MatrixHeatmap, ReliabilityComparisonChart } from "./ResearchCharts";

vi.mock("@mui/x-charts/BarChart", () => ({
    BarChart: ({
        series,
    }: {
        series: Array<{ label?: string; data: Array<number | null> }>;
    }) => (
        <div data-testid="bar-chart">
            {series.map((item) => (
                <span key={item.label}>{item.label}</span>
            ))}
        </div>
    ),
}));

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

describe("ReliabilityComparisonChart", () => {
    it("plots Cohen and alpha for two-coder labels", () => {
        render(
            <ReliabilityComparisonChart
                items={[
                    { label: "Label A", kappa: 0.8, alpha: 0.75 },
                    { label: "Label B", kappa: 0.6, alpha: 0.55 },
                ]}
            />
        );

        expect(screen.getByText("Cohen's κ")).toBeInTheDocument();
        expect(screen.getByText("Krippendorff's α")).toBeInTheDocument();
        expect(screen.queryByText("Fleiss' κ")).not.toBeInTheDocument();
    });

    it("plots Fleiss and alpha for multi-coder labels", () => {
        render(
            <ReliabilityComparisonChart
                items={[
                    { label: "Label A", fleiss: 0.7, alpha: 0.68 },
                    { label: "Label B", fleiss: 0.5, alpha: 0.48 },
                ]}
            />
        );

        expect(screen.getByText("Fleiss' κ")).toBeInTheDocument();
        expect(screen.getByText("Krippendorff's α")).toBeInTheDocument();
        expect(screen.queryByText("Cohen's κ")).not.toBeInTheDocument();
    });

    it("shows empty state when no metrics are present", () => {
        render(
            <ReliabilityComparisonChart
                items={[{ label: "Label A", kappa: null, fleiss: null, alpha: null }]}
            />
        );

        expect(
            screen.getByText("No evaluable reliability metrics to plot.")
        ).toBeInTheDocument();
    });
});
