import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { PageHeader } from "./PageHeader";
import { KeyValueList, keyValueItemsFromRecord } from "./KeyValueList";
import { AdvancedSettings } from "./AdvancedSettings";
import { ResponsiveCardGrid } from "./ResponsiveCardGrid";
import { MetricGrid } from "./MetricGrid";

function wrap(ui: React.ReactElement) {
    return render(<ThemeProvider theme={createTheme()}>{ui}</ThemeProvider>);
}

describe("PageHeader", () => {
    it("renders title, description, status, and action slots", async () => {
        const user = userEvent.setup();
        const onPrimary = vi.fn();
        const onOverflow = vi.fn();
        wrap(
            <PageHeader
                title="Corpus"
                description="Manage documents"
                status={<span>Ready</span>}
                secondaryActions={<button type="button">Secondary</button>}
                primaryAction={
                    <button type="button" onClick={onPrimary}>
                        Primary
                    </button>
                }
                overflowItems={[{ id: "export", label: "Export", onClick: onOverflow }]}
            />
        );

        expect(screen.getByRole("heading", { name: "Corpus" })).toBeInTheDocument();
        expect(screen.getByText("Manage documents")).toBeInTheDocument();
        expect(screen.getByText("Ready")).toBeInTheDocument();
        await user.click(screen.getByRole("button", { name: "Primary" }));
        expect(onPrimary).toHaveBeenCalled();
        await user.click(screen.getByRole("button", { name: "More actions" }));
        await user.click(screen.getByRole("menuitem", { name: "Export" }));
        expect(onOverflow).toHaveBeenCalled();
    });
});

describe("KeyValueList", () => {
    it("flattens shallow metrics and skips nested objects", () => {
        const items = keyValueItemsFromRecord({
            mean_cohens_kappa: 0.74,
            n: 12,
            nested: { a: 1 },
        });
        expect(items).toEqual([
            {
                key: "mean_cohens_kappa",
                label: "mean cohens kappa",
                value: "0.74",
                helpTermId: "cohens_kappa",
            },
            { key: "n", label: "n", value: "12", helpTermId: undefined },
        ]);
    });

    it("renders label/value pairs", () => {
        wrap(
            <KeyValueList items={[{ key: "kappa", label: "Kappa", value: "0.81" }]} />
        );
        expect(screen.getByText("Kappa")).toBeInTheDocument();
        expect(screen.getByText("0.81")).toBeInTheDocument();
    });
});

describe("AdvancedSettings", () => {
    it("exposes a collapsed advanced accordion", async () => {
        const user = userEvent.setup();
        wrap(
            <AdvancedSettings>
                <div>Secret options</div>
            </AdvancedSettings>
        );
        expect(screen.getByText("Advanced settings")).toBeInTheDocument();
        expect(screen.getByText("Secret options")).not.toBeVisible();
        await user.click(screen.getByRole("button", { name: /Advanced settings/i }));
        expect(screen.getByText("Secret options")).toBeVisible();
    });
});

describe("grids", () => {
    it("renders ResponsiveCardGrid and MetricGrid children", () => {
        const { container } = wrap(
            <MetricGrid>
                <div>A</div>
                <div>B</div>
            </MetricGrid>
        );
        expect(container.textContent).toContain("A");
        wrap(
            <ResponsiveCardGrid columns={3}>
                <div>C</div>
            </ResponsiveCardGrid>
        );
        expect(screen.getByText("C")).toBeInTheDocument();
    });
});
