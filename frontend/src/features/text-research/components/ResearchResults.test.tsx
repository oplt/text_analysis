import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResearchResultsTable } from "./ResearchResults";

describe("ResearchResultsTable", () => {
    it("sorts and paginates analysis rows", () => {
        render(
            <ResearchResultsTable
                pageSize={1}
                rows={[
                    { id: "a", term: "zebra", count: 1 },
                    { id: "b", term: "apple", count: 2 },
                ]}
                columns={[
                    { id: "term", label: "Term", value: (row) => row.term },
                    { id: "count", label: "Count", value: (row) => row.count },
                ]}
            />
        );

        expect(screen.getByText("zebra")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Term" }));
        expect(screen.getByText("apple")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Next" }));
        expect(screen.getByText("zebra")).toBeInTheDocument();
    });
});
