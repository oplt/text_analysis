import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DataTable } from "./DataTable";
import { TruncatedCell } from "./TruncatedCell";

type Row = { id: string; name: string; score: number };

const rows: Row[] = [
    { id: "1", name: "zebra", score: 1 },
    { id: "2", name: "apple", score: 3 },
    { id: "3", name: "mango", score: 2 },
];

describe("TruncatedCell", () => {
    it("shows full short text and truncates long text", () => {
        const { rerender } = render(<TruncatedCell maxChars={8}>short</TruncatedCell>);
        expect(screen.getByText("short")).toBeInTheDocument();

        rerender(<TruncatedCell maxChars={8}>{"abcdefghijklmnop"}</TruncatedCell>);
        expect(screen.getByText("abcdefg…")).toBeInTheDocument();
    });
});

describe("DataTable", () => {
    it("sorts, paginates, and toggles density", () => {
        render(
            <DataTable
                ariaLabel="Demo table"
                rows={rows}
                getRowId={(row) => row.id}
                clientSort
                pageSize={2}
                showDensityToggle
                columns={[
                    {
                        id: "name",
                        label: "Name",
                        sortable: true,
                        getSortValue: (row) => row.name,
                        render: (row) => row.name,
                    },
                    {
                        id: "score",
                        label: "Score",
                        sortable: true,
                        getSortValue: (row) => row.score,
                        render: (row) => row.score,
                    },
                ]}
            />
        );

        expect(screen.getByText("zebra")).toBeInTheDocument();
        expect(screen.getByText("apple")).toBeInTheDocument();
        expect(screen.queryByText("mango")).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: /Name/i }));
        expect(screen.getByText("apple")).toBeInTheDocument();
        expect(screen.getByText("mango")).toBeInTheDocument();
        expect(screen.queryByText("zebra")).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Go to next page" }));
        expect(screen.getByText("zebra")).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Comfortable" }));
        expect(screen.getByRole("button", { name: "Comfortable" })).toHaveAttribute(
            "aria-pressed",
            "true"
        );
    });

    it("renders empty and loading states", () => {
        const { rerender } = render(
            <DataTable
                ariaLabel="Empty table"
                rows={[]}
                getRowId={(row: Row) => row.id}
                columns={[{ id: "name", label: "Name", render: (row) => row.name }]}
                emptyIcon={<span>∅</span>}
                emptyTitle="Nothing here"
                emptyDescription="Add a row to begin."
            />
        );
        expect(screen.getByText("Nothing here")).toBeInTheDocument();

        rerender(
            <DataTable
                ariaLabel="Loading table"
                loading
                rows={[]}
                getRowId={(row: Row) => row.id}
                columns={[{ id: "name", label: "Name", render: (row) => row.name }]}
            />
        );
        expect(screen.queryByText("Nothing here")).not.toBeInTheDocument();
    });
});
