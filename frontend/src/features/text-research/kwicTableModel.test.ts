import { describe, expect, it } from "vitest";
import { filterKwicRows, kwicRowsToCsv, toKwicSearchRows } from "./kwicTableModel";

describe("kwicTableModel", () => {
    const matches = [
        {
            left_context: "The quick",
            keyword: "brown",
            right_context: "fox jumps",
            document_title: "Fable A",
            organization: "Org One",
        },
        {
            left: "Slow",
            match: "green",
            right: "turtle",
            title: "Fable B",
            organization: "Org Two",
        },
    ];

    it("builds searchable rows without JSON.stringify over the raw entry", () => {
        const rows = toKwicSearchRows(matches);
        expect(rows).toHaveLength(2);
        expect(rows[0].keyword).toBe("brown");
        expect(rows[0].searchText).toContain("brown");
        expect(rows[0].searchText).toContain("fable a");
        expect(rows[1].document).toBe("Fable B");
    });

    it("filters on the memoized searchText", () => {
        const rows = toKwicSearchRows(matches);
        expect(filterKwicRows(rows, "turtle")).toHaveLength(1);
        expect(filterKwicRows(rows, "ORG ONE")[0]?.keyword).toBe("brown");
        expect(filterKwicRows(rows, "  ")).toHaveLength(2);
    });

    it("serializes CSV from searchable rows", () => {
        const csv = kwicRowsToCsv(toKwicSearchRows(matches).slice(0, 1));
        expect(csv.split("\n")[0]).toBe("left,keyword,right,document,organization");
        expect(csv).toContain("brown");
    });
});
