import { describe, expect, it } from "vitest";
import { resolveCitationHighlightRange } from "./citationOffsetResolver";

describe("resolveCitationHighlightRange", () => {
    it("translates a page-two local offset instead of using it against canonical text", () => {
        const text = "same text on page one\n\npage two target";
        const range = resolveCitationHighlightRange({
            text,
            charStart: 9,
            charEnd: 15,
            pageNumber: 2,
            offsetScope: "page",
            pageProvenance: [
                { page_number: 1, char_start: 0, char_end: 21, source_span_ids: ["page-1"] },
                { page_number: 2, char_start: 23, char_end: text.length, source_span_ids: ["page-2"] },
            ],
        });

        expect(range).toEqual({ start: 32, end: 38 });
        expect(text.slice(range!.start, range!.end)).toBe("target");
    });
});
