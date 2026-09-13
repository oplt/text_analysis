import { describe, expect, it } from "vitest";
import { resolveCitationHighlightRange } from "./citationOffsetResolver";

describe("resolveCitationHighlightRange", () => {
    it("prefers an OCR scope and span over an ambiguous page hint", () => {
        const range = resolveCitationHighlightRange({
            text: "same same", pageNumber: 1, offsetScope: "page", offsetScopeId: "ocr:rev:2",
            sourceSpans: [{ scope_type: "page", scope_id: "ocr:rev:2", coordinate_system: "unicode_code_points_zero_based_end_exclusive", start: 0, end: 4, source_span_id: "second" }],
            pageProvenance: [
                { page_number: 1, char_start: 0, char_end: 4, offset_scope_id: "page:rev:1" },
                { page_number: 2, char_start: 5, char_end: 9, offset_scope_id: "ocr:rev:2", source_span_ids: ["second"] },
            ],
        });
        expect(range).toEqual({ start: 5, end: 9 });
    });

    it("converts Unicode code points to JavaScript offsets", () => {
        expect(resolveCitationHighlightRange({ text: "😀target", offsetScope: "canonical_document", offsetCoordinateSystem: "unicode_code_points_zero_based_end_exclusive", charStart: 1, charEnd: 7 })).toEqual({ start: 2, end: 8 });
    });

    it("rejects unavailable scopes, unsupported coordinates, and invalid offsets", () => {
        expect(resolveCitationHighlightRange({ text: "same", offsetScopeId: "old", pageNumber: 1, pageProvenance: [{ page_number: 1, offset_scope_id: "new", char_start: 0, char_end: 4 }] })).toBeNull();
        for (const start of [-1, 9, 1.5]) {
            expect(resolveCitationHighlightRange({ text: "same", offsetScope: "canonical_document", charStart: start, charEnd: 4 })).toBeNull();
        }
        expect(resolveCitationHighlightRange({ text: "same", offsetScope: "canonical_document", offsetCoordinateSystem: "bytes", charStart: 0, charEnd: 4 })).toBeNull();
    });

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
