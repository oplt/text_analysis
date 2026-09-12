export type PageProvenance = {
    page_number?: number | null;
    char_start?: number | null;
    char_end?: number | null;
    source_span_ids?: string[] | null;
};

type OffsetScope = "parsed_document" | "page" | "canonical_document" | null | undefined;

function validRange(
    start: number | null | undefined,
    end: number | null | undefined,
    textLength: number
): { start: number; end: number } | null {
    if (
        typeof start !== "number" ||
        typeof end !== "number" ||
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end <= start ||
        end > textLength
    ) {
        return null;
    }
    return { start, end };
}

export function resolveCitationHighlightRange({
    text,
    charStart,
    charEnd,
    pageNumber,
    pageProvenance,
    sourceSpanIds,
    offsetScope,
}: {
    text: string;
    charStart?: number | null;
    charEnd?: number | null;
    pageNumber?: number | null;
    pageProvenance?: PageProvenance[];
    sourceSpanIds?: string[] | null;
    offsetScope?: OffsetScope;
}): { start: number; end: number } | null {
    if (offsetScope === "canonical_document") {
        return validRange(charStart, charEnd, text.length);
    }

    const requestedSpans = new Set(sourceSpanIds ?? []);
    const page = pageProvenance?.find(
        (candidate) =>
            (pageNumber != null && candidate.page_number === pageNumber) ||
            (candidate.source_span_ids ?? []).some((id) => requestedSpans.has(id))
    );
    const pageRange = page && validRange(page.char_start, page.char_end, text.length);
    if (pageRange && (offsetScope === "page" || offsetScope === "parsed_document")) {
        const local = validRange(charStart, charEnd, pageRange.end - pageRange.start);
        if (local) return { start: pageRange.start + local.start, end: pageRange.start + local.end };
    }

    if (pageRange) return pageRange;
    return null;
}
