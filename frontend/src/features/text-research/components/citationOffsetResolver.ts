import type { CitationSourceCoordinate } from "../../../api/textResearch";

export type PageProvenance = {
    page_number?: number | null;
    char_start?: number | null;
    char_end?: number | null;
    source_span_ids?: string[] | null;
    offset_scope_id?: string | null;
    source_spans?: CitationSourceCoordinate[] | null;
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
    offsetScopeId,
    offsetCoordinateSystem,
    sourceSpans,
}: {
    text: string;
    charStart?: number | null;
    charEnd?: number | null;
    pageNumber?: number | null;
    pageProvenance?: PageProvenance[];
    sourceSpanIds?: string[] | null;
    offsetScope?: OffsetScope;
    offsetScopeId?: string | null;
    offsetCoordinateSystem?: string | null;
    sourceSpans?: CitationSourceCoordinate[] | null;
}): { start: number; end: number } | null {
    function resolveRange(start: number | null | undefined, end: number | null | undefined, source: string, system = offsetCoordinateSystem) {
        if (system && system !== "unicode_code_points_zero_based_end_exclusive") return null;
        const points = system ? Array.from(source) : null;
        const range = validRange(start, end, points?.length ?? source.length);
        if (!range || !points) return range;
        return { start: points.slice(0, range.start).join("").length, end: points.slice(0, range.end).join("").length };
    }
    const exactSpan = sourceSpans?.find(span => span.scope_id === offsetScopeId)
        ?? sourceSpans?.find(span => sourceSpanIds?.includes(span.source_span_id ?? ""));
    if (exactSpan) {
        offsetScope = exactSpan.scope_type;
        charStart = exactSpan.start;
        charEnd = exactSpan.end;
        offsetCoordinateSystem = exactSpan.coordinate_system;
        offsetScopeId = exactSpan.scope_id;
    }
    if (offsetScope === "canonical_document") {
        return resolveRange(charStart, charEnd, text);
    }

    const requestedSpans = new Set(sourceSpanIds ?? []);
    const page = pageProvenance?.find(candidate => offsetScopeId && candidate.offset_scope_id === offsetScopeId)
        ?? pageProvenance?.find(candidate => (candidate.source_span_ids ?? []).some(id => requestedSpans.has(id) || id === exactSpan?.source_span_id))
        ?? pageProvenance?.find(candidate => pageNumber != null && candidate.page_number === pageNumber && (!offsetScopeId || !candidate.offset_scope_id || candidate.offset_scope_id === offsetScopeId));
    const pageRange = page && resolveRange(page.char_start, page.char_end, text);
    if (pageRange && (offsetScope === "page" || offsetScope === "parsed_document")) {
        const local = resolveRange(charStart, charEnd, text.slice(pageRange.start, pageRange.end));
        if (local) return { start: pageRange.start + local.start, end: pageRange.start + local.end };
    }

    if (pageRange) return pageRange;
    return null;
}
