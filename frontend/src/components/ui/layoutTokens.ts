/**
 * Shared layout spacing and hierarchy tokens (Phase 3).
 * Prefer these over ad-hoc per-page padding/margins.
 */

export const layoutSpacing = {
    /** Vertical stack gap inside PageShell. */
    pageGap: {
        default: { xs: 3, md: 4 },
        dense: { xs: 2, md: 2.5 },
    },
    /** Horizontal page padding. */
    pagePaddingX: {
        default: { xs: 2, md: 3 },
        dense: { xs: 2, md: 2.5 },
    },
    /** Vertical page padding. */
    pagePaddingY: {
        default: { xs: 2.5, md: 4 },
        dense: { xs: 2, md: 2.5 },
    },
    /** Gap between sibling SectionCards / grids. */
    sectionGap: 2,
    /** Gap inside ResponsiveCardGrid / MetricGrid. */
    gridGap: 2,
    /** SectionCard padding. */
    cardPadding: {
        default: { xs: 2.5, md: 3 },
        compact: { xs: 1.5, md: 2 },
    },
    /** Space below page / section headers before body. */
    headerMarginBottom: {
        default: 2,
        dense: 1.5,
        compact: 1.5,
    },
} as const;

/** Heading roles for consistent hierarchy (map to MUI variants). */
export const headingHierarchy = {
    page: "h4",
    pageDense: "h5",
    section: "h5",
    sectionCompact: "subtitle1",
    subsection: "subtitle2",
} as const;

/** Prefer FormGrid / MetricGrid over hardcoded `repeat(4, 1fr)`. Dense counts wait for lg. */
export const responsiveColumns = {
    kpi: {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: "repeat(4, minmax(0, 1fr))",
    },
    twoUp: {
        xs: "minmax(0, 1fr)",
        md: "repeat(2, minmax(0, 1fr))",
    },
    threeUp: {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: "repeat(3, minmax(0, 1fr))",
    },
} as const;
