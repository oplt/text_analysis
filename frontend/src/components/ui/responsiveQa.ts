/**
 * Responsive QA viewports and shared scroll/layout helpers (Phase 22).
 *
 * MUI defaults used by the app:
 * - sm 600 · md 900 · lg 1200 · xl 1536
 *
 * Content width at 1024 with the permanent nav (~195px) is ~800px — too tight
 * for 3–4 equal columns, so dense grids wait until `lg`.
 */

export const RESPONSIVE_QA_VIEWPORTS = [390, 768, 1024, 1280, 1440, 1920] as const;

export type ResponsiveQaViewport = (typeof RESPONSIVE_QA_VIEWPORTS)[number];

/** Horizontal scroll wrapper for tables, charts, and dense toolbars. */
export const scrollContainerSx = {
    width: "100%",
    maxWidth: "100%",
    minWidth: 0,
    overflowX: "auto",
    WebkitOverflowScrolling: "touch",
    scrollbarWidth: "thin",
} as const;

/**
 * Column template for KPI / card grids.
 * 1 → 2 (sm) → full count only at lg+ (laptop+).
 */
export function denseCardColumns(columns: 2 | 3 | 4): {
    xs: string;
    sm: string;
    md: string;
    lg: string;
} {
    return {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: `repeat(${columns}, minmax(0, 1fr))`,
    };
}

/** Which Phase 22 viewport band a CSS width falls into (for tests / docs). */
export function responsiveQaBand(widthPx: number): "phone" | "tablet" | "laptop" | "desktop" {
    if (widthPx < 600) return "phone";
    if (widthPx < 900) return "tablet";
    if (widthPx < 1280) return "laptop";
    return "desktop";
}
