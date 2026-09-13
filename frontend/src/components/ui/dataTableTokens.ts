/**
 * Shared data-table tokens (Phase 17).
 * Prefer these over per-page Table sx for density, sticky chrome, and truncation.
 */

export type DataTableDensity = "compact" | "comfortable";

export const dataTableDensity = {
    compact: {
        size: "small" as const,
        rowMinHeight: 36,
        cellPy: 0.5,
        headerPy: 0.75,
    },
    comfortable: {
        size: "medium" as const,
        rowMinHeight: 52,
        cellPy: 1.25,
        headerPy: 1.25,
    },
} as const;

export const dataTableDefaults = {
    maxHeight: { xs: 360, sm: 420, md: "min(62vh, 720px)" } as const,
    truncateChars: 48,
    stickyZIndex: {
        header: 3,
        stickyColumn: 2,
        stickyHeaderColumn: 4,
    },
} as const;

export function tableSizeForDensity(density: DataTableDensity): "small" | "medium" {
    return dataTableDensity[density].size;
}
