import { alpha, type Theme } from "@mui/material/styles";
import { colors } from "../../app/designTokens";

/**
 * Shared surface / semantic color helpers for light + dark parity (Phase 23).
 * Prefer these over hard-coded `colors.lightAsh` / hex values in UI components.
 */

/** Primary card fill (SectionCard default, StatCard). */
export function surfaceCard(theme: Theme): string {
    return theme.palette.mode === "dark"
        ? theme.palette.background.paper
        : colors.lightAsh;
}

/** Softer nested / subtle fill. */
export function surfaceMuted(theme: Theme): string {
    return theme.palette.mode === "dark"
        ? alpha(theme.palette.common.white, 0.05)
        : colors.lightAsh;
}

/** Raised panel on the page background (nested tiles, dialogs content). */
export function surfaceRaised(theme: Theme): string {
    return theme.palette.mode === "dark"
        ? theme.palette.background.paper
        : colors.white;
}

/** Soft selected / flash highlight. */
export function surfaceSelected(theme: Theme): string {
    return alpha(
        theme.palette.primary.main,
        theme.palette.mode === "dark" ? 0.22 : 0.08
    );
}

/** Code / raw JSON block background. */
export function surfaceCode(theme: Theme): string {
    return theme.palette.mode === "dark"
        ? alpha(theme.palette.common.white, 0.06)
        : alpha(theme.palette.common.black, 0.04);
}

/** Sticky table edge shadow that reads in both themes. */
export function stickyEdgeShadow(theme: Theme, side: "left" | "right"): string {
    const edge = theme.palette.divider;
    return side === "left" ? `inset -1px 0 0 ${edge}` : `inset 1px 0 0 ${edge}`;
}

/**
 * Chart series colors with adequate contrast on both themes.
 * Index 0 is the primary “scientific” series.
 */
export function chartSeriesColors(theme: Theme): string[] {
    if (theme.palette.mode === "dark") {
        return [
            theme.palette.success.main,
            theme.palette.primary.light,
            theme.palette.warning.main,
            "#f472b6",
            "#2dd4bf",
        ];
    }
    return [
        theme.palette.success.main,
        theme.palette.primary.main,
        theme.palette.warning.main,
        "#9d174d",
        "#0f766e",
    ];
}

/** Heatmap cell fill from intensity 0–1 using the theme primary. */
export function chartHeatFill(theme: Theme, intensity: number): string {
    const t = Math.max(0, Math.min(1, intensity));
    return alpha(theme.palette.primary.main, 0.08 + t * 0.72);
}

/** Label fill on filled chart nodes (always light-on-dark node). */
export function chartNodeLabelColor(): string {
    return colors.white;
}
