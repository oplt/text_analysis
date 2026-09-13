import { describe, expect, it } from "vitest";
import { createTheme } from "@mui/material/styles";
import { lightTheme, darkTheme } from "../../app/theme";
import {
    chartHeatFill,
    chartSeriesColors,
    stickyEdgeShadow,
    surfaceCard,
    surfaceCode,
    surfaceMuted,
    surfaceRaised,
    surfaceSelected,
} from "./themeSurfaces";

describe("Phase 23 dark mode surfaces", () => {
    it("keeps light and dark themes available", () => {
        expect(lightTheme.palette.mode).toBe("light");
        expect(darkTheme.palette.mode).toBe("dark");
    });

    it("uses distinct card surfaces that remain opaque in both modes", () => {
        expect(surfaceCard(lightTheme)).not.toEqual(surfaceCard(darkTheme));
        expect(surfaceRaised(lightTheme)).not.toEqual(surfaceMuted(darkTheme));
        expect(surfaceCode(darkTheme)).toMatch(/rgba|rgb|#/i);
        expect(surfaceSelected(darkTheme)).toMatch(/rgba|rgb|#/i);
    });

    it("keeps semantic success/warning/error readable in dark mode", () => {
        expect(darkTheme.palette.success.main).not.toBe(lightTheme.palette.success.main);
        expect(darkTheme.palette.warning.main).not.toBe(lightTheme.palette.warning.main);
        expect(darkTheme.palette.error.main).not.toBe(lightTheme.palette.error.main);
        expect(darkTheme.palette.text.disabled).not.toBe(lightTheme.palette.text.disabled);
    });

    it("provides theme-aware chart colors and sticky edges", () => {
        const lightSeries = chartSeriesColors(lightTheme);
        const darkSeries = chartSeriesColors(darkTheme);
        expect(lightSeries).toHaveLength(5);
        expect(darkSeries).toHaveLength(5);
        expect(lightSeries[0]).not.toBe(darkSeries[0]);
        expect(chartHeatFill(darkTheme, 0.5)).toMatch(/rgba/i);
        expect(stickyEdgeShadow(darkTheme, "left")).toContain("inset -1px");
    });

    it("styles dialogs and tooltips without light-only assumptions", () => {
        const dark = createTheme(darkTheme);
        expect(dark.components?.MuiDialog?.styleOverrides).toBeTruthy();
        expect(dark.components?.MuiTooltip?.styleOverrides).toBeTruthy();
        expect(dark.components?.MuiChip?.styleOverrides).toBeTruthy();
        expect(dark.palette.action.disabledBackground).toBeTruthy();
    });
});
