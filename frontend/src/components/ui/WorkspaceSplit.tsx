import { Box, type SxProps, type Theme } from "@mui/material";
import { layoutSpacing } from "./layoutTokens";

export type WorkspaceSplitRatio = "8-4" | "9-3" | "3-9" | "4-8";

type WorkspaceSplitProps = {
    main: React.ReactNode;
    side?: React.ReactNode;
    /** Default main workspace + inspector. */
    ratio?: WorkspaceSplitRatio;
    /** Breakpoint at which the side column appears. */
    sideFrom?: "md" | "lg" | "xl";
    gap?: number;
    sx?: SxProps<Theme>;
};

const RATIO_TRACKS: Record<WorkspaceSplitRatio, string> = {
    "8-4": "minmax(0, 2fr) minmax(280px, 1fr)",
    "9-3": "minmax(0, 3fr) minmax(240px, 1fr)",
    "3-9": "minmax(220px, 1fr) minmax(0, 3fr)",
    "4-8": "minmax(280px, 1fr) minmax(0, 2fr)",
};

/**
 * Main workspace + optional inspector/sidebar using semantic 8/4 (or similar) ratios.
 * Full width below `sideFrom`; side column from `sideFrom` upward (default lg = 1200px).
 * At 1024px Ask Corpus stays a drawer so the main pane is not crushed.
 */
export function WorkspaceSplit({
    main,
    side,
    ratio = "8-4",
    sideFrom = "lg",
    gap = layoutSpacing.sectionGap,
    sx,
}: WorkspaceSplitProps) {
    const showSide = Boolean(side);
    return (
        <Box
            sx={[
                {
                    display: "grid",
                    gap,
                    alignItems: "start",
                    width: "100%",
                    maxWidth: "100%",
                    minWidth: 0,
                    gridTemplateColumns: showSide
                        ? {
                              xs: "minmax(0, 1fr)",
                              [sideFrom]: RATIO_TRACKS[ratio],
                          }
                        : "minmax(0, 1fr)",
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Box component="section" sx={{ minWidth: 0, width: "100%", maxWidth: "100%" }}>
                {main}
            </Box>
            {showSide ? (
                <Box
                    component="aside"
                    sx={{
                        minWidth: 0,
                        width: "100%",
                        maxWidth: "100%",
                        display: { xs: "none", [sideFrom]: "block" },
                    }}
                >
                    {side}
                </Box>
            ) : null}
        </Box>
    );
}
