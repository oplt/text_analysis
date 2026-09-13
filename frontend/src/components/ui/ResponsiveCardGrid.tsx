import { Box, type SxProps, type Theme } from "@mui/material";
import { layoutSpacing } from "./layoutTokens";
import { denseCardColumns } from "./responsiveQa";

export type ResponsiveCardColumns = 2 | 3 | 4;

type ResponsiveCardGridProps = {
    children: React.ReactNode;
    /** Desktop column count (tablet / mid-laptop stay at 2 until lg). */
    columns?: ResponsiveCardColumns;
    gap?: number;
    sx?: SxProps<Theme>;
};

/**
 * Responsive card / panel grid (Phase 22):
 * 1 column phone → 2 tablet/laptop → `columns` from lg (1200px)+.
 */
export function ResponsiveCardGrid({
    children,
    columns = 3,
    gap = layoutSpacing.gridGap,
    sx,
}: ResponsiveCardGridProps) {
    return (
        <Box
            sx={[
                {
                    display: "grid",
                    gap,
                    gridTemplateColumns: denseCardColumns(columns),
                    alignItems: "stretch",
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            {children}
        </Box>
    );
}
