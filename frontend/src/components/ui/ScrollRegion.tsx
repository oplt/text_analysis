import { Box, type BoxProps } from "@mui/material";
import { scrollContainerSx } from "./responsiveQa";

type ScrollRegionProps = BoxProps;

/**
 * Horizontal scroll wrapper for dense tables/charts (Phase 22/24).
 * Prefer this over repeating `overflowX: "auto"` inline.
 */
export function ScrollRegion({ sx, ...props }: ScrollRegionProps) {
    return (
        <Box
            {...props}
            sx={[
                scrollContainerSx,
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        />
    );
}
