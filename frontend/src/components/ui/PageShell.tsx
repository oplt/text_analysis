import { Box, Container, Stack, type Breakpoint, type SxProps, type Theme } from "@mui/material";
import { layoutSpacing } from "./layoutTokens";

export type PageWidth = "compact" | "readable" | "default" | "wide" | "full";

const WIDTH_TO_BREAKPOINT: Record<PageWidth, Breakpoint | false> = {
    compact: "sm",
    readable: "md",
    default: "lg",
    wide: "xl",
    full: false,
};

type PageShellProps = {
    children: React.ReactNode;
    /** Prefer `width` for content-aware layouts; `maxWidth` kept for compatibility. */
    width?: PageWidth;
    maxWidth?: Breakpoint | false;
    dense?: boolean;
    sx?: SxProps<Theme>;
};

export function PageShell({
    children,
    width,
    maxWidth,
    dense = false,
    sx,
}: PageShellProps) {
    const resolvedMaxWidth =
        maxWidth !== undefined ? maxWidth : width ? WIDTH_TO_BREAKPOINT[width] : "xl";
    const padX = dense ? layoutSpacing.pagePaddingX.dense : layoutSpacing.pagePaddingX.default;
    const padY = dense ? layoutSpacing.pagePaddingY.dense : layoutSpacing.pagePaddingY.default;
    const gap = dense ? layoutSpacing.pageGap.dense : layoutSpacing.pageGap.default;

    return (
        <Box
            sx={[
                {
                    position: "relative",
                    px: padX,
                    py: padY,
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Container maxWidth={resolvedMaxWidth} sx={{ px: "0 !important" }}>
                <Stack spacing={gap}>{children}</Stack>
            </Container>
        </Box>
    );
}
