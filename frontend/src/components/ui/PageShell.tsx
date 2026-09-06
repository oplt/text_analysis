import { Box, Container, Stack, type Breakpoint, type SxProps, type Theme } from "@mui/material";

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

    return (
        <Box
            sx={[
                {
                    position: "relative",
                    px: { xs: 2, md: dense ? 2.5 : 3 },
                    py: { xs: dense ? 2 : 2.5, md: dense ? 2.5 : 4 },
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Container maxWidth={resolvedMaxWidth} sx={{ px: "0 !important" }}>
                <Stack spacing={dense ? { xs: 2, md: 2.5 } : { xs: 3, md: 4 }}>{children}</Stack>
            </Container>
        </Box>
    );
}
