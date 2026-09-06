import { Box, Container, Stack, type Breakpoint, type SxProps, type Theme } from "@mui/material";

type PageShellProps = {
    children: React.ReactNode;
    maxWidth?: Breakpoint | false;
    sx?: SxProps<Theme>;
};

export function PageShell({ children, maxWidth = "xl", sx }: PageShellProps) {
    return (
        <Box
            sx={[
                {
                    position: "relative",
                    px: { xs: 2, md: 3 },
                    py: { xs: 2.5, md: 4 },
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Container maxWidth={maxWidth} sx={{ px: "0 !important" }}>
                <Stack spacing={{ xs: 3, md: 4 }}>{children}</Stack>
            </Container>
        </Box>
    );
}
