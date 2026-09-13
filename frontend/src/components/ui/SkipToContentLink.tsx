import { Box, Link } from "@mui/material";

/**
 * First focusable control for keyboard users (Phase 21).
 * Visually hidden until focused.
 */
export function SkipToContentLink({
    href = "#main-content",
    label = "Skip to main content",
}: {
    href?: string;
    label?: string;
}) {
    return (
        <Box
            component={Link}
            href={href}
            underline="none"
            sx={{
                position: "absolute",
                left: 16,
                top: 16,
                zIndex: (theme) => theme.zIndex.tooltip + 1,
                px: 1.5,
                py: 1,
                borderRadius: 1,
                bgcolor: "background.paper",
                color: "text.primary",
                border: 1,
                borderColor: "divider",
                boxShadow: 2,
                transform: "translateY(-160%)",
                transition: (theme) =>
                    theme.transitions.create("transform", {
                        duration: theme.transitions.duration.shorter,
                    }),
                "&:focus": {
                    transform: "translateY(0)",
                    outline: "none",
                },
                "&:focus-visible": {
                    transform: "translateY(0)",
                    boxShadow: (theme) =>
                        `0 0 0 2px ${theme.palette.background.paper}, 0 0 0 4px ${theme.palette.primary.main}`,
                },
            }}
        >
            {label}
        </Box>
    );
}
