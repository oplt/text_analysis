import { Box, Paper, Stack, Typography, type SxProps, type Theme } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { colors, radii } from "../../app/designTokens";

type SectionCardVariant = "default" | "subtle" | "outlined" | "flat";

type SectionCardProps = {
    title?: React.ReactNode;
    description?: React.ReactNode;
    action?: React.ReactNode;
    children: React.ReactNode;
    variant?: SectionCardVariant;
    compact?: boolean;
    sx?: SxProps<Theme>;
    contentSx?: SxProps<Theme>;
};

function variantSx(variant: SectionCardVariant, theme: Theme): SxProps<Theme> {
    const isDark = theme.palette.mode === "dark";
    switch (variant) {
        case "subtle":
            return {
                border: "none",
                boxShadow: "none",
                backgroundColor: isDark
                    ? alpha(theme.palette.common.white, 0.03)
                    : colors.lightAsh,
            };
        case "outlined":
            return {
                border: `1px solid ${theme.palette.divider}`,
                boxShadow: "none",
                backgroundColor: "transparent",
            };
        case "flat":
            return {
                border: "none",
                boxShadow: "none",
                backgroundColor: "transparent",
                p: 0,
            };
        case "default":
        default:
            return {
                border: "none",
                boxShadow: "none",
                backgroundColor: isDark ? theme.palette.background.paper : colors.lightAsh,
            };
    }
}

export function SectionCard({
    title,
    description,
    action,
    children,
    variant = "default",
    compact = false,
    sx,
    contentSx,
}: SectionCardProps) {
    const padding = variant === "flat" ? 0 : compact ? { xs: 1.5, md: 2 } : { xs: 2.5, md: 3 };

    return (
        <Paper
            sx={[
                (theme) => ({
                    p: padding,
                    borderRadius: variant === "flat" ? 0 : `${radii.card}px`,
                    ...variantSx(variant, theme),
                }),
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            {(title || description || action) && (
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    justifyContent="space-between"
                    alignItems={{ xs: "flex-start", sm: "flex-start" }}
                    spacing={compact ? 1 : 2}
                    sx={{ mb: compact ? 1.5 : 2.5 }}
                >
                    <Box sx={{ minWidth: 0, flex: 1 }}>
                        {title && (
                            <Typography
                                variant={compact ? "subtitle1" : "h5"}
                                sx={{ mb: description ? 0.5 : 0 }}
                            >
                                {title}
                            </Typography>
                        )}
                        {description && (
                            <Typography
                                variant="body2"
                                color="text.secondary"
                                sx={{ maxWidth: 720 }}
                            >
                                {description}
                            </Typography>
                        )}
                    </Box>
                    {action ? (
                        <Box
                            sx={{
                                flexShrink: 0,
                                width: { xs: "100%", sm: "auto" },
                                "& > *": { maxWidth: "100%" },
                            }}
                        >
                            {action}
                        </Box>
                    ) : null}
                </Stack>
            )}
            <Box
                sx={[
                    ...(Array.isArray(contentSx) ? contentSx : contentSx ? [contentSx] : []),
                ]}
            >
                {children}
            </Box>
        </Paper>
    );
}
