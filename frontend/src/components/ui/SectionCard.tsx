import { Box, Paper, Stack, Typography, type SxProps, type Theme } from "@mui/material";
import { radii } from "../../app/designTokens";
import { headingHierarchy, layoutSpacing } from "./layoutTokens";
import { surfaceCard, surfaceMuted } from "./themeSurfaces";

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
    switch (variant) {
        case "subtle":
            return {
                border: "none",
                boxShadow: "none",
                backgroundColor: surfaceMuted(theme),
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
                backgroundColor: surfaceCard(theme),
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
    const padding =
        variant === "flat"
            ? 0
            : compact
              ? layoutSpacing.cardPadding.compact
              : layoutSpacing.cardPadding.default;

    return (
        <Paper
            sx={[
                (theme) => ({
                    p: padding,
                    borderRadius: variant === "flat" ? 0 : `${radii.card}px`,
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
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
                    sx={{
                        mb: compact
                            ? layoutSpacing.headerMarginBottom.compact
                            : layoutSpacing.headerMarginBottom.default + 0.5,
                        flexShrink: 0,
                    }}
                >
                    <Box sx={{ minWidth: 0, flex: 1 }}>
                        {title && (
                            <Typography
                                variant={
                                    compact
                                        ? headingHierarchy.sectionCompact
                                        : headingHierarchy.section
                                }
                                component="h2"
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
                                display: "flex",
                                justifyContent: { xs: "flex-start", sm: "flex-end" },
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
                    { flex: 1, minHeight: 0 },
                    ...(Array.isArray(contentSx) ? contentSx : contentSx ? [contentSx] : []),
                ]}
            >
                {children}
            </Box>
        </Paper>
    );
}
