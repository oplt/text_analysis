import { Box, Paper, Stack, Typography, type SxProps, type Theme } from "@mui/material";
import { colors, radii } from "../../app/designTokens";

type SectionCardProps = {
    title?: React.ReactNode;
    description?: React.ReactNode;
    action?: React.ReactNode;
    children: React.ReactNode;
    sx?: SxProps<Theme>;
    contentSx?: SxProps<Theme>;
};

export function SectionCard({
    title,
    description,
    action,
    children,
    sx,
    contentSx,
}: SectionCardProps) {
    return (
        <Paper
            sx={[
                (theme) => ({
                    p: { xs: 2.5, md: 3 },
                    borderRadius: `${radii.card}px`,
                    border: "none",
                    boxShadow: "none",
                    backgroundColor:
                        theme.palette.mode === "dark" ? theme.palette.background.paper : colors.lightAsh,
                }),
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            {(title || description || action) && (
                <Stack
                    direction={{ xs: "column", sm: "row" }}
                    justifyContent="space-between"
                    alignItems={{ xs: "flex-start", sm: "flex-start" }}
                    spacing={2}
                    sx={{ mb: 2.5 }}
                >
                    <Box>
                        {title && (
                            <Typography variant="h5" sx={{ mb: description ? 0.5 : 0 }}>
                                {title}
                            </Typography>
                        )}
                        {description && (
                            <Typography variant="body2" color="text.secondary">
                                {description}
                            </Typography>
                        )}
                    </Box>
                    {action}
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
