import { Box, Stack, Typography, type SxProps, type Theme } from "@mui/material";

type PageHeaderProps = {
    title: React.ReactNode;
    description?: React.ReactNode;
    actions?: React.ReactNode;
    icon?: React.ReactNode;
    dense?: boolean;
    sx?: SxProps<Theme>;
};

export function PageHeader({
    title,
    description,
    actions,
    icon,
    dense = false,
    sx,
}: PageHeaderProps) {
    return (
        <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={dense ? 1.5 : 2}
            justifyContent="space-between"
            alignItems={{ xs: "stretch", sm: "flex-start" }}
            sx={[
                { mb: dense ? 1.5 : 2 },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <Stack direction="row" spacing={1.5} alignItems="flex-start" sx={{ minWidth: 0 }}>
                {icon ? (
                    <Box sx={{ color: "primary.main", display: "flex", mt: 0.25 }}>{icon}</Box>
                ) : null}
                <Box sx={{ minWidth: 0 }}>
                    <Typography variant={dense ? "h5" : "h4"} component="h1">
                        {title}
                    </Typography>
                    {description ? (
                        <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ mt: 0.5, maxWidth: 720 }}
                        >
                            {description}
                        </Typography>
                    ) : null}
                </Box>
            </Stack>
            {actions ? (
                <Stack
                    direction="row"
                    spacing={1}
                    flexWrap="wrap"
                    useFlexGap
                    sx={{ flexShrink: 0 }}
                >
                    {actions}
                </Stack>
            ) : null}
        </Stack>
    );
}
