import { Box, Stack, Typography } from "@mui/material";
import { layoutSpacing } from "./layoutTokens";

type EmptyStateProps = {
    icon: React.ReactNode;
    title: string;
    description: string;
    action?: React.ReactNode;
};

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
    return (
        <Box
            sx={{
                px: layoutSpacing.cardPadding.default,
                py: { xs: 3, md: 4 },
                textAlign: "center",
            }}
        >
            <Stack spacing={1.5} alignItems="center">
                <Box
                    sx={{
                        width: 48,
                        height: 48,
                        display: "grid",
                        placeItems: "center",
                        color: "text.secondary",
                    }}
                >
                    {icon}
                </Box>
                <Typography variant="h6" component="h2">
                    {title}
                </Typography>
                <Typography color="text.secondary" sx={{ maxWidth: 460 }}>
                    {description}
                </Typography>
                {action && <Box sx={{ pt: 0.5 }}>{action}</Box>}
            </Stack>
        </Box>
    );
}
