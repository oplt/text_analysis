import { Box, Paper, Skeleton, Stack, Typography } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { colors, fonts, radii } from "../../app/designTokens";

type StatCardProps = {
    label: string;
    value: React.ReactNode;
    description?: React.ReactNode;
    icon: React.ReactNode;
    loading?: boolean;
    color?: "primary" | "secondary" | "success" | "warning" | "error" | "info";
};

export function StatCard({
    label,
    value,
    description,
    icon,
    loading = false,
    color = "primary",
}: StatCardProps) {
    return (
        <Paper
            sx={(theme) => ({
                p: 2.5,
                borderRadius: `${radii.card}px`,
                border: "none",
                minHeight: "100%",
                backgroundColor:
                    theme.palette.mode === "dark" ? theme.palette.background.paper : colors.lightAsh,
                boxShadow: "none",
            })}
        >
            <Stack spacing={2}>
                <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={2}>
                    <Typography variant="body2" color="text.secondary">
                        {label}
                    </Typography>
                    <Box
                        sx={(theme) => ({
                            width: 40,
                            height: 40,
                            display: "grid",
                            placeItems: "center",
                            borderRadius: `${radii.button}px`,
                            color: `${color}.main`,
                            backgroundColor: alpha(
                                theme.palette[color].main,
                                theme.palette.mode === "dark" ? 0.18 : 0.1
                            ),
                        })}
                    >
                        {icon}
                    </Box>
                </Stack>
                {loading ? (
                    <Stack spacing={0.75}>
                        <Skeleton variant="text" width={120} height={36} />
                        <Skeleton variant="text" width="70%" />
                    </Stack>
                ) : (
                    <Stack spacing={0.75}>
                        <Typography
                            variant="h4"
                            sx={{
                                fontFamily: fonts.display,
                            }}
                        >
                            {value}
                        </Typography>
                        {description && (
                            <Typography variant="body2" color="text.secondary">
                                {description}
                            </Typography>
                        )}
                    </Stack>
                )}
            </Stack>
        </Paper>
    );
}
