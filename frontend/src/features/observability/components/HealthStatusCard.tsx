import { Box, Chip, Paper, Stack, Typography } from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import type { ReactNode } from "react";
import type { HealthStatus } from "../types";

const STATUS_LABELS: Record<HealthStatus, string> = {
    healthy: "Healthy",
    degraded: "Degraded",
    down: "Down",
    unknown: "Unknown",
    not_configured: "Not configured",
};

const STATUS_COLORS: Record<HealthStatus, "success" | "warning" | "error" | "default" | "info"> = {
    healthy: "success",
    degraded: "warning",
    down: "error",
    unknown: "default",
    not_configured: "info",
};

type HealthStatusCardProps = {
    title: string;
    status: HealthStatus;
    detail: string;
    metric?: string | null;
    lastCheckedAt?: string | null;
    icon: ReactNode;
};

function formatCheckedAt(value?: string | null) {
    if (!value) {
        return null;
    }
    return new Intl.DateTimeFormat(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    }).format(new Date(value));
}

export function HealthStatusCard({
    title,
    status,
    detail,
    metric,
    lastCheckedAt,
    icon,
}: HealthStatusCardProps) {
    const theme = useTheme();
    const paletteColor = STATUS_COLORS[status] === "default"
        ? theme.palette.text.secondary
        : theme.palette[STATUS_COLORS[status]].main;
    const checkedAt = formatCheckedAt(lastCheckedAt);

    return (
        <Paper
            sx={(theme) => ({
                p: 2,
                borderRadius: 1,
                border: "none",
                boxShadow: "none",
                minHeight: 180,
                backgroundColor:
                    theme.palette.mode === "dark" ? theme.palette.background.paper : "var(--tesla-light-ash)",
            })}
        >
            <Stack spacing={1.5} sx={{ height: "100%" }}>
                <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={1.5}>
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="subtitle2">{title}</Typography>
                        <Chip
                            label={STATUS_LABELS[status]}
                            color={STATUS_COLORS[status]}
                            size="small"
                            variant={status === "healthy" ? "filled" : "outlined"}
                            sx={{ mt: 1 }}
                        />
                    </Box>
                    <Box
                        sx={{
                            width: 38,
                            height: 38,
                            borderRadius: 3,
                            display: "grid",
                            placeItems: "center",
                            color: paletteColor,
                            bgcolor: alpha(paletteColor, theme.palette.mode === "dark" ? 0.18 : 0.1),
                            flex: "0 0 auto",
                        }}
                    >
                        {icon}
                    </Box>
                </Stack>
                {metric && (
                    <Typography variant="h6" sx={{ lineHeight: 1.2 }}>
                        {metric}
                    </Typography>
                )}
                <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    {detail}
                </Typography>
                {checkedAt && (
                    <Typography variant="caption" color="text.secondary">
                        Checked {checkedAt}
                    </Typography>
                )}
            </Stack>
        </Paper>
    );
}
