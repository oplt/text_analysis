import { Button, Chip, Paper, Stack, Typography } from "@mui/material";
import { OpenInNew as OpenInNewIcon } from "@mui/icons-material";
import type { ReactNode } from "react";

type ObservabilityShortcutCardProps = {
    title: string;
    description: string;
    buttonText: string;
    url: string | null;
    allowed: boolean;
    configured: boolean;
    onOpen: (url: string) => void;
    icon: ReactNode;
};

export function ObservabilityShortcutCard({
    title,
    description,
    buttonText,
    url,
    allowed,
    configured,
    onOpen,
    icon,
}: ObservabilityShortcutCardProps) {
    const disabledReason = !allowed ? "Admin only" : !configured || !url ? "Not configured" : null;

    return (
        <Paper
            sx={(theme) => ({
                p: 2.5,
                borderRadius: 1,
                border: "none",
                boxShadow: "none",
                minHeight: 218,
                display: "flex",
                backgroundColor:
                    theme.palette.mode === "dark" ? theme.palette.background.paper : "var(--tesla-light-ash)",
            })}
        >
            <Stack spacing={2} sx={{ width: "100%" }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1.5}>
                    <Typography variant="h6">{title}</Typography>
                    {icon}
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    {description}
                </Typography>
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1.5}>
                    {disabledReason ? (
                        <Chip label={disabledReason} size="small" variant="outlined" />
                    ) : (
                        <span />
                    )}
                    <Button
                        variant="contained"
                        endIcon={<OpenInNewIcon />}
                        disabled={Boolean(disabledReason) || !url}
                        onClick={() => {
                            if (url) {
                                onOpen(url);
                            }
                        }}
                    >
                        {buttonText}
                    </Button>
                </Stack>
            </Stack>
        </Paper>
    );
}
