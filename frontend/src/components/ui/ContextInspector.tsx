import { Box, IconButton, Stack, Typography, type SxProps, type Theme } from "@mui/material";
import { Close as CloseIcon } from "@mui/icons-material";
import { SectionCard } from "./SectionCard";
import { layoutSpacing } from "./layoutTokens";

type ContextInspectorProps = {
    title?: React.ReactNode;
    description?: React.ReactNode;
    children: React.ReactNode;
    /** When true, show a close control (drawer mode). */
    dismissible?: boolean;
    onDismiss?: () => void;
    /** Sticky aside for desktop layouts. */
    sticky?: boolean;
    width?: number | { lg?: number; xl?: number };
    sx?: SxProps<Theme>;
};

/**
 * Reusable right-side contextual panel (Ask Corpus, evidence, help).
 */
export function ContextInspector({
    title = "Context",
    description,
    children,
    dismissible = false,
    onDismiss,
    sticky = true,
    width,
    sx,
}: ContextInspectorProps) {
    return (
        <Box
            component="aside"
            aria-label={typeof title === "string" ? title : "Contextual panel"}
            sx={[
                {
                    position: sticky ? "sticky" : "static",
                    top: sticky ? 16 : undefined,
                    width: width ?? { lg: 360 },
                    maxWidth: "100%",
                    minWidth: 0,
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <SectionCard
                title={title}
                description={description}
                variant="subtle"
                compact
                sx={{ mt: 0 }}
                action={
                    dismissible ? (
                        <IconButton
                            aria-label="Close contextual panel"
                            size="small"
                            onClick={onDismiss}
                        >
                            <CloseIcon fontSize="small" />
                        </IconButton>
                    ) : undefined
                }
                contentSx={{ pt: 0 }}
            >
                <Stack spacing={layoutSpacing.sectionGap / 2}>{children}</Stack>
            </SectionCard>
        </Box>
    );
}

/** Compact title-only strip for inspector headers without SectionCard chrome. */
export function ContextInspectorHeader({
    title,
    description,
}: {
    title: React.ReactNode;
    description?: React.ReactNode;
}) {
    return (
        <Box sx={{ mb: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                {title}
            </Typography>
            {description ? (
                <Typography variant="body2" color="text.secondary">
                    {description}
                </Typography>
            ) : null}
        </Box>
    );
}
