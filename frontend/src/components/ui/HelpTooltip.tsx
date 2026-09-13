import { useId, useState } from "react";
import {
    Box,
    IconButton,
    Popover,
    Stack,
    Tooltip,
    Typography,
    type SxProps,
    type Theme,
} from "@mui/material";
import {
    HelpOutline as HelpIcon,
    InfoOutlined as InfoIcon,
} from "@mui/icons-material";
import {
    getHelpTerm,
    type HelpTermId,
} from "../../config/helpText";

type HelpTooltipProps = {
    /** Registry term id. */
    termId?: HelpTermId | string;
    /** Inline override when no registry term (e.g. disabled reason). */
    title?: React.ReactNode;
    /** Longer body for popover when not using registry detail. */
    detail?: React.ReactNode;
    /**
     * `help` — subtle help icon (default).
     * `info` — info icon.
     * `label` — render children + icon as a field label row.
     */
    variant?: "help" | "info" | "label";
    /** Optional label text/node when variant is label (or beside icon). */
    children?: React.ReactNode;
    /** Prefer popover even when only a short definition exists. */
    forcePopover?: boolean;
    size?: "small" | "medium";
    sx?: SxProps<Theme>;
};

/**
 * Contextual help control (Phase 5).
 * Short definitions → tooltip; longer `detail` → popover "Learn more".
 */
export function HelpTooltip({
    termId,
    title,
    detail,
    variant = "help",
    children,
    forcePopover = false,
    size = "small",
    sx,
}: HelpTooltipProps) {
    const reactId = useId();
    const popoverId = `${reactId}-help`;
    const titleId = `${reactId}-title`;
    const [anchor, setAnchor] = useState<HTMLElement | null>(null);
    const term = termId ? getHelpTerm(termId) : null;
    const shortText = title ?? term?.definition ?? null;
    const longText = detail ?? term?.detail ?? null;
    const heading = term?.title;
    const hasPopover = Boolean(longText) || forcePopover;
    const open = Boolean(anchor);

    if (!shortText && !longText) {
        return variant === "label" ? <>{children}</> : null;
    }

    const iconFont = size === "small" ? "inherit" : "small";
    const Icon = variant === "info" ? InfoIcon : HelpIcon;

    const button = (
        <IconButton
            size="small"
            aria-label={heading ? `About ${heading}` : "About this term"}
            aria-haspopup={hasPopover ? "dialog" : undefined}
            aria-expanded={hasPopover ? open : undefined}
            aria-controls={hasPopover && open ? popoverId : undefined}
            onClick={
                hasPopover
                    ? (event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          setAnchor(event.currentTarget);
                      }
                    : undefined
            }
            sx={{ p: 0.25, color: "text.secondary", ...((sx as object) ?? {}) }}
        >
            <Icon fontSize={iconFont} />
        </IconButton>
    );

    const trigger = hasPopover ? (
        <Tooltip title={typeof shortText === "string" ? shortText : ""} arrow enterDelay={400}>
            <span>{button}</span>
        </Tooltip>
    ) : (
        <Tooltip title={shortText ?? ""} arrow enterDelay={300}>
            <span>{button}</span>
        </Tooltip>
    );

    const popover = hasPopover ? (
        <Popover
            id={popoverId}
            open={open}
            anchorEl={anchor}
            onClose={() => setAnchor(null)}
            anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
            transformOrigin={{ vertical: "top", horizontal: "left" }}
            slotProps={{
                paper: {
                    role: "dialog",
                    "aria-modal": false,
                    "aria-labelledby": heading ? titleId : undefined,
                    "aria-label": heading ? undefined : "Help details",
                    sx: { maxWidth: 360, p: 1.75 },
                },
            }}
        >
            <Stack spacing={1}>
                {heading ? (
                    <Typography id={titleId} variant="subtitle2" sx={{ fontWeight: 700 }}>
                        {heading}
                    </Typography>
                ) : null}
                {shortText ? (
                    <Typography variant="body2">{shortText}</Typography>
                ) : null}
                {longText ? (
                    <Typography variant="body2" color="text.secondary">
                        {longText}
                    </Typography>
                ) : null}
            </Stack>
        </Popover>
    ) : null;

    if (variant === "label") {
        return (
            <Box
                component="span"
                sx={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 0.35,
                    maxWidth: "100%",
                }}
            >
                <Box component="span" sx={{ minWidth: 0 }}>
                    {children}
                </Box>
                {trigger}
                {popover}
            </Box>
        );
    }

    return (
        <>
            {children}
            {trigger}
            {popover}
        </>
    );
}

/** Convenience: label node suitable for MUI TextField `label` prop. */
export function HelpFieldLabel({
    termId,
    children,
}: {
    termId: HelpTermId | string;
    children: React.ReactNode;
}) {
    return (
        <HelpTooltip termId={termId} variant="label">
            {children}
        </HelpTooltip>
    );
}
