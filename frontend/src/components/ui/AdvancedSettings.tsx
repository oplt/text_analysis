import { useId } from "react";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Typography,
    type SxProps,
    type Theme,
} from "@mui/material";
import { ExpandMore as ExpandIcon } from "@mui/icons-material";

type AdvancedSettingsProps = {
    children: React.ReactNode;
    title?: string;
    description?: string;
    defaultExpanded?: boolean;
    disabled?: boolean;
    sx?: SxProps<Theme>;
};

/**
 * Collapsed-by-default accordion for infrequently changed options.
 */
export function AdvancedSettings({
    children,
    title = "Advanced settings",
    description,
    defaultExpanded = false,
    disabled = false,
    sx,
}: AdvancedSettingsProps) {
    const reactId = useId();
    const headerId = `${reactId}-header`;
    const contentId = `${reactId}-content`;

    return (
        <Accordion
            disableGutters
            elevation={0}
            defaultExpanded={defaultExpanded}
            disabled={disabled}
            sx={[
                {
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    "&:before": { display: "none" },
                    bgcolor: "transparent",
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            <AccordionSummary
                expandIcon={<ExpandIcon />}
                aria-controls={contentId}
                id={headerId}
                sx={{ minHeight: 48, px: 1.5 }}
            >
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {title}
                </Typography>
                {description ? (
                    <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ ml: 1.5, alignSelf: "center" }}
                    >
                        {description}
                    </Typography>
                ) : null}
            </AccordionSummary>
            <AccordionDetails id={contentId} sx={{ px: 1.5, pt: 0, pb: 1.5 }}>
                {children}
            </AccordionDetails>
        </Accordion>
    );
}
