import { Tooltip, Typography, type TypographyProps } from "@mui/material";
import { dataTableDefaults } from "./dataTableTokens";

type TruncatedCellProps = {
    children: string | number | null | undefined;
    /** Soft character budget before ellipsis; full value stays in tooltip. */
    maxChars?: number;
    emptyLabel?: string;
    variant?: TypographyProps["variant"];
    component?: TypographyProps["component"];
};

/**
 * Keeps long cell text from blowing out table layout; full value via tooltip.
 */
export function TruncatedCell({
    children,
    maxChars = dataTableDefaults.truncateChars,
    emptyLabel = "—",
    variant = "body2",
    component = "span",
}: TruncatedCellProps) {
    if (children == null || children === "") {
        return (
            <Typography variant={variant} component={component} color="text.secondary">
                {emptyLabel}
            </Typography>
        );
    }
    const full = String(children);
    if (full.length <= maxChars) {
        return (
            <Typography variant={variant} component={component} noWrap>
                {full}
            </Typography>
        );
    }
    const preview = `${full.slice(0, Math.max(1, maxChars - 1))}…`;
    return (
        <Tooltip title={full} enterDelay={400}>
            <Typography
                variant={variant}
                component={component}
                noWrap
                sx={{ maxWidth: "100%", display: "block", cursor: "help" }}
            >
                {preview}
            </Typography>
        </Tooltip>
    );
}
