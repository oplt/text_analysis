import type { ReactElement } from "react";
import { Tooltip } from "@mui/material";

type DisabledWithReasonProps = {
    /** Explains why the control is disabled. Omit or null when enabled. */
    reason?: string | null;
    children: ReactElement;
};

/**
 * MUI disables pointer events on disabled buttons, so wrap with a span for tooltips.
 * Use for non-obvious disabled primary actions (Phase 20).
 */
export function DisabledWithReason({ reason, children }: DisabledWithReasonProps) {
    if (!reason) {
        return children;
    }
    return (
        <Tooltip title={reason} describeChild enterDelay={200}>
            <span style={{ display: "inline-flex", alignSelf: "flex-start" }}>{children}</span>
        </Tooltip>
    );
}
