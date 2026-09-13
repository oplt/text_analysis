import { Chip, type ChipProps } from "@mui/material";
import { runStatusAriaLabel } from "../../app/a11y";
import { runStatusLabel, runStatusTone } from "./runStatusModel";

type RunStatusChipProps = {
    status: string | null | undefined;
    size?: ChipProps["size"];
    /** Show raw backend status in tooltip title via Chip label suffix when different. */
    showRaw?: boolean;
};

/**
 * Unified status chip for long-running jobs (Phase 16).
 * Label text carries meaning; color is secondary (not color-alone).
 */
export function RunStatusChip({ status, size = "small", showRaw = false }: RunStatusChipProps) {
    const label = runStatusLabel(status);
    const tone = runStatusTone(status);
    const raw = status?.trim() ?? "";
    const display =
        showRaw && raw && raw.toLowerCase() !== label.toLowerCase()
            ? `${label} (${raw})`
            : label;
    return (
        <Chip
            size={size}
            color={tone}
            label={display}
            variant="outlined"
            aria-label={runStatusAriaLabel(display)}
        />
    );
}
