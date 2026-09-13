import type { ReactNode } from "react";
import {
    Alert,
    Box,
    Button,
    LinearProgress,
    Stack,
    Typography,
} from "@mui/material";
import { Replay as ReplayIcon, Refresh as RetryIcon } from "@mui/icons-material";
import { KeyValueList } from "./KeyValueList";
import { RunStatusChip } from "./RunStatusChip";
import {
    formatRunDurationMs,
    formatRunTimestamp,
    isActiveCanonicalStatus,
    runDurationMs,
} from "./runStatusModel";

export type RunStatusPanelProps = {
    status: string | null | undefined;
    title?: string;
    runId?: string | null;
    /** 0–100 when known; indeterminate while active if omitted. */
    progress?: number | null;
    stage?: string | null;
    startedAt?: string | null;
    completedAt?: string | null;
    createdAt?: string | null;
    /** Optional explicit duration (e.g. agent latency_ms). */
    latencyMs?: number | null;
    errorMessage?: string | null;
    onRetry?: () => void;
    onReplay?: () => void;
    retryLabel?: string;
    replayLabel?: string;
    retryDisabled?: boolean;
    replayDisabled?: boolean;
    replayReason?: string | null;
    /** Extra actions (e.g. cancel). */
    actions?: ReactNode;
    dense?: boolean;
};

/**
 * Shared long-running job panel: status, progress, stage, times, failure, retry/replay.
 */
export function RunStatusPanel({
    status,
    title = "Run status",
    runId,
    progress,
    stage,
    startedAt,
    completedAt,
    createdAt,
    latencyMs,
    errorMessage,
    onRetry,
    onReplay,
    retryLabel = "Retry",
    replayLabel = "Replay",
    retryDisabled = false,
    replayDisabled = false,
    replayReason = null,
    actions,
    dense = false,
}: RunStatusPanelProps) {
    const active = isActiveCanonicalStatus(status);
    const duration = runDurationMs({
        startedAt,
        completedAt,
        createdAt,
        latencyMs,
        nowMs: active ? Date.now() : undefined,
    });
    const startDisplay = formatRunTimestamp(startedAt ?? createdAt);
    const hasProgress = typeof progress === "number" && Number.isFinite(progress);

    return (
        <Stack
            spacing={dense ? 1 : 1.5}
            role="status"
            aria-live={active ? "polite" : "off"}
            aria-atomic="false"
        >
            <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={1}
                alignItems={{ sm: "center" }}
                justifyContent="space-between"
            >
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant={dense ? "subtitle2" : "subtitle1"} component="h3">
                        {title}
                    </Typography>
                    <RunStatusChip status={status} />
                    {runId ? (
                        <Typography variant="caption" color="text.secondary">
                            {runId}
                        </Typography>
                    ) : null}
                </Stack>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                    {actions}
                    {onRetry ? (
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={<RetryIcon />}
                            disabled={retryDisabled || active}
                            onClick={onRetry}
                        >
                            {retryLabel}
                        </Button>
                    ) : null}
                    {onReplay ? (
                        <Button
                            size="small"
                            variant="outlined"
                            startIcon={<ReplayIcon />}
                            disabled={replayDisabled || active}
                            onClick={onReplay}
                            title={replayReason ?? undefined}
                        >
                            {replayLabel}
                        </Button>
                    ) : null}
                </Stack>
            </Stack>

            {active ? (
                <Box>
                    <LinearProgress
                        variant={hasProgress ? "determinate" : "indeterminate"}
                        value={hasProgress ? Math.max(0, Math.min(100, progress)) : undefined}
                        aria-label={
                            hasProgress
                                ? `Run progress ${Math.round(progress)} percent`
                                : "Run in progress"
                        }
                    />
                    {hasProgress ? (
                        <Typography variant="caption" color="text.secondary">
                            {Math.round(progress)}%
                        </Typography>
                    ) : null}
                </Box>
            ) : null}

            <KeyValueList
                dense
                items={[
                    {
                        key: "stage",
                        label: "Current stage",
                        value: stage?.trim() || (active ? "In progress" : "—"),
                    },
                    {
                        key: "started",
                        label: "Started",
                        value: startDisplay,
                    },
                    {
                        key: "duration",
                        label: "Duration",
                        value: formatRunDurationMs(duration),
                    },
                    {
                        key: "completed",
                        label: "Completed",
                        value: formatRunTimestamp(completedAt),
                    },
                ]}
            />

            {errorMessage ? <Alert severity="error">{errorMessage}</Alert> : null}
            {replayReason && onReplay && replayDisabled ? (
                <Typography variant="caption" color="text.secondary">
                    {replayReason}
                </Typography>
            ) : null}
        </Stack>
    );
}
