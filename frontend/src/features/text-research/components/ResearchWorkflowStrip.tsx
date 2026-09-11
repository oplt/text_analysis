import {
    Box,
    LinearProgress,
    Stack,
    Tooltip,
    Typography,
    useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { ExpandMore as ExpandIcon } from "@mui/icons-material";
import {
    compactStageDetail,
    statusAccent,
    workflowProgress,
} from "../workflowDisplay";
import { StatusIcon } from "../StatusIcon";
import type { WorkflowStageState } from "../workflow";

type ResearchWorkflowStripProps = {
    stages: WorkflowStageState[];
    activeStageId: string | false;
    onSelectStage: (stage: WorkflowStageState) => void;
    onOpenWorkflow: () => void;
};

function StageStep({
    stage,
    selected,
    showDetail,
    showConnector,
    onSelect,
}: {
    stage: WorkflowStageState;
    selected: boolean;
    showDetail: boolean;
    showConnector: boolean;
    onSelect: () => void;
}) {
    const detail = showDetail ? compactStageDetail(stage) : null;
    const tooltip = stage.blockedReason ?? stage.detail ?? stage.description;
    const disabled = stage.status === "blocked";

    return (
        <Stack direction="row" alignItems="center" sx={{ flexShrink: 0 }}>
            <Tooltip title={tooltip} placement="bottom">
                <Box
                    component="button"
                    type="button"
                    onClick={onSelect}
                    aria-current={selected ? "step" : undefined}
                    aria-label={`${stage.label}${detail ? ` ${detail}` : ""} — ${stage.status}`}
                    sx={{
                        appearance: "none",
                        border: 0,
                        background: "transparent",
                        cursor: disabled ? "not-allowed" : "pointer",
                        opacity: disabled ? 0.55 : 1,
                        px: 0.25,
                        py: 0.25,
                        borderRadius: 1,
                        minHeight: 40,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 0,
                        color: statusAccent(stage.status),
                        outlineOffset: 2,
                        bgcolor: selected ? "action.selected" : "transparent",
                        "&:hover": disabled
                            ? undefined
                            : { bgcolor: "action.hover" },
                    }}
                >
                    <Stack direction="row" spacing={0.2} alignItems="center">
                        <Box sx={{ display: "flex", "& .MuiSvgIcon-root": { fontSize: 16 } }}>
                            <StatusIcon status={stage.status} size="inherit" />
                        </Box>
                        <Typography
                            variant="body2"
                            sx={{
                                fontSize: { md: 12.5, xl: 13 },
                                fontWeight: selected || stage.status === "current" ? 700 : 600,
                                color:
                                    stage.status === "blocked"
                                        ? "text.disabled"
                                        : "text.primary",
                                whiteSpace: "nowrap",
                                lineHeight: 1.2,
                            }}
                        >
                            {stage.label}
                        </Typography>
                    </Stack>
                    {detail ? (
                        <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ fontSize: 10, lineHeight: 1.15 }}
                        >
                            {detail}
                        </Typography>
                    ) : null}
                </Box>
            </Tooltip>
            {showConnector ? (
                <Box
                    aria-hidden
                    sx={{
                        width: { md: 4, xl: 6 },
                        height: 2,
                        mx: 0,
                        borderRadius: 1,
                        bgcolor: "divider",
                        flexShrink: 0,
                    }}
                />
            ) : null}
        </Stack>
    );
}

export function ResearchWorkflowStrip({
    stages,
    activeStageId,
    onSelectStage,
    onOpenWorkflow,
}: ResearchWorkflowStripProps) {
    const theme = useTheme();
    const isLarge = useMediaQuery(theme.breakpoints.up("xl"));
    const isMedium = useMediaQuery(theme.breakpoints.up("md"));
    const progress = workflowProgress(stages);

    if (!isMedium) {
        return (
            <Box
                component="button"
                type="button"
                onClick={onOpenWorkflow}
                aria-label="Open research workflow"
                sx={{
                    appearance: "none",
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 2,
                    width: "100%",
                    textAlign: "left",
                    cursor: "pointer",
                    bgcolor: "background.paper",
                    px: 1.5,
                    py: 1.25,
                    minHeight: 64,
                    "&:hover": { bgcolor: "action.hover" },
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                    <Box sx={{ minWidth: 0, flex: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                            Research workflow
                        </Typography>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                            Step {progress.currentIndex + 1} of {progress.total}
                            {progress.current ? ` · ${progress.current.label}` : ""}
                        </Typography>
                        <LinearProgress
                            variant="determinate"
                            value={progress.percent}
                            sx={{ mt: 1, height: 6, borderRadius: 1 }}
                        />
                    </Box>
                    <ExpandIcon color="action" />
                </Stack>
            </Box>
        );
    }

    const showDetail = isLarge;
    // On laptop widths, keep the strip compact: show all stages but hide secondary lines.
    return (
        <Box
            component="nav"
            aria-label="Research workflow"
            sx={{
                position: "sticky",
                top: 0,
                zIndex: 2,
                display: "flex",
                alignItems: "center",
                gap: 0.5,
                minHeight: 48,
                px: 0.5,
                py: 0.25,
                borderRadius: 2,
                border: 1,
                borderColor: "divider",
                bgcolor: (t) =>
                    t.palette.mode === "dark"
                        ? "background.paper"
                        : "background.default",
            }}
        >
            <Box
                sx={{
                    flex: 1,
                    minWidth: 0,
                    overflowX: "auto",
                    display: "flex",
                    alignItems: "center",
                    py: 0.25,
                    scrollbarWidth: "thin",
                }}
            >
                {stages.map((stage, index) => (
                    <StageStep
                        key={stage.id}
                        stage={stage}
                        selected={activeStageId === stage.id}
                        showDetail={showDetail && (stage.status === "warning" || stage.status === "current")}
                        showConnector={index < stages.length - 1}
                        onSelect={() => onSelectStage(stage)}
                    />
                ))}
            </Box>
        </Box>
    );
}
