import {
    Box,
    Button,
    Chip,
    Stack,
    Tab,
    Tabs,
    Tooltip,
    Typography,
} from "@mui/material";
import {
    CheckCircle as CompleteIcon,
    ErrorOutline as BlockedIcon,
    RadioButtonUnchecked as IncompleteIcon,
    WarningAmber as WarningIcon,
    Circle as CurrentIcon,
} from "@mui/icons-material";
import { motion, radii } from "../../../app/designTokens";
import type { WorkflowStageState, WorkflowStatus } from "../workflow";

const STATUS_LABEL: Record<WorkflowStatus, string> = {
    complete: "Complete",
    current: "Current",
    incomplete: "Incomplete",
    blocked: "Blocked",
    warning: "In progress",
};

function StatusIcon({ status }: { status: WorkflowStatus }) {
    const fontSize = "small" as const;
    switch (status) {
        case "complete":
            return <CompleteIcon color="success" fontSize={fontSize} />;
        case "current":
            return <CurrentIcon color="primary" fontSize={fontSize} />;
        case "warning":
            return <WarningIcon color="warning" fontSize={fontSize} />;
        case "blocked":
            return <BlockedIcon color="disabled" fontSize={fontSize} />;
        default:
            return <IncompleteIcon color="action" fontSize={fontSize} />;
    }
}

function statusAccent(status: WorkflowStatus): string {
    switch (status) {
        case "complete":
            return "success.main";
        case "current":
            return "primary.main";
        case "warning":
            return "warning.main";
        case "blocked":
            return "text.disabled";
        default:
            return "text.secondary";
    }
}

type ResearchWorkflowNavigatorProps = {
    stages: WorkflowStageState[];
    activeStageId: string | false;
    onSelectStage: (stage: WorkflowStageState) => void;
    onOpenDashboard: () => void;
    dashboardSelected: boolean;
    orientation?: "horizontal" | "vertical";
};

export function ResearchWorkflowNavigator({
    stages,
    activeStageId,
    onSelectStage,
    onOpenDashboard,
    dashboardSelected,
    orientation = "horizontal",
}: ResearchWorkflowNavigatorProps) {
    return (
        <Box>
            <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={1}
                alignItems={{ xs: "stretch", sm: "center" }}
                justifyContent="space-between"
                sx={{ mb: 1.5 }}
            >
                <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        Research workflow
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Follow the stages in order. Status and counts update from the active corpus.
                    </Typography>
                </Box>
                <Button
                    size="small"
                    variant={dashboardSelected ? "contained" : "outlined"}
                    onClick={onOpenDashboard}
                    sx={{ alignSelf: { xs: "stretch", sm: "center" } }}
                >
                    Overview
                </Button>
            </Stack>

            <Tabs
                orientation={orientation}
                value={activeStageId === false ? false : activeStageId}
                onChange={(_, value: string) => {
                    const stage = stages.find((item) => item.id === value);
                    if (stage) onSelectStage(stage);
                }}
                variant={orientation === "horizontal" ? "scrollable" : "standard"}
                scrollButtons="auto"
                allowScrollButtonsMobile
                sx={{
                    minHeight: orientation === "horizontal" ? 72 : 44,
                    "& .MuiTabs-indicator": {
                        height: 3,
                        borderRadius: `${radii.button}px`,
                        transition: `all ${motion.durationMs}ms ${motion.easing}`,
                    },
                    "& .MuiTab-root": {
                        minHeight: orientation === "horizontal" ? 72 : 44,
                        textTransform: "none",
                        alignItems: "stretch",
                        opacity: 1,
                        alignSelf: orientation === "vertical" ? "stretch" : undefined,
                    },
                }}
            >
                {stages.map((stage, index) => {
                    const tooltip = stage.blockedReason ?? stage.detail ?? stage.description;
                    return (
                        <Tab
                            key={stage.id}
                            value={stage.id}
                            aria-label={stage.label}
                            label={
                                <Tooltip title={tooltip} placement="top">
                                    <Stack
                                        spacing={0.5}
                                        alignItems="flex-start"
                                        sx={{
                                            py: 0.5,
                                            minWidth: orientation === "horizontal" ? 108 : 0,
                                            maxWidth: orientation === "horizontal" ? 140 : "100%",
                                            color: statusAccent(stage.status),
                                        }}
                                    >
                                        <Stack direction="row" spacing={0.75} alignItems="center">
                                            <Typography
                                                variant="caption"
                                                color="text.secondary"
                                                sx={{ fontVariantNumeric: "tabular-nums" }}
                                            >
                                                {String(index + 1).padStart(2, "0")}
                                            </Typography>
                                            <StatusIcon status={stage.status} />
                                            <Typography
                                                variant="body2"
                                                sx={{
                                                    fontWeight: stage.status === "current" ? 700 : 600,
                                                    color:
                                                        stage.status === "blocked"
                                                            ? "text.disabled"
                                                            : "text.primary",
                                                }}
                                            >
                                                {stage.label}
                                            </Typography>
                                        </Stack>
                                        <Chip
                                            size="small"
                                            label={stage.detail ?? STATUS_LABEL[stage.status]}
                                            variant="outlined"
                                            sx={{
                                                height: 22,
                                                maxWidth: "100%",
                                                borderColor:
                                                    stage.status === "current"
                                                        ? "primary.main"
                                                        : undefined,
                                                "& .MuiChip-label": {
                                                    px: 0.75,
                                                    fontSize: 11,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                },
                                            }}
                                        />
                                    </Stack>
                                </Tooltip>
                            }
                        />
                    );
                })}
            </Tabs>
        </Box>
    );
}
