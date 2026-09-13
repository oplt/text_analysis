import {
    Box,
    Button,
    Divider,
    Drawer,
    IconButton,
    LinearProgress,
    Stack,
    Typography,
} from "@mui/material";
import { Close as CloseIcon } from "@mui/icons-material";
import { StatusIcon } from "../workflowDisplay";
import {
    STATUS_LABEL,
    stageActionLabel,
    statusAccent,
    workflowProgress,
} from "../workflowDisplayModel";
import {
    RESEARCH_NAV_GROUPS,
    type ResearchNavItem,
    type ResolvedResearchNav,
} from "../researchNavigation";
import type { WorkflowStageState } from "../workflow";

type ResearchWorkflowDrawerProps = {
    open: boolean;
    onClose: () => void;
    stages: WorkflowStageState[];
    activeStageId: string | false;
    onSelectStage: (stage: WorkflowStageState) => void;
    onNavigateItem: (item: ResearchNavItem) => void;
    resolved: ResolvedResearchNav | null;
    onOpenDashboard: () => void;
    dashboardSelected: boolean;
};

export function ResearchWorkflowDrawer({
    open,
    onClose,
    stages,
    activeStageId,
    onSelectStage,
    onNavigateItem,
    resolved,
    onOpenDashboard,
    dashboardSelected,
}: ResearchWorkflowDrawerProps) {
    const progress = workflowProgress(stages);
    const nextStage =
        stages.find((stage) => stage.status === "current" || stage.status === "warning") ??
        stages.find((stage) => stage.status === "incomplete");

    return (
        <Drawer
            anchor="right"
            open={open}
            onClose={onClose}
            PaperProps={{
                sx: {
                    width: { xs: "100%", sm: 400 },
                    maxWidth: "100%",
                    p: 2,
                },
            }}
        >
            <Stack
                direction="row"
                alignItems="flex-start"
                justifyContent="space-between"
                spacing={1}
                sx={{ mb: 2 }}
            >
                <Box>
                    <Typography variant="h6" component="h2">
                        Research navigation
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Grouped destinations plus pipeline progress for the active corpus.
                    </Typography>
                </Box>
                <IconButton aria-label="Close research navigation" size="small" onClick={onClose}>
                    <CloseIcon fontSize="small" />
                </IconButton>
            </Stack>

            <Box sx={{ mb: 2 }}>
                <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.75 }}>
                    <Typography variant="subtitle2">Pipeline progress</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {progress.completed}/{progress.total} · {progress.percent}%
                    </Typography>
                </Stack>
                <LinearProgress
                    variant="determinate"
                    value={progress.percent}
                    sx={{ height: 8, borderRadius: 1 }}
                />
                {nextStage ? (
                    <Button
                        fullWidth
                        size="small"
                        variant="contained"
                        sx={{ mt: 1.25 }}
                        onClick={() => {
                            onSelectStage(nextStage);
                            onClose();
                        }}
                    >
                        Next: {stageActionLabel(nextStage)}
                    </Button>
                ) : null}
            </Box>

            <Button
                fullWidth
                size="small"
                variant={dashboardSelected ? "contained" : "outlined"}
                onClick={() => {
                    onOpenDashboard();
                    onClose();
                }}
                sx={{ mb: 2 }}
            >
                Research overview
            </Button>

            <Typography variant="overline" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                Destinations by phase
            </Typography>

            <Stack spacing={2} sx={{ mb: 2.5 }}>
                {RESEARCH_NAV_GROUPS.map((group) => (
                    <Box key={group.id}>
                        <Typography variant="subtitle2" sx={{ mb: 0.75 }}>
                            {group.label}
                        </Typography>
                        <Stack spacing={0.5}>
                            {group.items.map((item) => {
                                const selected = resolved?.item.id === item.id;
                                return (
                                    <Button
                                        key={item.id}
                                        fullWidth
                                        size="small"
                                        variant={selected ? "contained" : "text"}
                                        onClick={() => {
                                            onNavigateItem(item);
                                            onClose();
                                        }}
                                        sx={{
                                            justifyContent: "flex-start",
                                            textTransform: "none",
                                            fontWeight: selected ? 700 : 500,
                                        }}
                                    >
                                        {item.label}
                                    </Button>
                                );
                            })}
                        </Stack>
                    </Box>
                ))}
            </Stack>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="overline" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                Workflow stages
            </Typography>

            <Stack spacing={1.25} component="ol" sx={{ m: 0, p: 0, listStyle: "none" }}>
                {stages.map((stage, index) => {
                    const selected = activeStageId === stage.id;
                    const showAction =
                        stage.status === "current" ||
                        stage.status === "warning" ||
                        stage.status === "incomplete";

                    return (
                        <Box
                            key={stage.id}
                            component="li"
                            sx={{
                                p: 1.25,
                                borderRadius: 2,
                                border: 1,
                                borderColor: selected ? "primary.main" : "divider",
                                bgcolor: selected ? "action.selected" : "transparent",
                            }}
                        >
                            <Stack direction="row" spacing={1} alignItems="flex-start">
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{
                                        fontVariantNumeric: "tabular-nums",
                                        mt: 0.35,
                                        minWidth: 18,
                                    }}
                                >
                                    {index + 1}
                                </Typography>
                                <Box sx={{ color: statusAccent(stage.status), mt: 0.15 }}>
                                    <StatusIcon status={stage.status} size="small" />
                                </Box>
                                <Box sx={{ minWidth: 0, flex: 1 }}>
                                    <Stack
                                        direction="row"
                                        spacing={1}
                                        alignItems="center"
                                        flexWrap="wrap"
                                        useFlexGap
                                    >
                                        <Typography
                                            variant="subtitle2"
                                            sx={{
                                                fontWeight: selected ? 700 : 600,
                                                color:
                                                    stage.status === "blocked"
                                                        ? "text.disabled"
                                                        : "text.primary",
                                            }}
                                        >
                                            {stage.label}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {STATUS_LABEL[stage.status]}
                                        </Typography>
                                    </Stack>
                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                        sx={{ mt: 0.35 }}
                                    >
                                        {stage.blockedReason ?? stage.detail ?? stage.description}
                                    </Typography>
                                    {showAction ? (
                                        <Button
                                            size="small"
                                            variant={
                                                stage.status === "current" ||
                                                stage.status === "warning"
                                                    ? "contained"
                                                    : "outlined"
                                            }
                                            sx={{ mt: 1 }}
                                            onClick={() => {
                                                onSelectStage(stage);
                                                onClose();
                                            }}
                                        >
                                            {stageActionLabel(stage)}
                                        </Button>
                                    ) : (
                                        <Button
                                            size="small"
                                            variant="text"
                                            sx={{ mt: 0.75, px: 0 }}
                                            disabled={stage.status === "blocked"}
                                            onClick={() => {
                                                if (stage.status === "blocked") return;
                                                onSelectStage(stage);
                                                onClose();
                                            }}
                                        >
                                            Open {stage.label.toLowerCase()} →
                                        </Button>
                                    )}
                                </Box>
                            </Stack>
                        </Box>
                    );
                })}
            </Stack>
        </Drawer>
    );
}
