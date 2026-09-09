import { useState } from "react";
import {
    Alert,
    Box,
    Button,
    Drawer,
    IconButton,
    Skeleton,
    Stack,
    Typography,
} from "@mui/material";
import {
    ArrowBack as BackIcon,
    AssignmentTurnedIn as DoneIcon,
    CalendarMonth as CalendarIcon,
    Close as CloseIcon,
    PlaylistAddCheck as TaskIcon,
    Science as LabIcon,
    ViewKanban as BoardIcon,
} from "@mui/icons-material";
import type { ProjectTask } from "../../../api/projects";
import { PageShell } from "../../../components/ui/PageShell";
import { StatCard } from "../../../components/ui/StatCard";
import { StatCardSkeletonGrid } from "../../../components/ui/StatCardSkeletonGrid";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ProjectFlowSection } from "../components/ProjectFlowSection";
import { TaskEditorSection } from "../components/TaskEditorSection";
import { useProjectDetail } from "../hooks/useProjectDetail";

export default function ProjectDetailPage() {
    const m = useProjectDetail();
    const [taskDrawerOpen, setTaskDrawerOpen] = useState(false);

    if (m.projectQuery.isLoading) {
        return (
            <PageShell maxWidth="xl">
                <Stack spacing={2}>
                    <Skeleton variant="text" width={280} height={48} />
                    <Skeleton variant="text" width="60%" height={28} />
                    <StatCardSkeletonGrid />
                    <Skeleton variant="rounded" height={420} sx={{ borderRadius: 3 }} />
                </Stack>
            </PageShell>
        );
    }

    if (!m.projectQuery.data) {
        return (
            <PageShell maxWidth="xl">
                <Alert severity="error">
                    {getQueryErrorMessage(m.projectQuery.error, "Project not found.")}
                </Alert>
            </PageShell>
        );
    }

    const done = m.orderedTasks.filter((task) => task.status === "done").length;
    const flight = m.orderedTasks.filter((task) =>
        ["todo", "in_progress", "review"].includes(task.status)
    ).length;
    const overdue = m.orderedTasks.filter(
        (task) =>
            task.status !== "done" &&
            task.due_date &&
            new Date(`${task.due_date}T23:59:59`) < new Date()
    ).length;

    function openNewTask() {
        m.createMutation.reset();
        m.deleteMutation.reset();
        m.reset();
        setTaskDrawerOpen(true);
    }

    function openTask(task: ProjectTask) {
        m.createMutation.reset();
        m.deleteMutation.reset();
        m.select(task);
        setTaskDrawerOpen(true);
    }

    function closeTaskDrawer() {
        setTaskDrawerOpen(false);
    }

    return (
        <PageShell maxWidth="xl">
            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <Button
                    variant="outlined"
                    startIcon={<BackIcon />}
                    onClick={() => m.navigate("/projects")}
                >
                    Back to projects
                </Button>
                <Button
                    variant="contained"
                    color="secondary"
                    startIcon={<LabIcon />}
                    onClick={() => m.navigate(`/research/${m.projectId}/dashboard`)}
                >
                    Open Text Research
                </Button>
            </Stack>

            <Box
                sx={{
                    display: "grid",
                    gap: 2,
                    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                    mb: 2,
                }}
            >
                <StatCard
                    label="Total tasks"
                    value={m.orderedTasks.length}
                    description="All tracked work items inside this project"
                    icon={<TaskIcon />}
                />
                <StatCard
                    label="In flight"
                    value={flight}
                    description="Tasks currently being worked, reviewed, or queued next"
                    icon={<BoardIcon />}
                    color="warning"
                />
                <StatCard
                    label="Completed"
                    value={done}
                    description="Tasks already moved to done"
                    icon={<DoneIcon />}
                    color="success"
                />
                <StatCard
                    label="Overdue"
                    value={overdue}
                    description="Open tasks with a due date already behind today"
                    icon={<CalendarIcon />}
                    color={overdue > 0 ? "error" : "secondary"}
                />
            </Box>

            {(m.projectQuery.error || m.tasksQuery.error) && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {getQueryErrorMessage(
                        m.projectQuery.error ?? m.tasksQuery.error,
                        "Failed to load project workspace."
                    )}
                </Alert>
            )}

            <ProjectFlowSection model={m} onNewTask={openNewTask} onSelectTask={openTask} />

            <Drawer
                anchor="right"
                open={
                    taskDrawerOpen && !m.createMutation.isSuccess && !m.deleteMutation.isSuccess
                }
                onClose={closeTaskDrawer}
                PaperProps={{
                    sx: {
                        width: { xs: "100%", sm: 420 },
                    },
                }}
            >
                <Box
                    sx={{
                        p: 2.5,
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        height: "100%",
                        overflow: "auto",
                    }}
                >
                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Typography variant="h6">
                            {m.selectedTask ? "Edit task" : "Create task"}
                        </Typography>
                        <IconButton
                            aria-label="Close task editor"
                            onClick={closeTaskDrawer}
                            edge="end"
                        >
                            <CloseIcon />
                        </IconButton>
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                        {m.selectedTask
                            ? "Update task details, assignee, due date, or status."
                            : "Add the next task and place it directly into the workflow."}
                    </Typography>
                    <TaskEditorSection model={m} embedded />
                </Box>
            </Drawer>
        </PageShell>
    );
}
