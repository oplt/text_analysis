import { Alert, Box, Button, Skeleton, Stack } from "@mui/material";
import { ArrowBack as BackIcon, AssignmentTurnedIn as DoneIcon, CalendarMonth as CalendarIcon, PlaylistAddCheck as TaskIcon, Science as LabIcon, ViewKanban as BoardIcon } from "@mui/icons-material";
import { PageShell } from "../../../components/ui/PageShell";
import { StatCard } from "../../../components/ui/StatCard";
import { StatCardSkeletonGrid } from "../../../components/ui/StatCardSkeletonGrid";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { ProjectFlowSection } from "../components/ProjectFlowSection";
import { TaskEditorSection } from "../components/TaskEditorSection";
import { useProjectDetail } from "../hooks/useProjectDetail";

export default function ProjectDetailPage() {
    const m = useProjectDetail();
    if (m.projectQuery.isLoading) return <PageShell maxWidth="xl"><Stack spacing={2}><Skeleton variant="text" width={280} height={48} /><Skeleton variant="text" width="60%" height={28} /><StatCardSkeletonGrid /><Skeleton variant="rounded" height={420} sx={{ borderRadius: 3 }} /></Stack></PageShell>;
    if (!m.projectQuery.data) return <PageShell maxWidth="xl"><Alert severity="error">{getQueryErrorMessage(m.projectQuery.error, "Project not found.")}</Alert></PageShell>;
    const done = m.orderedTasks.filter((task) => task.status === "done").length;
    const flight = m.orderedTasks.filter((task) => ["todo", "in_progress", "review"].includes(task.status)).length;
    const overdue = m.orderedTasks.filter((task) => task.status !== "done" && task.due_date && new Date(`${task.due_date}T23:59:59`) < new Date()).length;
    return <PageShell maxWidth="xl"><Stack direction="row" spacing={1} sx={{ mb: 2 }}><Button variant="outlined" startIcon={<BackIcon />} onClick={() => m.navigate("/projects")}>Back to projects</Button><Button variant="contained" color="secondary" startIcon={<LabIcon />} onClick={() => m.navigate(`/research/${m.projectId}/dashboard`)}>Open Policy Text Lab</Button></Stack>
        <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", xl: "repeat(4, minmax(0, 1fr))" } }}><StatCard label="Total tasks" value={m.orderedTasks.length} description="All tracked work items inside this project" icon={<TaskIcon />} /><StatCard label="In flight" value={flight} description="Tasks currently being worked, reviewed, or queued next" icon={<BoardIcon />} color="warning" /><StatCard label="Completed" value={done} description="Tasks already moved to done" icon={<DoneIcon />} color="success" /><StatCard label="Overdue" value={overdue} description="Open tasks with a due date already behind today" icon={<CalendarIcon />} color={overdue > 0 ? "error" : "secondary"} /></Box>
        {(m.projectQuery.error || m.tasksQuery.error) && <Alert severity="error">{getQueryErrorMessage(m.projectQuery.error ?? m.tasksQuery.error, "Failed to load project workspace.")}</Alert>}
        <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", lg: "360px minmax(0, 1fr)" }, alignItems: "start" }}><TaskEditorSection model={m} /><ProjectFlowSection model={m} /></Box>
    </PageShell>;
}
