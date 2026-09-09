import {
    Box,
    Button,
    Chip,
    Divider,
    Skeleton,
    Stack,
    Tab,
    Tabs,
    Typography,
} from "@mui/material";
import {
    Add as AddIcon,
    PlaylistAddCheck as TaskIcon,
    ViewKanban as BoardIcon,
    ViewList as ListIcon,
} from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import { EmptyState } from "../../../components/ui/EmptyState";
import { SectionCard } from "../../../components/ui/SectionCard";
import type { ProjectTask } from "../../../api/projects";
import type { ProjectDetailModel } from "../hooks/useProjectDetail";
import { TASK_STATUS_OPTIONS, type TaskView } from "../projectTaskModel";
import { TaskBoardCard, TaskListCard } from "./ProjectTaskCards";

type ProjectFlowSectionProps = {
    model: ProjectDetailModel;
    onNewTask: () => void;
    onSelectTask: (task: ProjectTask) => void;
};

export function ProjectFlowSection({
    model: m,
    onNewTask,
    onSelectTask,
}: ProjectFlowSectionProps) {
    return (
        <SectionCard
            title="Project flow"
            description="Switch between a sortable Kanban board and a detailed list view."
            action={
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<AddIcon />}
                        onClick={onNewTask}
                    >
                        New task
                    </Button>
                    <Tabs
                        value={m.taskView}
                        onChange={(_, value: TaskView) => m.setTaskView(value)}
                        sx={{ minHeight: "auto" }}
                    >
                        <Tab
                            value="board"
                            icon={<BoardIcon fontSize="small" />}
                            iconPosition="start"
                            label="Board"
                        />
                        <Tab
                            value="list"
                            icon={<ListIcon fontSize="small" />}
                            iconPosition="start"
                            label="List"
                        />
                    </Tabs>
                </Stack>
            }
        >
            {m.tasksQuery.isLoading ? (
                <Stack spacing={1.5}>
                    <Skeleton variant="rounded" height={280} sx={{ borderRadius: 3 }} />
                    <Skeleton variant="rounded" height={280} sx={{ borderRadius: 3 }} />
                </Stack>
            ) : m.orderedTasks.length === 0 ? (
                <EmptyState
                    icon={<TaskIcon />}
                    title="No tasks yet"
                    description="Use New task to add the first item to this project flow."
                    action={
                        <Button variant="contained" startIcon={<AddIcon />} onClick={onNewTask}>
                            New task
                        </Button>
                    }
                />
            ) : m.taskView === "list" ? (
                <Stack spacing={1.5}>
                    {m.orderedTasks.map((task) => (
                        <TaskListCard
                            key={task.id}
                            task={task}
                            selected={task.id === m.selectedTaskId}
                            onSelect={() => onSelectTask(task)}
                        />
                    ))}
                </Stack>
            ) : (
                <Box
                    sx={{
                        overflowX: "auto",
                        WebkitOverflowScrolling: "touch",
                        pb: 1,
                    }}
                >
                    <Box
                        sx={{
                            display: "grid",
                            gap: 1.5,
                            minWidth: 1280,
                            gridTemplateColumns: "repeat(5, minmax(240px, 1fr))",
                        }}
                    >
                        {TASK_STATUS_OPTIONS.map((option) => {
                            const tasks = m.orderedTasks.filter(
                                (task) => task.status === option.value
                            );
                            return (
                                <Box
                                    key={option.value}
                                    onDragOver={(e) => e.preventDefault()}
                                    onDrop={(e) => {
                                        e.preventDefault();
                                        m.drop(option.value);
                                    }}
                                    sx={(t) => ({
                                        minHeight: 240,
                                        p: 1.25,
                                        borderRadius: 4,
                                        border: `1px solid ${t.palette.divider}`,
                                        backgroundColor: alpha(
                                            t.palette.background.paper,
                                            t.palette.mode === "dark" ? 0.82 : 0.72
                                        ),
                                    })}
                                >
                                    <Stack spacing={1.25}>
                                        <Stack direction="row" justifyContent="space-between">
                                            <Typography variant="subtitle2">
                                                {option.label}
                                            </Typography>
                                            <Chip
                                                label={tasks.length}
                                                size="small"
                                                variant="outlined"
                                            />
                                        </Stack>
                                        <Divider />
                                        <Stack spacing={1}>
                                            {tasks.length ? (
                                                tasks.map((task) => (
                                                    <TaskBoardCard
                                                        key={task.id}
                                                        task={task}
                                                        selected={task.id === m.selectedTaskId}
                                                        isDragging={m.draggingTaskId === task.id}
                                                        onSelect={() => onSelectTask(task)}
                                                        onDragStart={() =>
                                                            m.setDraggingTaskId(task.id)
                                                        }
                                                        onDropBefore={() =>
                                                            m.drop(option.value, task.id)
                                                        }
                                                        onMove={(status) =>
                                                            m.drop(status, undefined, task.id)
                                                        }
                                                    />
                                                ))
                                            ) : (
                                                <Box
                                                    sx={(t) => ({
                                                        p: 2,
                                                        borderRadius: 3,
                                                        border: `1px dashed ${t.palette.divider}`,
                                                        textAlign: "center",
                                                        color: "text.secondary",
                                                    })}
                                                >
                                                    <Typography variant="body2">
                                                        Drop tasks here
                                                    </Typography>
                                                </Box>
                                            )}
                                        </Stack>
                                    </Stack>
                                </Box>
                            );
                        })}
                    </Box>
                </Box>
            )}
        </SectionCard>
    );
}
