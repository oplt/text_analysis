import { z } from "zod";
import type { ProjectTask, ProjectTaskStatus } from "../../api/projects";
import { PROJECT_TASK_PRIORITIES } from "./taskOptions";

export const projectTaskSchema = z.object({
    title: z.string().trim().min(2, "Task title must be at least 2 characters.").max(255),
    description: z.string().max(5000),
    status: z.enum(["backlog", "todo", "in_progress", "review", "done"]),
    priority: z.enum(["low", "medium", "high", "urgent"]),
    due_date: z.string(),
    assignee_id: z.string(),
});
export type TaskDraft = z.infer<typeof projectTaskSchema>;
export type TaskView = "board" | "list";
export const EMPTY_TASK_DRAFT: TaskDraft = { title: "", description: "", status: "backlog", priority: "medium", due_date: "", assignee_id: "" };
export const TASK_STATUS_OPTIONS: Array<{ value: ProjectTaskStatus; label: string }> = [
    { value: "backlog", label: "Backlog" }, { value: "todo", label: "Todo" },
    { value: "in_progress", label: "In progress" }, { value: "review", label: "Review" }, { value: "done", label: "Done" },
];
export const TASK_PRIORITY_OPTIONS = PROJECT_TASK_PRIORITIES.map((value) => ({ value, label: value[0].toUpperCase() + value.slice(1) }));
export function sortTasks(tasks: ProjectTask[]) {
    const order: Record<ProjectTaskStatus, number> = { backlog: 0, todo: 1, in_progress: 2, review: 3, done: 4 };
    return [...tasks].sort((a, b) => order[a.status] - order[b.status] || a.position - b.position || a.created_at.localeCompare(b.created_at));
}
export function taskToDraft(task: ProjectTask): TaskDraft {
    return { title: task.title, description: task.description ?? "", status: task.status, priority: task.priority,
        due_date: task.due_date ?? "", assignee_id: task.assignee?.id ?? "" };
}
