import type { ProjectTaskPriority } from "../../api/projects";

export const PROJECT_TASK_PRIORITIES = [
    "low",
    "medium",
    "high",
    "urgent",
] as const satisfies readonly ProjectTaskPriority[];
