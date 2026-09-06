import { useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { createProjectTask, deleteProjectTask, getProject, listProjectTasks, reorderProjectTasks, updateProjectTask, type ProjectTask, type ProjectTaskStatus } from "../../../api/projects";
import { listUserDirectory } from "../../../api/users";
import { useSnackbar } from "../../../app/snackbarContext";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { EMPTY_TASK_DRAFT, projectTaskSchema, sortTasks, taskToDraft, TASK_STATUS_OPTIONS, type TaskDraft, type TaskView } from "../projectTaskModel";

export function useProjectDetail() {
    const { projectId = "" } = useParams(); const navigate = useNavigate(); const client = useQueryClient(); const { showToast } = useSnackbar();
    const [taskView, setTaskView] = useState<TaskView>("board"); const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
    const taskForm = useForm<TaskDraft>({ resolver: zodResolver(projectTaskSchema), defaultValues: EMPTY_TASK_DRAFT, mode: "onBlur" });
    const taskDraft = useWatch({ control: taskForm.control }); const [taskError, setTaskError] = useState("");
    const [draggingTaskId, setDraggingTaskId] = useState<string | null>(null);
    const projectQuery = useQuery({ queryKey: queryKeys.projects.detail(projectId), queryFn: () => getProject(projectId), enabled: Boolean(projectId) });
    const tasksQuery = useQuery({ queryKey: queryKeys.projects.tasks(projectId), queryFn: () => listProjectTasks(projectId), enabled: Boolean(projectId) });
    const usersQuery = useQuery({ queryKey: queryKeys.users.directory, queryFn: listUserDirectory, staleTime: QUERY_STALE_TIMES.userDirectory });
    const orderedTasks = sortTasks(tasksQuery.data ?? []); const selectedTask = orderedTasks.find((task) => task.id === selectedTaskId) ?? null;
    const reset = () => { setSelectedTaskId(null); taskForm.reset(EMPTY_TASK_DRAFT); setTaskError(""); };
    const select = (task: ProjectTask) => { setSelectedTaskId(task.id); taskForm.reset(taskToDraft(task)); setTaskError(""); };
    const saveCache = (task: ProjectTask) => client.setQueryData<ProjectTask[]>(queryKeys.projects.tasks(projectId), (current) => sortTasks((current ?? []).filter((item) => item.id !== task.id).concat(task)));
    const createMutation = useMutation({ mutationFn: () => { const draft = taskForm.getValues(); return createProjectTask(projectId, { title: draft.title.trim(), description: draft.description.trim() || undefined, status: draft.status, priority: draft.priority, due_date: draft.due_date || null, assignee_id: draft.assignee_id || null }); },
        onSuccess: (task) => { saveCache(task); reset(); void client.invalidateQueries({ queryKey: queryKeys.projects.all }); showToast({ message: "Task created.", severity: "success" }); },
        onError: (error) => setTaskError(getQueryErrorMessage(error, "Failed to create task.")) });
    const updateMutation = useMutation({ mutationFn: () => { const draft = taskForm.getValues(); return updateProjectTask(projectId, selectedTaskId ?? "", { title: draft.title.trim(), description: draft.description.trim() || null, status: draft.status, priority: draft.priority, due_date: draft.due_date || null, assignee_id: draft.assignee_id || null }); },
        onSuccess: (task) => { saveCache(task); select(task); showToast({ message: "Task updated.", severity: "success" }); },
        onError: (error) => setTaskError(getQueryErrorMessage(error, "Failed to update task.")) });
    const deleteMutation = useMutation({ mutationFn: () => deleteProjectTask(projectId, selectedTaskId ?? ""), onSuccess: () => { client.setQueryData<ProjectTask[]>(queryKeys.projects.tasks(projectId), (current) => (current ?? []).filter((task) => task.id !== selectedTaskId)); reset(); showToast({ message: "Task deleted.", severity: "success" }); }, onError: (error) => setTaskError(getQueryErrorMessage(error, "Failed to delete task.")) });
    const reorderMutation = useMutation({ mutationFn: (payload: { columns: Array<{ status: ProjectTaskStatus; task_ids: string[] }> }) => reorderProjectTasks(projectId, payload), onSuccess: (tasks) => client.setQueryData(queryKeys.projects.tasks(projectId), sortTasks(tasks)), onError: () => { void client.invalidateQueries({ queryKey: queryKeys.projects.tasks(projectId) }); showToast({ message: "Failed to reorder tasks.", severity: "error" }); } });
    const submit = taskForm.handleSubmit(() => { if (selectedTaskId) updateMutation.mutate(); else createMutation.mutate(); });
    const drop = (status: ProjectTaskStatus, beforeId?: string, overrideId?: string) => {
        const activeId = overrideId ?? draggingTaskId; const tasks = tasksQuery.data; if (!activeId || !tasks || reorderMutation.isPending) return;
        if (beforeId === activeId) { setDraggingTaskId(null); return; } const active = tasks.find((task) => task.id === activeId); if (!active) return;
        const columns = Object.fromEntries(TASK_STATUS_OPTIONS.map((option) => [option.value, [] as ProjectTask[]])) as Record<ProjectTaskStatus, ProjectTask[]>;
        tasks.forEach((task) => { if (task.id !== activeId) columns[task.status].push(task); }); const target = columns[status]; const index = beforeId ? target.findIndex((task) => task.id === beforeId) : -1;
        target.splice(index < 0 ? target.length : index, 0, { ...active, status }); const next = sortTasks(TASK_STATUS_OPTIONS.flatMap((option) => columns[option.value].map((task, position) => ({ ...task, status: option.value, position }))));
        client.setQueryData(queryKeys.projects.tasks(projectId), next); reorderMutation.mutate({ columns: TASK_STATUS_OPTIONS.map((option) => ({ status: option.value, task_ids: next.filter((task) => task.status === option.value).sort((a, b) => a.position - b.position).map((task) => task.id) })) }); setDraggingTaskId(null);
    };
    return { projectId, navigate, projectQuery, tasksQuery, usersQuery, orderedTasks, selectedTask, selectedTaskId, taskView, setTaskView,
        taskDraft, taskForm, taskError, draggingTaskId, setDraggingTaskId,
        createMutation, updateMutation, deleteMutation, reorderMutation, reset, select, submit, drop };
}
export type ProjectDetailModel = ReturnType<typeof useProjectDetail>;
