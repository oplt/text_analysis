import { Alert, Button, MenuItem, Stack, TextField } from "@mui/material";
import { DeleteOutline as DeleteIcon } from "@mui/icons-material";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { formatDateTime } from "../../../utils/formatters";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import type { ProjectDetailModel } from "../hooks/useProjectDetail";
import { TASK_PRIORITY_OPTIONS, TASK_STATUS_OPTIONS } from "../projectTaskModel";

type TaskEditorSectionProps = {
    model: ProjectDetailModel;
    /** When true, render the form only (drawer provides chrome). */
    embedded?: boolean;
};

export function TaskEditorSection({ model: m, embedded = false }: TaskEditorSectionProps) {
    const d = m.taskDraft;
    const form = (
        <Stack spacing={2}>
            <TextField
                label="Title"
                {...m.taskForm.register("title")}
                error={Boolean(m.taskForm.formState.errors.title)}
                helperText={m.taskForm.formState.errors.title?.message}
            />
            <TextField
                label="Description"
                {...m.taskForm.register("description")}
                error={Boolean(m.taskForm.formState.errors.description)}
                helperText={m.taskForm.formState.errors.description?.message}
                multiline
                minRows={4}
            />
            <TextField label="Status" select value={d.status} {...m.taskForm.register("status")}>
                {TASK_STATUS_OPTIONS.map((o) => (
                    <MenuItem key={o.value} value={o.value}>
                        {o.label}
                    </MenuItem>
                ))}
            </TextField>
            <TextField
                label="Priority"
                select
                value={d.priority}
                {...m.taskForm.register("priority")}
            >
                {TASK_PRIORITY_OPTIONS.map((o) => (
                    <MenuItem key={o.value} value={o.value}>
                        {o.label}
                    </MenuItem>
                ))}
            </TextField>
            <TextField
                label="Due date"
                type="date"
                {...m.taskForm.register("due_date")}
                InputLabelProps={{ shrink: true }}
            />
            <TextField
                label="Assignee"
                select
                value={d.assignee_id}
                {...m.taskForm.register("assignee_id")}
                error={m.usersQuery.isError}
                helperText={
                    m.usersQuery.isError
                        ? getQueryErrorMessage(m.usersQuery.error, "Failed to load user directory.")
                        : undefined
                }
            >
                <MenuItem value="">Unassigned</MenuItem>
                {(m.usersQuery.data ?? []).map((u) => (
                    <MenuItem key={u.id} value={u.id}>
                        {u.full_name || u.email}
                    </MenuItem>
                ))}
            </TextField>
            {m.usersQuery.isError && (
                <QueryErrorAlert
                    error={m.usersQuery.error}
                    fallback="Failed to load assignee directory."
                    onRetry={() => void m.usersQuery.refetch()}
                />
            )}
            {m.selectedTask && (
                <Alert severity="info">
                    Created {formatDateTime(m.selectedTask.created_at)}. Last updated{" "}
                    {formatDateTime(m.selectedTask.updated_at)}.
                </Alert>
            )}
            {m.taskError && <Alert severity="error">{m.taskError}</Alert>}
            <Stack direction="row" spacing={1.25}>
                <Button
                    variant="contained"
                    onClick={m.submit}
                    disabled={m.createMutation.isPending || m.updateMutation.isPending}
                    fullWidth
                >
                    {m.selectedTask
                        ? m.updateMutation.isPending
                            ? "Saving..."
                            : "Save task"
                        : m.createMutation.isPending
                          ? "Creating..."
                          : "Create task"}
                </Button>
                {m.selectedTask && (
                    <Button
                        color="error"
                        variant="outlined"
                        onClick={() => m.deleteMutation.mutate()}
                        disabled={m.deleteMutation.isPending}
                        startIcon={<DeleteIcon />}
                    >
                        {m.deleteMutation.isPending ? "Deleting..." : "Delete"}
                    </Button>
                )}
            </Stack>
        </Stack>
    );

    if (embedded) return form;

    return (
        <SectionCard
            title={m.selectedTask ? "Edit task" : "Create task"}
            description={
                m.selectedTask
                    ? "Update task details, assignee, due date, or status."
                    : "Add the next task and place it directly into the workflow."
            }
        >
            {form}
        </SectionCard>
    );
}
