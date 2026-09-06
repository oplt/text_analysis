import { Alert, Box, Button, Divider, Drawer, MenuItem, Stack, TextField, Typography } from "@mui/material";
import type { CalendarItemType } from "../../../api/calendar";
import type { ProjectTaskPriority } from "../../../api/projects";
import { PROJECT_TASK_PRIORITIES } from "../../projects/taskOptions";
import { formatDateOnly, humanizeKey } from "../../../utils/formatters";
import { ITEM_TYPE_OPTIONS } from "../calendarDisplay";
import { DayItems } from "./DayItems";
import type { DashboardCalendarModel } from "../hooks/useDashboardCalendar";

export function CalendarItemDrawer({ m }: { m: DashboardCalendarModel }) {
    const { drawerOpen, setDrawerOpen, selectedDateKey, draft, setDraft, fieldErrors, setFieldErrors, projects, projectsLoading, onOpenProjects, createItemMutation, submitDraft, itemsByDate } = m;
    return (
    <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
            sx: {
                width: { xs: "100%", sm: 420 },
                p: 2.5,
            },
        }}
    >
        <Stack spacing={2}>
            <Box>
                <Typography variant="h5" sx={{ mb: 0.5 }}>
                    {selectedDateKey ? formatDateOnly(selectedDateKey) : "Select a day"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Add an event, appointment, or a real task due on this day.
                </Typography>
            </Box>

            <Divider />

            <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Day agenda
                </Typography>
                <DayItems
                    items={selectedDateKey ? itemsByDate[selectedDateKey] ?? [] : []}
                    emptyTitle="Nothing scheduled yet"
                    emptyDescription="Pick a type below and add the first calendar item for this day."
                />
            </Box>

            <Divider />

            <Stack spacing={1.5}>
                <Typography variant="subtitle2">Add new item</Typography>
                <TextField
                    label="Type"
                    select
                    value={draft.type}
                    onChange={(event) => {
                        const nextType = event.target.value as CalendarItemType;
                        setDraft((current) => ({
                            ...current,
                            type: nextType,
                            start_time: nextType === "task" ? "" : current.start_time,
                            end_time: nextType === "task" ? "" : current.end_time,
                            project_id: current.project_id || projects[0]?.id || "",
                        }));
                        setFieldErrors({});
                    }}
                    fullWidth
                >
                    {ITEM_TYPE_OPTIONS.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                            {option.label}
                        </MenuItem>
                    ))}
                </TextField>
                <TextField
                    label="Title"
                    value={draft.title}
                    onChange={(event) => {
                        setDraft((current) => ({ ...current, title: event.target.value }));
                        setFieldErrors((current) => ({ ...current, title: undefined }));
                    }}
                    error={Boolean(fieldErrors.title)}
                    helperText={fieldErrors.title}
                    fullWidth
                />
                <TextField
                    label="Description"
                    value={draft.description}
                    onChange={(event) =>
                        setDraft((current) => ({ ...current, description: event.target.value }))
                    }
                    fullWidth
                    multiline
                    minRows={3}
                />

                {draft.type === "task" ? (
                    <>
                        <TextField
                            label="Project"
                            select
                            value={draft.project_id}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, project_id: event.target.value }));
                                setFieldErrors((current) => ({ ...current, project_id: undefined }));
                            }}
                            fullWidth
                            disabled={projectsLoading}
                            error={Boolean(fieldErrors.project_id)}
                            helperText={
                                fieldErrors.project_id ??
                                (projects.length > 0
                                    ? "Task will be created in Todo with this date as its due date."
                                    : "Create a project first before scheduling tasks from the calendar.")
                            }
                        >
                            {projects.map((project) => (
                                <MenuItem key={project.id} value={project.id}>
                                    {project.name}
                                </MenuItem>
                            ))}
                        </TextField>
                        <TextField
                            label="Priority"
                            select
                            value={draft.priority}
                            onChange={(event) =>
                                setDraft((current) => ({
                                    ...current,
                                    priority: event.target.value as ProjectTaskPriority,
                                }))
                            }
                            fullWidth
                        >
                            {PROJECT_TASK_PRIORITIES.map((priority) => (
                                <MenuItem key={priority} value={priority}>
                                    {humanizeKey(priority)}
                                </MenuItem>
                            ))}
                        </TextField>
                        {projects.length === 0 && (
                            <Button variant="outlined" onClick={onOpenProjects}>
                                Create a project
                            </Button>
                        )}
                    </>
                ) : (
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1.25}>
                        <TextField
                            label="Start time"
                            type="time"
                            value={draft.start_time}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, start_time: event.target.value }));
                                setFieldErrors((current) => ({
                                    ...current,
                                    start_time: undefined,
                                    end_time: undefined,
                                }));
                            }}
                            error={Boolean(fieldErrors.start_time)}
                            helperText={fieldErrors.start_time}
                            fullWidth
                            InputLabelProps={{ shrink: true }}
                        />
                        <TextField
                            label="End time"
                            type="time"
                            value={draft.end_time}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, end_time: event.target.value }));
                                setFieldErrors((current) => ({ ...current, end_time: undefined }));
                            }}
                            error={Boolean(fieldErrors.end_time)}
                            helperText={fieldErrors.end_time}
                            fullWidth
                            InputLabelProps={{ shrink: true }}
                        />
                    </Stack>
                )}

                {fieldErrors.general && <Alert severity="error">{fieldErrors.general}</Alert>}

                <Stack direction="row" spacing={1}>
                    <Button variant="outlined" onClick={() => setDrawerOpen(false)} fullWidth>
                        Close
                    </Button>
                    <Button
                        variant="contained"
                        onClick={submitDraft}
                        disabled={
                            createItemMutation.isPending ||
                            (draft.type === "task" && projects.length === 0)
                        }
                        fullWidth
                    >
                        {createItemMutation.isPending ? "Saving..." : "Save"}
                    </Button>
                </Stack>
            </Stack>
        </Stack>
    </Drawer>
    );
}
